"""
Offline specialized KG extraction service.

This module replaces generic LLM-based extraction with two local NLP models:
- GLiNER: entity extraction
- mREBEL: relation extraction

Output format intentionally mirrors LightRAG's extraction format:
  entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
  relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description
  <|COMPLETE|>
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from threading import Lock
from typing import Sequence

import torch
from gliner import GLiNER
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.core.config import settings

logger = logging.getLogger(__name__)

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"


@dataclass(frozen=True)
class ExtractedEntity:
    name: str
    entity_type: str
    description: str
    score: float = 0.0


@dataclass(frozen=True)
class ExtractedRelation:
    source: str
    target: str
    relation_type: str
    description: str


class SpecializedKGExtractor:
    """Singleton extractor that keeps GLiNER + mREBEL loaded in memory."""

    _instance: SpecializedKGExtractor | None = None
    _instance_lock: Lock = Lock()

    _LANGUAGE_TO_MREBEL_CODE = {
        "english": "en_XX",
        "vietnamese": "vi_VN",
        "french": "fr_XX",
        "german": "de_DE",
        "spanish": "es_XX",
        "italian": "it_IT",
        "portuguese": "pt_XX",
        "russian": "ru_RU",
    }

    def __init__(self) -> None:
        self._gliner_model_name = settings.NEXUSRAG_KG_GLINER_MODEL
        self._mrebel_model_name = settings.NEXUSRAG_KG_RELATION_MODEL

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._gliner_model: GLiNER | None = None
        self._mrebel_tokenizer = None
        self._mrebel_model = None
        self._decoder_start_token_id: int | None = None
        self._inference_lock: Lock = Lock()

        self._initialize_models()

    @classmethod
    def get_instance(cls) -> SpecializedKGExtractor:
        """Return a single process-wide instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def extract_entities_and_relations(self, text: str) -> str:
        """
        Async wrapper for CPU/GPU-bound extraction.

        This method is safe to call from async request handlers.
        """
        return await asyncio.to_thread(self.extract_entities_and_relations_sync, text)

    def extract_entities_and_relations_sync(
        self,
        text: str,
        entity_types: Sequence[str] | None = None,
    ) -> str:
        """Run extraction in blocking mode and return LightRAG-compatible text."""
        clean_text = text.strip()
        if not clean_text:
            return COMPLETION_DELIMITER

        try:
            normalized_entity_types = self._normalize_entity_types(entity_types)
            # Shared tokenizer/model instances are not thread-safe under concurrent calls.
            with self._inference_lock:
                entities = self._extract_entities(clean_text, normalized_entity_types)
                relations = self._extract_relations(clean_text)
            entities = self._ensure_entities_for_relations(entities, relations)
            return self._format_for_lightrag(entities, relations)
        except Exception as exc:
            logger.exception("Specialized KG extraction failed: %s", exc)
            raise

    def _initialize_models(self) -> None:
        """Load cached models only (strict offline behavior at runtime)."""

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        logger.info(
            "Loading specialized KG models from local cache only "
            "(gliner=%s, mrebel=%s, device=%s)",
            self._gliner_model_name,
            self._mrebel_model_name,
            self._device,
        )

        try:
            # GLiNER newer versions
            try:
                self._gliner_model = GLiNER.from_pretrained(
                    self._gliner_model_name,
                    map_location=str(self._device),
                    local_files_only=True,
                )
            except TypeError:
                # Backward compatibility with older GLiNER signatures.
                self._gliner_model = GLiNER.from_pretrained(self._gliner_model_name)
        except Exception as exc:
            raise FileNotFoundError(
                "Unable to load GLiNER from local cache in offline mode. "
                f"Model: {self._gliner_model_name}. "
                "Run: python scripts/download_models.py"
            ) from exc

        try:
            self._mrebel_tokenizer = AutoTokenizer.from_pretrained(
                self._mrebel_model_name,
                use_fast=False,
                local_files_only=True,
            )
            self._mrebel_model = AutoModelForSeq2SeqLM.from_pretrained(
                self._mrebel_model_name,
                local_files_only=True,
            ).to(self._device)
        except Exception as exc:
            raise FileNotFoundError(
                "Unable to load mREBEL from local cache in offline mode. "
                f"Model: {self._mrebel_model_name}. "
                "Run: python scripts/download_models.py"
            ) from exc

        self._mrebel_model.eval()

        src_lang = self._resolve_mrebel_source_language(settings.NEXUSRAG_KG_LANGUAGE)
        if hasattr(self._mrebel_tokenizer, "src_lang"):
            self._mrebel_tokenizer.src_lang = src_lang

        self._decoder_start_token_id = self._mrebel_tokenizer.convert_tokens_to_ids("tp_XX")
        if self._decoder_start_token_id is None or self._decoder_start_token_id < 0:
            raise RuntimeError("Unable to resolve mREBEL decoder start token id for 'tp_XX'")

        logger.info("Specialized KG models loaded successfully")

    def _extract_entities(self, text: str, entity_types: Sequence[str]) -> list[ExtractedEntity]:
        if self._gliner_model is None:
            raise RuntimeError("GLiNER model is not initialized")

        predictions = self._gliner_model.predict_entities(
            text,
            list(entity_types),
            threshold=0.45,
        )

        dedup: dict[str, ExtractedEntity] = {}
        for item in predictions:
            raw_name = self._sanitize_field(item.get("text", ""))
            if not raw_name:
                continue

            raw_type = self._sanitize_entity_type(item.get("label", "Other"))
            score = float(item.get("score", 0.0) or 0.0)

            start = int(item.get("start", 0) or 0)
            end = int(item.get("end", start) or start)
            description = self._build_entity_description(text, start, end, raw_type)

            key = raw_name.casefold()
            current = dedup.get(key)
            if current is None or score > current.score:
                dedup[key] = ExtractedEntity(
                    name=raw_name,
                    entity_type=raw_type,
                    description=description,
                    score=score,
                )

        return list(dedup.values())

    def _extract_relations(self, text: str) -> list[ExtractedRelation]:
        if self._mrebel_model is None or self._mrebel_tokenizer is None:
            raise RuntimeError("mREBEL model/tokenizer is not initialized")

        segments = self._split_text_for_relations(text)
        relation_map: dict[tuple[str, str, str], ExtractedRelation] = {}

        for segment in segments:
            inputs = self._mrebel_tokenizer(
                segment,
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}

            with torch.inference_mode():
                generated = self._mrebel_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=256,
                    num_beams=3,
                    length_penalty=0.0,
                    num_return_sequences=1,
                    decoder_start_token_id=self._decoder_start_token_id,
                    forced_bos_token_id=None,
                )

            decoded_outputs = self._mrebel_tokenizer.batch_decode(
                generated,
                skip_special_tokens=False,
            )

            for raw_output in decoded_outputs:
                for triplet in self._parse_mrebel_triplets(raw_output):
                    source = self._sanitize_field(triplet.get("head", ""))
                    target = self._sanitize_field(triplet.get("tail", ""))
                    relation_type = self._sanitize_relation_type(triplet.get("type", ""))

                    if not source or not target or source.casefold() == target.casefold():
                        continue

                    pair_key = tuple(sorted([source.casefold(), target.casefold()]))
                    dedup_key = (pair_key[0], pair_key[1], relation_type.casefold())
                    if dedup_key in relation_map:
                        continue

                    description = f"{source} {relation_type.replace('_', ' ')} {target}."
                    relation_map[dedup_key] = ExtractedRelation(
                        source=source,
                        target=target,
                        relation_type=relation_type,
                        description=description,
                    )

        return list(relation_map.values())

    def _parse_mrebel_triplets(self, text: str) -> list[dict[str, str]]:
        """
        Parse mREBEL generation format.

        Adapted from Babelscape model-card parser logic.
        """
        triplets: list[dict[str, str]] = []
        current = "x"
        subject, relation, object_ = "", "", ""
        object_type, subject_type = "", ""

        cleaned = (
            text.strip()
            .replace("<s>", "")
            .replace("<pad>", "")
            .replace("</s>", "")
            .replace("tp_XX", "")
            .replace("__en__", "")
        )

        for token in cleaned.split():
            if token in {"<triplet>", "<relation>"}:
                current = "t"
                if relation:
                    triplets.append(
                        {
                            "head": subject.strip(),
                            "head_type": subject_type,
                            "type": relation.strip(),
                            "tail": object_.strip(),
                            "tail_type": object_type,
                        }
                    )
                    relation = ""
                subject = ""
            elif token.startswith("<") and token.endswith(">"):
                if current in {"t", "o"}:
                    current = "s"
                    if relation:
                        triplets.append(
                            {
                                "head": subject.strip(),
                                "head_type": subject_type,
                                "type": relation.strip(),
                                "tail": object_.strip(),
                                "tail_type": object_type,
                            }
                        )
                    object_ = ""
                    subject_type = token[1:-1]
                else:
                    current = "o"
                    object_type = token[1:-1]
                    relation = ""
            else:
                if current == "t":
                    subject += f" {token}"
                elif current == "s":
                    object_ += f" {token}"
                elif current == "o":
                    relation += f" {token}"

        if subject.strip() and relation.strip() and object_.strip():
            triplets.append(
                {
                    "head": subject.strip(),
                    "head_type": subject_type,
                    "type": relation.strip(),
                    "tail": object_.strip(),
                    "tail_type": object_type,
                }
            )

        return triplets

    def _ensure_entities_for_relations(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> list[ExtractedEntity]:
        entity_map = {entity.name.casefold(): entity for entity in entities}

        for relation in relations:
            for name in (relation.source, relation.target):
                key = name.casefold()
                if key not in entity_map:
                    entity_map[key] = ExtractedEntity(
                        name=name,
                        entity_type="Other",
                        description="Entity inferred from relation extraction.",
                        score=0.0,
                    )

        return list(entity_map.values())

    def _format_for_lightrag(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
    ) -> str:
        lines: list[str] = []

        for entity in sorted(entities, key=lambda e: e.name.casefold()):
            lines.append(
                f"entity{TUPLE_DELIMITER}"
                f"{self._sanitize_field(entity.name)}{TUPLE_DELIMITER}"
                f"{self._sanitize_entity_type(entity.entity_type)}{TUPLE_DELIMITER}"
                f"{self._sanitize_field(entity.description)}"
            )

        for relation in sorted(
            relations,
            key=lambda r: (
                r.source.casefold(),
                r.target.casefold(),
                r.relation_type.casefold(),
            ),
        ):
            keywords = self._sanitize_keywords(relation.relation_type)
            lines.append(
                f"relation{TUPLE_DELIMITER}"
                f"{self._sanitize_field(relation.source)}{TUPLE_DELIMITER}"
                f"{self._sanitize_field(relation.target)}{TUPLE_DELIMITER}"
                f"{keywords}{TUPLE_DELIMITER}"
                f"{self._sanitize_field(relation.description)}"
            )

        lines.append(COMPLETION_DELIMITER)
        return "\n".join(lines)

    def _split_text_for_relations(self, text: str, max_chars: int = 900) -> list[str]:
        normalized = " ".join(text.split())
        if len(normalized) <= max_chars:
            return [normalized]

        parts = re.split(r"(?<=[.!?])\s+", normalized)
        chunks: list[str] = []
        current = ""

        for part in parts:
            sentence = part.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                for i in range(0, len(sentence), max_chars):
                    piece = sentence[i : i + max_chars].strip()
                    if piece:
                        chunks.append(piece)
                continue

            candidate = f"{current} {sentence}".strip()
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks if chunks else [normalized[:max_chars]]

    def _build_entity_description(
        self,
        text: str,
        start: int,
        end: int,
        entity_type: str,
    ) -> str:
        left = max(0, start - 64)
        right = min(len(text), end + 64)
        context = " ".join(text[left:right].split())
        if len(context) > 180:
            context = context[:177] + "..."

        if context:
            return f"Mentioned as {entity_type} in context: {context}"
        return f"Mentioned as {entity_type} in the source text."

    def _normalize_entity_types(self, entity_types: Sequence[str] | None) -> list[str]:
        if entity_types:
            cleaned = [self._sanitize_entity_type(item) for item in entity_types]
            cleaned = [item for item in cleaned if item]
            if cleaned:
                return cleaned

        fallback = [self._sanitize_entity_type(item) for item in settings.NEXUSRAG_KG_ENTITY_TYPES]
        fallback = [item for item in fallback if item]
        return fallback or ["Other"]

    def _resolve_mrebel_source_language(self, language: str) -> str:
        if not language:
            return "en_XX"
        return self._LANGUAGE_TO_MREBEL_CODE.get(language.strip().lower(), "en_XX")

    @staticmethod
    def _sanitize_field(value: object) -> str:
        text = str(value or "")
        text = text.replace("\n", " ").replace("\r", " ")
        text = text.replace(TUPLE_DELIMITER, " ")
        text = text.replace(COMPLETION_DELIMITER, " ")
        text = " ".join(text.split())
        return text.strip()

    @staticmethod
    def _sanitize_entity_type(value: object) -> str:
        text = SpecializedKGExtractor._sanitize_field(value)
        return text if text else "Other"

    @staticmethod
    def _sanitize_relation_type(value: object) -> str:
        text = SpecializedKGExtractor._sanitize_field(value)
        if not text:
            return "related_to"
        text = text.replace(" ", "_")
        return text

    @staticmethod
    def _sanitize_keywords(value: object) -> str:
        text = SpecializedKGExtractor._sanitize_field(value)
        text = text.replace("_", " ")
        return text if text else "related to"


def get_specialized_kg_extractor() -> SpecializedKGExtractor:
    """Factory helper for the singleton extractor."""
    return SpecializedKGExtractor.get_instance()