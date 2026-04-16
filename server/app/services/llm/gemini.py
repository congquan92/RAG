"""
Gemini LLM & Embedding Providers
=================================
Concrete implementations using Google's ``google-genai`` SDK.

Supports both Gemini 2.5 (thinking_budget_tokens) and Gemini 3.x+
(thinking_level: minimal | low | medium | high).
"""
from __future__ import annotations

import logging
import re
from typing import AsyncGenerator, Optional

import numpy as np
from google import genai
from google.genai import types

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

# Regex to extract major version from model name: gemini-2.5-flash → 2, gemini-3.1-flash-lite → 3
_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)")


class GeminiLLMProvider(LLMProvider):
    """Google Gemini text/multimodal generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        thinking_level: str = "medium",
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._thinking_level = thinking_level
        self._major_version = self._parse_major_version(model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_major_version(model: str) -> int:
        """Extract major version number from model name (e.g. 'gemini-3.1-flash' → 3)."""
        match = _GEMINI_VERSION_RE.search(model)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _build_parts(msg: LLMMessage) -> list[types.Part]:
        """Convert an LLMMessage into a list of Gemini Part objects."""
        parts: list[types.Part] = []
        if msg.content:
            parts.append(types.Part.from_text(text=msg.content))
        for img in msg.images:
            parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))
        return parts

    def _to_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        """Map a list of LLMMessage into Gemini Content objects.

        System messages are injected as a fake user→model exchange
        (Gemini does not support a native system role in ``contents``).

        If a message carries ``_raw_provider_content`` (a native Gemini
        ``types.Content``), it is used directly — this preserves opaque
        fields like ``thought_signature`` that cannot be reconstructed
        from plain text.
        """
        contents: list[types.Content] = []
        for msg in messages:
            # Raw Gemini Content — use as-is (preserves thought_signature)
            if msg._raw_provider_content is not None:
                contents.append(msg._raw_provider_content)
                continue

            if msg.role == "system":
                # Gemini: system role not allowed in contents → inject as
                # user instruction + model acknowledgement pair.
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text=f"[System Instructions]: {msg.content}",
                    )],
                ))
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(
                        text="Understood. I will follow these instructions.",
                    )],
                ))
            else:
                role = "model" if msg.role == "assistant" else "user"
                contents.append(types.Content(
                    role=role,
                    parts=self._build_parts(msg),
                ))
        return contents

    def _build_thinking_config(self) -> types.ThinkingConfig:
        """Build ThinkingConfig based on model version.

        Gemini 2.5: uses ``thinking_budget_tokens`` (does NOT support thinking_level).
        Gemini 3.x+: uses ``thinking_level`` + ``include_thoughts=True``.
        """
        if self._major_version >= 3:
            return types.ThinkingConfig(
                thinking_level=self._thinking_level,
                include_thoughts=True,
            )
        # Gemini 2.5 — use budget-based thinking
        _BUDGET_MAP = {"minimal": 1024, "low": 2048, "medium": 4096, "high": 8192}
        budget = _BUDGET_MAP.get(self._thinking_level, 4096)
        return types.ThinkingConfig(thinking_budget=budget)

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        contents = self._to_contents(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        use_think = think and self.supports_thinking()
        if use_think:
            config.thinking_config = self._build_thinking_config()

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            if use_think:
                return self._extract_with_thinking(response)
            return response.text or ""
        except Exception as e:
            logger.error(f"Gemini LLM call failed: {e}")
            return LLMResult(content="") if use_think else ""

    @staticmethod
    def _extract_with_thinking(response) -> LLMResult:
        """Extract content and thinking from a Gemini response."""
        content = ""
        thinking = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "thought") and part.thought:
                    thinking += (part.text or "")
                else:
                    content += (part.text or "")
        return LLMResult(content=content, thinking=thinking)

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streaming generation via Gemini's async stream API.

        After streaming completes, ``self.last_response_content`` holds the
        accumulated ``types.Content`` with all parts (including opaque
        ``thought_signature`` fields).  Callers that need to build proper
        multi-turn history (e.g. after a function call) should read this
        attribute and pass it back via ``LLMMessage._raw_provider_content``.
        """
        contents = self._to_contents(messages)

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_prompt:
            config.system_instruction = system_prompt
        if tools:
            config.tools = tools

        use_think = think and self.supports_thinking()
        if use_think:
            config.thinking_config = self._build_thinking_config()

        # Accumulate raw parts so callers can access the full response
        # including thought_signature for proper multi-turn circulation.
        accumulated_parts: list[types.Part] = []

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    accumulated_parts.append(part)

                    if getattr(part, "thought", False):
                        if part.text:
                            yield StreamChunk(type="thinking", text=part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        yield StreamChunk(
                            type="function_call",
                            function_call={
                                "name": fc.name,
                                "args": dict(fc.args) if fc.args else {},
                            },
                        )
                    elif hasattr(part, "text") and part.text:
                        yield StreamChunk(type="text", text=part.text)
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            yield StreamChunk(type="text", text="")
        finally:
            # Store the complete response Content for callers that need
            # thought_signature circulation (Gemini 3 function calling).
            self.last_response_content = types.Content(
                role="model",
                parts=accumulated_parts,
            ) if accumulated_parts else None

    def supports_vision(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        """Gemini 2.5+ and 3.x+ models support thinking."""
        return self._major_version >= 2

    def supports_native_tools(self) -> bool:
        return True


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Google Gemini text embedding (``gemini-embedding-001``, 3072-dim)."""

    _BATCH_SIZE = 100  # Gemini API limit
    _MAX_BATCH_RETRIES = 1

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimension: int | None = None

    @staticmethod
    def _to_embed_contents(texts: list[str]) -> list[types.Content]:
        """Build one Gemini Content per input text (1:1 text→vector contract)."""
        contents: list[types.Content] = []
        for text in texts:
            normalized = text.strip() or "[empty]"
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=normalized)],
            ))
        return contents

    def _embed_one(self, text: str) -> list[float]:
        """Embed a single text and return one vector."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=self._to_embed_contents([text]),
        )
        if not result.embeddings:
            raise ValueError("Gemini embedding returned no vectors for single input")
        values = result.embeddings[0].values
        if self._dimension is None:
            self._dimension = len(values)
        return values

    def _embed_batch_once(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch once and enforce 1:1 input/output count."""
        result = self._client.models.embed_content(
            model=self._model,
            contents=self._to_embed_contents(texts),
        )
        batch_embeddings = [emb.values for emb in result.embeddings]

        if len(batch_embeddings) != len(texts):
            raise ValueError(
                "Embedding count mismatch for batch "
                f"(expected={len(texts)} got={len(batch_embeddings)})"
            )

        if batch_embeddings and self._dimension is None:
            self._dimension = len(batch_embeddings[0])

        return batch_embeddings

    def _embed_batch_resilient(
        self,
        texts: list[str],
        *,
        batch_start: int,
        depth: int = 0,
    ) -> list[list[float]]:
        """Retry full batch, then split recursively to isolate bad items."""
        last_error: Exception | None = None

        for attempt in range(self._MAX_BATCH_RETRIES + 1):
            try:
                return self._embed_batch_once(texts)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Gemini batch embed retry %d/%d failed at batch_start=%d size=%d depth=%d: %s",
                    attempt + 1,
                    self._MAX_BATCH_RETRIES + 1,
                    batch_start,
                    len(texts),
                    depth,
                    e,
                )

        if self._is_non_retriable_error(last_error):
            logger.error(
                "Gemini non-retriable embed error at batch_start=%d size=%d: %s",
                batch_start,
                len(texts),
                last_error,
            )
            return [[0.0] * self.get_dimension() for _ in texts]

        if len(texts) == 1:
            # Leaf fallback: keep 1:1 contract even for stubborn failures.
            try:
                return [self._embed_one(texts[0])]
            except Exception as leaf_error:
                logger.error(
                    "Gemini embedding failed at leaf batch_start=%d: %s (last batch error: %s)",
                    batch_start,
                    leaf_error,
                    last_error,
                )
                return [[0.0] * self.get_dimension()]

        mid = len(texts) // 2
        left = self._embed_batch_resilient(
            texts[:mid], batch_start=batch_start, depth=depth + 1,
        )
        right = self._embed_batch_resilient(
            texts[mid:], batch_start=batch_start + mid, depth=depth + 1,
        )
        return left + right

    @staticmethod
    def _is_non_retriable_error(error: Exception | None) -> bool:
        if error is None:
            return False
        msg = str(error).lower()
        fatal_markers = [
            "api key",
            "permission",
            "unauthorized",
            "forbidden",
            "invalid argument",
            "not found",
            "quota",
        ]
        return any(marker in msg for marker in fatal_markers)

    def _detect_dimension(self) -> int:
        """Probe actual embedding dimension from the active model."""
        from app.core.config import settings

        try:
            dim = len(self._embed_one("dimension probe"))
            logger.info("Detected Gemini embedding dimension: %d for model %s", dim, self._model)
            return dim
        except Exception as e:
            logger.warning(
                "Failed to detect Gemini embedding dimension for model %s: %s. "
                "Falling back to KG_EMBEDDING_DIMENSION=%d",
                self._model,
                e,
                settings.KG_EMBEDDING_DIMENSION,
            )
            return settings.KG_EMBEDDING_DIMENSION

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            all_embeddings.extend(self._embed_batch_resilient(batch, batch_start=i))

        return np.array(all_embeddings, dtype=np.float32)

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._detect_dimension()
        return self._dimension
