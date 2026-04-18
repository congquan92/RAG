"""
Gemini LLM va Embedding Provider
================================
Triển khai cụ thể bằng SDK ``google-genai`` của Google.

Hỗ trợ cả Gemini 2.5 (thinking_budget_tokens) và Gemini 3.x+
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

# Regex trích xuất major version từ tên model: gemini-2.5-flash -> 2, gemini-3.1-flash-lite -> 3
_GEMINI_VERSION_RE = re.compile(r"gemini-(\d+)")


class GeminiLLMProvider(LLMProvider):
    """Sinh text/multimodal bằng Google Gemini."""

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
    # Trợ giúp nội bộ
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_major_version(model: str) -> int:
        """Trích xuất major version từ tên model (vd: 'gemini-3.1-flash' -> 3)."""
        match = _GEMINI_VERSION_RE.search(model)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _build_parts(msg: LLMMessage) -> list[types.Part]:
        """Chuyển một LLMMessage thành danh sách Gemini Part object."""
        parts: list[types.Part] = []
        if msg.content:
            parts.append(types.Part.from_text(text=msg.content))
        for img in msg.images:
            parts.append(types.Part.from_bytes(data=img.data, mime_type=img.mime_type))
        return parts

    def _to_contents(self, messages: list[LLMMessage]) -> list[types.Content]:
        """Ánh xạ danh sách LLMMessage thành Gemini Content object.

        System message được inject thành cặp user->model giả lập
        (Gemini không hỗ trợ native system role trong ``contents``).

        Nếu message có ``_raw_provider_content`` (``types.Content`` native của Gemini)
        thì dùng trực tiếp, để giữ các field opaque như ``thought_signature``
        vốn không thể tái tạo từ plain text.
        """
        contents: list[types.Content] = []
        for msg in messages:
            # Raw Gemini Content - dùng nguyên bản (giữ thought_signature)
            if msg._raw_provider_content is not None:
                contents.append(msg._raw_provider_content)
                continue

            if msg.role == "system":
                # Gemini: system role không được phép trong contents -> inject thành
                # cặp user instruction + model acknowledgement.
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
        """Tạo ThinkingConfig theo phiên bản model.

        Gemini 2.5: dùng ``thinking_budget_tokens`` (KHÔNG hỗ trợ thinking_level).
        Gemini 3.x+: dùng ``thinking_level`` + ``include_thoughts=True``.
        """
        if self._major_version >= 3:
            return types.ThinkingConfig(
                thinking_level=self._thinking_level,
                include_thoughts=True,
            )
        # Gemini 2.5 - dùng thinking theo budget
        _BUDGET_MAP = {"minimal": 1024, "low": 2048, "medium": 4096, "high": 8192}
        budget = _BUDGET_MAP.get(self._thinking_level, 4096)
        return types.ThinkingConfig(thinking_budget=budget)

    # ------------------------------------------------------------------
    # Giao diện LLMProvider
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
        """Trích xuất content và thinking từ Gemini response."""
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
        """Streaming generation qua async stream API của Gemini.

        Sau khi streaming xong, ``self.last_response_content`` chứa
        ``types.Content`` đã tích lũy đủ mọi part (kể cả field opaque
        ``thought_signature``). Caller nào cần dựng history nhiều lượt đúng
        chuẩn (vd sau function call) nên đọc thuộc tính này và truyền lại
        qua ``LLMMessage._raw_provider_content``.
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

        # Tích lũy raw part để caller truy cập full response,
        # bao gồm thought_signature cho chu trình multi-turn chuẩn.
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
            # Lưu Content response đầy đủ cho caller cần
            # luân chuyển thought_signature (Gemini 3 function calling).
            self.last_response_content = types.Content(
                role="model",
                parts=accumulated_parts,
            ) if accumulated_parts else None

    def supports_vision(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        """Model Gemini 2.5+ và 3.x+ hỗ trợ thinking."""
        return self._major_version >= 2

    def supports_native_tools(self) -> bool:
        return True


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Text embedding của Google Gemini (``gemini-embedding-001``, 3072-dim)."""

    _BATCH_SIZE = 100  # giới hạn Gemini API
    _MAX_BATCH_RETRIES = 1

    def __init__(self, api_key: str, model: str = "gemini-embedding-001"):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dimension: int | None = None

    @staticmethod
    def _to_embed_contents(texts: list[str]) -> list[types.Content]:
        """Tạo một Gemini Content cho mỗi input text (hợp đồng 1:1 text->vector)."""
        contents: list[types.Content] = []
        for text in texts:
            normalized = text.strip() or "[empty]"
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=normalized)],
            ))
        return contents

    def _embed_one(self, text: str) -> list[float]:
        """Embed một text đơn và trả về một vector."""
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
        """Embed một batch và ép ràng buộc số lượng input/output 1:1."""
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
        """Thử lại cả batch, rồi tách đệ quy để cô lập item lỗi."""
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
            # Fallback ở node lá: vẫn giữ hợp đồng 1:1 cả khi lỗi cứng đầu.
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
        """Probe embedding dimension thực tế từ model đang dùng."""
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
