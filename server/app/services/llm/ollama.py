"""
Ollama LLM va Embedding Provider
"""
from __future__ import annotations

import json
import logging
import re
from typing import AsyncGenerator, Optional

import numpy as np

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

# Regex để loại bỏ block <think>...</think> khỏi output model
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OllamaLLMProvider(LLMProvider):
    """Sinh text/multimodal local bằng Ollama."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "gemma3:12b"):
        self._host = host
        self._model = model
        self._thinking_supported: bool | None = None  # probe lazy
        self._native_tools_supported: bool | None = None  # probe lazy
        self.last_response_message: dict | None = None  # dùng cho lịch sử native tool call

    # ------------------------------------------------------------------
    # Trợ giúp nội bộ
    # ------------------------------------------------------------------

    @staticmethod
    def _to_ollama_messages(
        messages: list[LLMMessage],
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Chuyển danh sách LLMMessage thành dict message của Ollama."""
        result: list[dict] = []

        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            # Dict message thô của Ollama - truyền thẳng nguyên trạng
            # (dùng cho native tool call history: assistant có tool_calls, tool results)
            if msg._raw_provider_content is not None:
                result.append(msg._raw_provider_content)
                continue

            entry: dict = {"role": msg.role, "content": msg.content}
            if msg.images:
                # Ollama chấp nhận raw bytes trong field 'images'
                entry["images"] = [img.data for img in msg.images]
            result.append(entry)

        return result

    @staticmethod
    def _extract_content(response, keep_thinking: bool = False) -> str | LLMResult:
        """Trích xuất text hữu dụng từ Ollama response.

        Xử lý các edge case:
        - ``content`` rỗng nhưng field ``thinking`` lại có câu trả lời
        - ``content`` chứa block ``<think>...</think>`` nhúng

        Khi *keep_thinking* là True, trả về LLMResult với
        thinking text được giữ riêng.
        """
        content = response.message.content or ""
        thinking = getattr(response.message, "thinking", None) or ""

        # Loại bỏ block <think>...</think> khỏi content
        if "<think>" in content:
            content = _THINK_RE.sub("", content).strip()

        # Fallback: nếu content vẫn rỗng, kiểm tra field thinking
        if not content:
            if thinking:
                logger.warning(
                    "Ollama response.content is empty but thinking has %d chars — "
                    "using thinking as fallback", len(thinking)
                )
                content = _THINK_RE.sub("", thinking).strip()

        if keep_thinking:
            return LLMResult(content=content, thinking=thinking)
        return content

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
        import ollama

        ollama_msgs = self._to_ollama_messages(messages, system_prompt)
        use_think = think and self.supports_thinking()

        try:
            client = ollama.Client(host=self._host)
            response = client.chat(
                model=self._model,
                messages=ollama_msgs,
                options={"temperature": temperature, "num_predict": max_tokens},
                think=True if use_think else None,
            )
            result = self._extract_content(response, keep_thinking=use_think)
            content = result.content if isinstance(result, LLMResult) else result
            if not content:
                logger.warning(
                    "Ollama complete() returned empty | model=%s | "
                    "content=%r | thinking=%r",
                    self._model,
                    response.message.content,
                    getattr(response.message, "thinking", None),
                )
            return result
        except Exception as e:
            logger.error(f"Ollama LLM call failed: {e}", exc_info=True)
            return LLMResult(content="") if use_think else ""

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """Async native qua ollama.AsyncClient (tốt hơn to_thread)."""
        import ollama

        ollama_msgs = self._to_ollama_messages(messages, system_prompt)
        use_think = think and self.supports_thinking()

        try:
            client = ollama.AsyncClient(host=self._host)
            response = await client.chat(
                model=self._model,
                messages=ollama_msgs,
                options={"temperature": temperature, "num_predict": max_tokens},
                think=True if use_think else None,
            )
            result = self._extract_content(response, keep_thinking=use_think)
            content = result.content if isinstance(result, LLMResult) else result
            if not content:
                logger.warning(
                    "Ollama acomplete() returned empty | model=%s | "
                    "content=%r | thinking=%r",
                    self._model,
                    response.message.content,
                    getattr(response.message, "thinking", None),
                )
            return result
        except Exception as e:
            logger.error(f"Ollama async LLM call failed: {e}", exc_info=True)
            return LLMResult(content="") if use_think else ""

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
        """Streaming generation qua async stream API của Ollama.

        Khi có *tools*, dùng native tool calling và tool call sẽ đến qua
        ``chunk.message.tool_calls``. Ngược lại, tag XML ``<tool_call>``
        kiểu prompt-based sẽ được phát hiện bằng state machine.
        """
        import ollama

        ollama_msgs = self._to_ollama_messages(messages, system_prompt)
        use_think = think and self.supports_thinking()

        try:
            client = ollama.AsyncClient(host=self._host)

            kwargs: dict = dict(
                model=self._model,
                messages=ollama_msgs,
                options={"temperature": temperature, "num_predict": max_tokens},
                stream=True,
                think=True if use_think else None,
            )
            if tools:
                kwargs["tools"] = tools

            stream = await client.chat(**kwargs)

            if tools:
                # -- Nhánh native tool calling --
                self.last_response_message = None

                async for chunk in stream:
                    thinking = getattr(chunk.message, "thinking", None) or ""
                    content = chunk.message.content or ""

                    if thinking:
                        yield StreamChunk(type="thinking", text=thinking)

                    if content:
                        cleaned = _THINK_RE.sub("", content)
                        if cleaned:
                            yield StreamChunk(type="text", text=cleaned)

                    # Native tool call đến dưới dạng object đầy đủ
                    tool_calls = getattr(chunk.message, "tool_calls", None)
                    if tool_calls:
                        self.last_response_message = {
                            "role": "assistant",
                            "content": content,
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    }
                                }
                                for tc in tool_calls
                            ],
                        }
                        for tc in tool_calls:
                            args = tc.function.arguments
                            yield StreamChunk(
                                type="function_call",
                                function_call={
                                    "name": tc.function.name,
                                    "args": args if isinstance(args, dict) else {},
                                },
                            )
            else:
                # -- Nhánh tool calling dạng prompt (XML state machine) --
                tool_buffer = ""
                in_tool_call = False

                async for chunk in stream:
                    thinking = getattr(chunk.message, "thinking", None) or ""
                    content = chunk.message.content or ""

                    if thinking:
                        yield StreamChunk(type="thinking", text=thinking)

                    if not content:
                        continue

                    if in_tool_call:
                        tool_buffer += content
                        if "</tool_call>" in tool_buffer:
                            match = re.search(
                                r"<tool_call>(.*?)</tool_call>",
                                tool_buffer,
                                re.DOTALL,
                            )
                            if match:
                                try:
                                    tool_data = json.loads(match.group(1).strip())
                                    yield StreamChunk(
                                        type="function_call",
                                        function_call={
                                            "name": tool_data.get("name", ""),
                                            "args": tool_data.get("arguments", {}),
                                        },
                                    )
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse tool call JSON: %s", match.group(1))
                                    yield StreamChunk(type="text", text=tool_buffer)
                            else:
                                yield StreamChunk(type="text", text=tool_buffer)
                            after = tool_buffer.split("</tool_call>", 1)[1]
                            tool_buffer = ""
                            in_tool_call = False
                            if after.strip():
                                yield StreamChunk(type="text", text=after)
                    elif "<tool_call>" in content:
                        before, rest = content.split("<tool_call>", 1)
                        if before.strip():
                            yield StreamChunk(type="text", text=before)
                        in_tool_call = True
                        tool_buffer = "<tool_call>" + rest
                        if "</tool_call>" in tool_buffer:
                            match = re.search(
                                r"<tool_call>(.*?)</tool_call>",
                                tool_buffer,
                                re.DOTALL,
                            )
                            if match:
                                try:
                                    tool_data = json.loads(match.group(1).strip())
                                    yield StreamChunk(
                                        type="function_call",
                                        function_call={
                                            "name": tool_data.get("name", ""),
                                            "args": tool_data.get("arguments", {}),
                                        },
                                    )
                                except json.JSONDecodeError:
                                    logger.warning("Failed to parse tool call JSON: %s", match.group(1))
                                    yield StreamChunk(type="text", text=tool_buffer)
                            after = tool_buffer.split("</tool_call>", 1)[1]
                            tool_buffer = ""
                            in_tool_call = False
                            if after.strip():
                                yield StreamChunk(type="text", text=after)
                    else:
                        cleaned = _THINK_RE.sub("", content)
                        if cleaned:
                            yield StreamChunk(type="text", text=cleaned)

                if in_tool_call and tool_buffer:
                    yield StreamChunk(type="text", text=tool_buffer)

        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}", exc_info=True)
            yield StreamChunk(type="text", text="")

    def supports_vision(self) -> bool:
        # Hỗ trợ vision phụ thuộc model (vd: qwen3-vl, llava, ...)
        # Trả về True và để model tự xử lý; nếu model không hỗ trợ,
        # Ollama API sẽ trả lỗi một cách an toàn.
        return True

    def supports_thinking(self) -> bool:
        """Phát hiện model có hỗ trợ thinking mode bằng probe call."""
        if self._thinking_supported is not None:
            return self._thinking_supported

        import ollama

        try:
            client = ollama.Client(host=self._host)
            response = client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                options={"num_predict": 2},
                think=True,
            )
            # Nếu chạy tới đây không lỗi thì thinking được hỗ trợ
            thinking = getattr(response.message, "thinking", None) or ""
            self._thinking_supported = True
            logger.info(
                f"Ollama thinking probe: model={self._model} host={self._host} "
                f"supported=True (thinking={len(thinking)} chars)"
            )
        except Exception as e:
            self._thinking_supported = False
            logger.info(
                f"Ollama thinking probe: model={self._model} host={self._host} "
                f"supported=False ({e})"
            )

        return self._thinking_supported

    def supports_native_tools(self) -> bool:
        """Phát hiện model có hỗ trợ native tool calling bằng probe call.

        Gửi một câu hỏi có khả năng kích hoạt tool call. Chỉ đánh dấu model
        là hỗ trợ native tools khi nó *thực sự* sinh response ``tool_calls``,
        không chỉ đơn giản là API chấp nhận tham số.
        """
        if self._native_tools_supported is not None:
            return self._native_tools_supported

        import ollama

        _PROBE_TOOL = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up information. You MUST call this for any question.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "search query"}},
                    "required": ["query"],
                },
            },
        }]

        try:
            client = ollama.Client(host=self._host)
            use_think = self.supports_thinking()
            response = client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You MUST use the lookup tool for any question."},
                    {"role": "user", "content": "What is the capital of France?"},
                ],
                options={"num_predict": 256},
                tools=_PROBE_TOOL,
                think=True if use_think else None,
            )
            tool_calls = getattr(response.message, "tool_calls", None)
            self._native_tools_supported = bool(tool_calls)
            logger.info(
                "Ollama native tools probe: model=%s host=%s supported=%s "
                "(tool_calls=%s)",
                self._model, self._host, self._native_tools_supported,
                bool(tool_calls),
            )
        except Exception as e:
            self._native_tools_supported = False
            logger.info(
                "Ollama native tools probe: model=%s host=%s supported=False (%s)",
                self._model, self._host, e,
            )

        return self._native_tools_supported


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Text embedding local bằng Ollama."""

    _BATCH_SIZE = 64
    _MAX_BATCH_RETRIES = 1
    _CHUNK_FALLBACK_MAX_CHARS = 1600
    _CHUNK_FALLBACK_OVERLAP_CHARS = 200

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "bge-m3",
    ):
        self._host = host
        self._model = model
        self._dimension: Optional[int] = None
        self._sync_client = None

    def _get_sync_client(self):
        """Tái sử dụng một sync client theo host cho mọi lần gọi embedding."""
        import ollama

        if self._sync_client is None:
            self._sync_client = ollama.Client(host=self._host)
        return self._sync_client

    def _detect_dimension(self) -> int:
        """Phát hiện embedding dimension bằng cách chạy probe."""
        try:
            result = self._get_sync_client().embed(model=self._model, input=["dimension probe"])
            dim = len(result.embeddings[0])
            logger.info(f"Detected Ollama embedding dimension: {dim} for model {self._model}")
            return dim
        except Exception as e:
            logger.warning(f"Failed to detect embedding dimension: {e}, defaulting to config")
            from app.core.config import settings
            return settings.KG_EMBEDDING_DIMENSION

    def _embed_batch_once_sync(self, texts: list[str]) -> np.ndarray:
        """Embed một batch và ép ràng buộc số lượng input/output 1:1."""
        result = self._get_sync_client().embed(model=self._model, input=texts)
        arr = np.array(result.embeddings, dtype=np.float32)

        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValueError(
                "Ollama embedding count mismatch "
                f"(expected={len(texts)} got={arr.shape[0] if arr.ndim >= 1 else 0})"
            )

        if np.any(np.isnan(arr)):
            logger.warning("Ollama embed_sync produced NaN values — replacing with zeros")
            arr = np.nan_to_num(arr, nan=0.0)

        if self._dimension is None and arr.shape[0] > 0:
            self._dimension = int(arr.shape[1])

        return arr

    async def _embed_batch_once_async(self, client, texts: list[str]) -> np.ndarray:
        """Embed batch async với kiểm tra output nghiêm ngặt."""
        result = await client.embed(model=self._model, input=texts)
        arr = np.array(result.embeddings, dtype=np.float32)

        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise ValueError(
                "Ollama async embedding count mismatch "
                f"(expected={len(texts)} got={arr.shape[0] if arr.ndim >= 1 else 0})"
            )

        if np.any(np.isnan(arr)):
            logger.warning("Ollama async embed produced NaN values — replacing with zeros")
            arr = np.nan_to_num(arr, nan=0.0)

        if self._dimension is None and arr.shape[0] > 0:
            self._dimension = int(arr.shape[1])

        return arr

    def _embed_batch_resilient_sync(
        self,
        texts: list[str],
        *,
        batch_start: int,
        depth: int = 0,
    ) -> np.ndarray:
        """Thử lại cả batch, rồi tách đệ quy để cô lập item không ổn định."""
        last_error: Exception | None = None
        for attempt in range(self._MAX_BATCH_RETRIES + 1):
            try:
                return self._embed_batch_once_sync(texts)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Ollama batch embed retry %d/%d failed at batch_start=%d size=%d depth=%d: %s",
                    attempt + 1,
                    self._MAX_BATCH_RETRIES + 1,
                    batch_start,
                    len(texts),
                    depth,
                    e,
                )

        if self._is_non_retriable_error(last_error):
            logger.error(
                "Ollama non-retriable embed error at batch_start=%d size=%d: %s",
                batch_start,
                len(texts),
                last_error,
            )
            return np.zeros((len(texts), self.get_dimension()), dtype=np.float32)

        if len(texts) == 1:
            if self._is_context_length_error(last_error):
                return self._embed_single_with_chunk_fallback_sync(
                    texts[0],
                    batch_start=batch_start,
                    depth=depth,
                )
            logger.error(
                "Ollama embedding failed at leaf batch_start=%d (last batch error: %s)",
                batch_start,
                last_error,
            )
            return np.zeros((1, self.get_dimension()), dtype=np.float32)

        mid = len(texts) // 2
        left = self._embed_batch_resilient_sync(
            texts[:mid], batch_start=batch_start, depth=depth + 1,
        )
        right = self._embed_batch_resilient_sync(
            texts[mid:], batch_start=batch_start + mid, depth=depth + 1,
        )
        return np.vstack([left, right])

    async def _embed_batch_resilient_async(
        self,
        client,
        texts: list[str],
        *,
        batch_start: int,
        depth: int = 0,
    ) -> np.ndarray:
        """Biến thể async của cơ chế retry + fallback tách batch."""
        last_error: Exception | None = None
        for attempt in range(self._MAX_BATCH_RETRIES + 1):
            try:
                return await self._embed_batch_once_async(client, texts)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Ollama async batch retry %d/%d failed at batch_start=%d size=%d depth=%d: %s",
                    attempt + 1,
                    self._MAX_BATCH_RETRIES + 1,
                    batch_start,
                    len(texts),
                    depth,
                    e,
                )

        if self._is_non_retriable_error(last_error):
            logger.error(
                "Ollama async non-retriable embed error at batch_start=%d size=%d: %s",
                batch_start,
                len(texts),
                last_error,
            )
            return np.zeros((len(texts), self.get_dimension()), dtype=np.float32)

        if len(texts) == 1:
            if self._is_context_length_error(last_error):
                return await self._embed_single_with_chunk_fallback_async(
                    client,
                    texts[0],
                    batch_start=batch_start,
                    depth=depth,
                )
            logger.error(
                "Ollama async embedding failed at leaf batch_start=%d (last batch error: %s)",
                batch_start,
                last_error,
            )
            return np.zeros((1, self.get_dimension()), dtype=np.float32)

        mid = len(texts) // 2
        left = await self._embed_batch_resilient_async(
            client,
            texts[:mid],
            batch_start=batch_start,
            depth=depth + 1,
        )
        right = await self._embed_batch_resilient_async(
            client,
            texts[mid:],
            batch_start=batch_start + mid,
            depth=depth + 1,
        )
        return np.vstack([left, right])

    @staticmethod
    def _is_non_retriable_error(error: Exception | None) -> bool:
        if error is None:
            return False
        msg = str(error).lower()
        fatal_markers = [
            "not found",
            "invalid",
            "unauthorized",
            "forbidden",
            "permission",
        ]
        return any(marker in msg for marker in fatal_markers)

    @staticmethod
    def _is_context_length_error(error: Exception | None) -> bool:
        if error is None:
            return False
        msg = str(error).lower()
        markers = [
            "input length exceeds the context length",
            "context length",
            "context window",
            "prompt is too long",
            "token limit",
            "maximum context",
        ]
        return any(marker in msg for marker in markers)

    def _split_text_for_embedding(self, text: str) -> list[str]:
        """Tách text quá dài thành cửa sổ theo câu để fallback embedding."""
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return ["[empty]"]

        max_chars = self._CHUNK_FALLBACK_MAX_CHARS
        overlap = self._CHUNK_FALLBACK_OVERLAP_CHARS
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

                step = max(1, max_chars - overlap)
                for start in range(0, len(sentence), step):
                    piece = sentence[start : start + max_chars].strip()
                    if piece:
                        chunks.append(piece)
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = sentence

        if current:
            chunks.append(current)

        return chunks if chunks else [normalized[:max_chars]]

    def _force_bisect_text(self, text: str) -> list[str]:
        """Fallback tách cứng khi cách tách theo câu không giảm được thêm."""
        normalized = " ".join(str(text or "").split())
        if len(normalized) <= 128:
            return [normalized] if normalized else ["[empty]"]

        overlap = min(self._CHUNK_FALLBACK_OVERLAP_CHARS, max(16, len(normalized) // 8))
        mid = len(normalized) // 2
        left = normalized[:mid].strip()
        right_start = max(0, mid - overlap)
        right = normalized[right_start:].strip()
        parts = [part for part in (left, right) if part]
        return parts if parts else [normalized]

    @staticmethod
    def _pool_chunk_embeddings(vectors: np.ndarray, chunks: list[str]) -> np.ndarray:
        """Gộp vector của chunk về một vector bằng trọng số theo độ dài."""
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise ValueError("Cannot pool empty chunk embeddings")
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                "Chunk embedding count mismatch "
                f"(vectors={vectors.shape[0]} chunks={len(chunks)})"
            )

        weights = np.array([max(1, len(chunk)) for chunk in chunks], dtype=np.float32)
        pooled = np.average(vectors, axis=0, weights=weights).astype(np.float32)
        if np.any(np.isnan(pooled)):
            pooled = np.nan_to_num(pooled, nan=0.0)
        return pooled

    def _embed_single_with_chunk_fallback_sync(
        self,
        text: str,
        *,
        batch_start: int,
        depth: int,
    ) -> np.ndarray:
        chunks = self._split_text_for_embedding(text)
        if len(chunks) <= 1:
            chunks = self._force_bisect_text(text)
        if len(chunks) <= 1:
            logger.error(
                "Ollama context-length leaf failure at batch_start=%d depth=%d; returning zeros",
                batch_start,
                depth,
            )
            return np.zeros((1, self.get_dimension()), dtype=np.float32)

        logger.warning(
            "Ollama embedding fallback: splitting oversized input at batch_start=%d "
            "depth=%d into %d chunks",
            batch_start,
            depth,
            len(chunks),
        )

        vectors = self._embed_batch_resilient_sync(
            chunks,
            batch_start=batch_start,
            depth=depth + 1,
        )
        pooled = self._pool_chunk_embeddings(vectors, chunks)
        if self._dimension is None:
            self._dimension = int(pooled.shape[0])
        return pooled.reshape(1, -1)

    async def _embed_single_with_chunk_fallback_async(
        self,
        client,
        text: str,
        *,
        batch_start: int,
        depth: int,
    ) -> np.ndarray:
        chunks = self._split_text_for_embedding(text)
        if len(chunks) <= 1:
            chunks = self._force_bisect_text(text)
        if len(chunks) <= 1:
            logger.error(
                "Ollama async context-length leaf failure at batch_start=%d depth=%d; returning zeros",
                batch_start,
                depth,
            )
            return np.zeros((1, self.get_dimension()), dtype=np.float32)

        logger.warning(
            "Ollama async embedding fallback: splitting oversized input at batch_start=%d "
            "depth=%d into %d chunks",
            batch_start,
            depth,
            len(chunks),
        )

        vectors = await self._embed_batch_resilient_async(
            client,
            chunks,
            batch_start=batch_start,
            depth=depth + 1,
        )
        pooled = self._pool_chunk_embeddings(vectors, chunks)
        if self._dimension is None:
            self._dimension = int(pooled.shape[0])
        return pooled.reshape(1, -1)

    @staticmethod
    def _sanitize_texts(texts: list[str]) -> list[str]:
        """Làm sạch text để tránh lỗi NaN khi embedding bằng Ollama.

        Một số text (rỗng, chỉ có ký tự đặc biệt, quá dài) có thể khiến
        bge-m3 qua Ollama trả NaN embedding hoặc lỗi 500.
        """
        sanitized = []
        for t in texts:
            t = t.strip()
            if not t:
                t = "[empty]"
            # Cắt bớt text quá dài (>8192 tokens ≈ 32k chars)
            if len(t) > 32000:
                t = t[:32000]
            sanitized.append(t)
        return sanitized

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        clean = self._sanitize_texts(texts)
        parts: list[np.ndarray] = []
        for i in range(0, len(clean), self._BATCH_SIZE):
            batch = clean[i : i + self._BATCH_SIZE]
            parts.append(self._embed_batch_resilient_sync(batch, batch_start=i))
        return np.vstack(parts)

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embedding async native qua ollama.AsyncClient."""
        import ollama

        if not texts:
            return np.zeros((0, self.get_dimension()), dtype=np.float32)

        clean = self._sanitize_texts(texts)
        client = ollama.AsyncClient(host=self._host)
        parts: list[np.ndarray] = []
        for i in range(0, len(clean), self._BATCH_SIZE):
            batch = clean[i : i + self._BATCH_SIZE]
            parts.append(
                await self._embed_batch_resilient_async(
                    client,
                    batch,
                    batch_start=i,
                )
            )
        return np.vstack(parts)

    def get_dimension(self) -> int:
        if self._dimension is None:
            self._dimension = self._detect_dimension()
        return self._dimension
