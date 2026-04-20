"""
Base Class cho LLM Provider
===========================
Interface trừu tượng cho sinh text/vision và embedding của LLM.
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

import numpy as np

from app.services.llm.types import LLMMessage, LLMResult, StreamChunk


class LLMProvider(ABC):
    """Giao diện trừu tượng cho sinh text/multimodal của LLM."""

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """
        Sinh text đồng bộ.

        Args:
            messages: Lịch sử hội thoại (có thể gồm image).
            temperature: Nhiệt độ sampling.
            max_tokens: Số output token tối đa.
            system_prompt: Chỉ dẫn mức system (provider tự xử lý việc inject).
            think: Nếu True và được hỗ trợ, trả về LLMResult có thinking text.

        Returns:
            Chuỗi text sinh ra, hoặc LLMResult khi think=True.
        """
        ...

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        """
        Sinh text bất đồng bộ.
        Mặc định: chạy complete() trong thread pool.
        Provider có native async nên override hàm này.
        """
        return await asyncio.to_thread(
            self.complete,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think,
        )

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
        """Sinh stream bất đồng bộ. Yield đối tượng StreamChunk.

        Fallback mặc định: gọi acomplete() và yield một text chunk duy nhất.
        Provider có native streaming nên override hàm này.
        """
        result = await self.acomplete(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            think=think,
        )
        if isinstance(result, LLMResult):
            if result.thinking:
                yield StreamChunk(type="thinking", text=result.thinking)
            yield StreamChunk(type="text", text=result.content)
        else:
            yield StreamChunk(type="text", text=result)

    @abstractmethod
    def supports_vision(self) -> bool:
        """Provider/model này có hỗ trợ input image hay không."""
        ...

    def supports_thinking(self) -> bool:
        """Provider/model này có hỗ trợ thinking mode hay không."""
        return False

    def supports_native_tools(self) -> bool:
        """Provider/model này có hỗ trợ native tool calling hay không."""
        return False


class EmbeddingProvider(ABC):
    """Giao diện trừu tượng cho sinh text embedding (dùng cho KG)."""

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> np.ndarray:
        """
        Sinh embedding theo batch đồng bộ.

        Returns:
            numpy array có shape (len(texts), embedding_dim).
        """
        ...

    async def embed(self, texts: list[str]) -> np.ndarray:
        """
        Sinh embedding theo batch bất đồng bộ.
        Mặc định: chạy embed_sync() trong thread pool.
        """
        return await asyncio.to_thread(self.embed_sync, texts)

    @abstractmethod
    def get_dimension(self) -> int:
        """Trả về embedding vector dimension của model này."""
        ...
