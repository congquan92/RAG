"""
Kieu du lieu LLM Provider
=========================
Data class dùng chung cho tầng trừu tượng multi-provider của LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMResult:
    """Kết quả từ một lần gọi LLM, có thể kèm thinking text."""
    content: str
    thinking: str = ""


@dataclass
class LLMImagePart:
    """Image attachment cho một LLM message."""
    data: bytes
    mime_type: str = "image/png"


@dataclass
class LLMMessage:
    """Một message trong conversation."""
    role: str  # "system" | "user" | "assistant"
    content: str = ""
    images: list[LLMImagePart] = field(default_factory=list)
    # Nội dung đặc thù theo provider ở dạng opaque (ví dụ Gemini Content có
    # thought_signature). Khi được set, provider nên dùng trực tiếp phần này
    # thay vì tự build từ ``content``/``images``.
    _raw_provider_content: object | None = field(default=None, repr=False)


@dataclass
class StreamChunk:
    """Một chunk từ luồng output streaming của LLM."""
    type: str  # "text" | "thinking" | "function_call"
    text: str = ""
    function_call: dict | None = None  # {"name": str, "args": dict}
