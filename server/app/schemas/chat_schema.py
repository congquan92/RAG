"""
Chat Schemas — Pydantic DTOs cho luồng Chat API.

Định nghĩa format JSON request/response cho:
  - Tạo session mới
  - Gửi tin nhắn (query)
  - Nhận phản hồi (answer + citations)
  - Lấy lịch sử chat
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# Request DTOs
# ═════════════════════════════════════════════════════════════════════════════

class ChatSessionCreate(BaseModel):
    """Request tạo phiên hội thoại mới."""

    title: str = Field(
        default="New Conversation",
        max_length=255,
        description="Tiêu đề phiên chat (tùy chọn).",
    )


class ChatSessionUpdate(BaseModel):
    """Request cập nhật metadata phiên hội thoại."""

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Tiêu đề phiên chat.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Mô tả workspace/session.",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Prompt hệ thống tùy chỉnh.",
    )


class ChatQueryRequest(BaseModel):
    """Request gửi câu hỏi trong phiên chat."""

    session_id: str = Field(
        ...,
        description="ID phiên hội thoại đang hoạt động.",
    )
    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Câu hỏi của user.",
    )
    stream: bool = Field(
        default=False,
        description="True để nhận response qua SSE streaming.",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Response DTOs
# ═════════════════════════════════════════════════════════════════════════════

class CitationItem(BaseModel):
    """Một nguồn trích dẫn trong câu trả lời."""

    document_id: str = Field(..., description="ID document nguồn.")
    filename: str = Field(..., description="Tên file nguồn.")
    chunk_text: str = Field(
        default="",
        description="Đoạn text được trích dẫn.",
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Điểm liên quan (0-1) từ reranker.",
    )


class ChatMessageResponse(BaseModel):
    """Response cho 1 tin nhắn (cả user lẫn assistant)."""

    id: str
    session_id: str
    role: str = Field(..., description="'user' | 'assistant' | 'system'")
    content: str
    citations: Optional[list[CitationItem]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionResponse(BaseModel):
    """Response thông tin phiên hội thoại."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    message_count: int = Field(
        default=0,
        description="Tổng số tin nhắn trong phiên.",
    )

    model_config = {"from_attributes": True}


class ChatSessionDetailResponse(BaseModel):
    """Response chi tiết phiên chat kèm lịch sử messages."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    messages: list[ChatMessageResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ChatAnswerResponse(BaseModel):
    """Response cho câu trả lời từ RAG pipeline."""

    session_id: str
    answer: str = Field(..., description="Câu trả lời từ LLM.")
    citations: list[CitationItem] = Field(
        default_factory=list,
        description="Danh sách nguồn trích dẫn.",
    )
    user_message_id: str = Field(
        ..., description="ID message user vừa gửi."
    )
    assistant_message_id: str = Field(
        ..., description="ID message assistant vừa tạo."
    )
