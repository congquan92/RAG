"""
Chat Models — SQLAlchemy ORM cho lịch sử hội thoại.

Tables:
  - chat_sessions: Mỗi phiên hội thoại (conversation) của user.
  - chat_messages: Từng tin nhắn (user/assistant) trong phiên.

Quan hệ: ChatSession 1─N ChatMessage (cascade delete).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    """Helper trả về UTC now (timezone-aware)."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Sinh UUID4 dạng string cho primary key."""
    return str(uuid.uuid4())


class ChatSession(Base):
    """
    Phiên hội thoại — nhóm các messages lại thành 1 cuộc trò chuyện.

    Columns:
      - id: UUID primary key
      - title: Tiêu đề phiên (auto-gen từ câu hỏi đầu hoặc user đặt)
        - description: Mô tả ngắn của workspace/session (nullable)
        - system_prompt: Prompt hệ thống tùy chỉnh (nullable)
      - created_at: Thời điểm tạo phiên
      - updated_at: Cập nhật mỗi khi có message mới
    """

    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    title: Mapped[str] = mapped_column(
        String(255), default="New Conversation"
    )
    description: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    system_prompt: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Relationship: 1 session → nhiều messages
    messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id!r} title={self.title!r}>"


class ChatMessage(Base):
    """
    Tin nhắn trong phiên hội thoại.

    Columns:
      - id: UUID primary key
      - session_id: FK → chat_sessions.id
      - role: "user" | "assistant" | "system"
      - content: Nội dung tin nhắn (full text)
      - citations: JSON string chứa danh sách nguồn trích dẫn (nullable)
      - created_at: Thời điểm gửi tin
    """

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(20),  # "user", "assistant", "system"
    )
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationship ngược
    session: Mapped[ChatSession] = relationship(
        "ChatSession", back_populates="messages"
    )

    def __repr__(self) -> str:
        preview = self.content[:50] if self.content else ""
        return f"<ChatMessage id={self.id!r} role={self.role!r} content={preview!r}>"
