"""
Document Models — SQLAlchemy ORM cho metadata file và trạng thái ingestion.

Tables:
  - documents: Metadata của file đã upload (tên, path, kích thước, loại).
  - ingestion_tasks: Theo dõi trạng thái xử lý từng file (pending → processing → done/failed).

Quan hệ: Document 1─N IngestionTask (cascade delete).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    return str(uuid.uuid4())


class Document(Base):
    """
    Metadata file đã upload vào hệ thống.

    Columns:
      - id: UUID primary key
      - filename: Tên file gốc user upload
      - file_path: Đường dẫn lưu trên server
      - file_size: Kích thước file (bytes)
      - mime_type: Loại file (pdf, docx, ...) từ python-magic
      - chunk_count: Số chunks đã tạo sau khi xử lý (0 nếu chưa xử lý)
      - created_at: Thời điểm upload
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    workspace_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True, default=None
    )
    filename: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str] = mapped_column(String(1000))
    file_size: Mapped[int] = mapped_column(BigInteger, default=0)
    mime_type: Mapped[str] = mapped_column(String(100), default="application/octet-stream")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )

    # Relationship: 1 document → nhiều ingestion tasks (retry, re-process)
    ingestion_tasks: Mapped[list[IngestionTask]] = relationship(
        "IngestionTask",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="IngestionTask.created_at.desc()",
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id!r} filename={self.filename!r}>"


class IngestionTask(Base):
    """
    Theo dõi trạng thái xử lý (ingestion) của 1 file.

    Mỗi lần upload hoặc re-process sẽ tạo 1 task mới.
    Background worker cập nhật status theo tiến trình.

    Columns:
      - id: UUID primary key (cũng là task_id trả về cho client)
      - document_id: FK → documents.id
      - status: "pending" | "processing" | "completed" | "failed"
      - error_message: Chi tiết lỗi nếu failed (nullable)
      - chunks_processed: Số chunks đã xử lý xong (progress tracking)
      - created_at: Thời điểm tạo task
      - updated_at: Cập nhật mỗi khi status thay đổi
    """

    __tablename__ = "ingestion_tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_generate_uuid
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None
    )
    chunks_processed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    # Relationship ngược
    document: Mapped[Document] = relationship(
        "Document", back_populates="ingestion_tasks"
    )

    def __repr__(self) -> str:
        return (
            f"<IngestionTask id={self.id!r} "
            f"document_id={self.document_id!r} "
            f"status={self.status!r}>"
        )
