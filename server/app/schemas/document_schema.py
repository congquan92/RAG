"""
Document Schemas — Pydantic DTOs cho luồng Upload & Ingestion API.

Định nghĩa format JSON request/response cho:
  - Upload file
  - Theo dõi trạng thái ingestion task
  - Liệt kê documents đã xử lý
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ═════════════════════════════════════════════════════════════════════════════
# Response DTOs
# ═════════════════════════════════════════════════════════════════════════════

class DocumentUploadResponse(BaseModel):
    """Response sau khi upload file — trả về task_id để theo dõi."""

    document_id: str = Field(..., description="ID document vừa tạo.")
    task_id: str = Field(
        ...,
        description="ID ingestion task — dùng để poll trạng thái.",
    )
    filename: str
    file_size: int = Field(..., description="Kích thước file (bytes).")
    mime_type: str
    message: str = Field(
        default="File uploaded. Ingestion started in background.",
        description="Thông báo cho user.",
    )


class IngestionTaskResponse(BaseModel):
    """Response trạng thái xử lý ingestion."""

    task_id: str
    document_id: str
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        ..., description="Trạng thái hiện tại của task."
    )
    chunks_processed: int = Field(
        default=0,
        description="Số chunks đã xử lý xong.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Chi tiết lỗi nếu status='failed'.",
    )
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentResponse(BaseModel):
    """Response thông tin 1 document đã upload."""

    id: str
    filename: str
    file_size: int
    mime_type: str
    chunk_count: int = Field(
        default=0,
        description="Số chunks đã tạo sau khi xử lý.",
    )
    created_at: datetime
    latest_task_status: Optional[str] = Field(
        default=None,
        description="Status của ingestion task gần nhất.",
    )

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    """Response danh sách documents (có pagination)."""

    documents: list[DocumentResponse] = Field(default_factory=list)
    total: int = Field(default=0, description="Tổng số documents.")


class DocumentDeleteResponse(BaseModel):
    """Response sau khi xóa document."""

    document_id: str
    filename: str
    message: str = Field(default="Document deleted successfully.")
