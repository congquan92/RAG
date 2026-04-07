"""
Document Router — API endpoints cho Upload & Ingestion pipeline.

Endpoints:
  POST   /documents/upload           → Upload file → background ingestion
  GET    /documents/status/{task_id} → Poll trạng thái ingestion task
  GET    /documents                  → Liệt kê documents (pagination)
  GET    /documents/{document_id}    → Chi tiết 1 document
  DELETE /documents/{document_id}    → Xóa document + file + ChromaDB chunks

Upload flow:
  1. Client gửi file multipart/form-data
  2. Server lưu file + tạo DB records (Document, IngestionTask)
  3. Background task parse → chunk → embed → lưu ChromaDB
  4. Client poll GET /status/{task_id} để biết khi nào xong
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_embeddings
from app.core.settings import settings
from app.schemas.document_schema import (
    DocumentBatchProcessRequest,
    DocumentBatchProcessResponse,
    DocumentBatchTaskItem,
    DocumentDeleteResponse,
    DocumentIngestionTriggerResponse,
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    IngestionTaskResponse,
)
from app.services import document_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# ═════════════════════════════════════════════════════════════════════════════
# Upload
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=202,
    summary="Upload file → khởi động background ingestion",
)
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
):
    """
    Nhận file upload (multipart), validate → lưu disk → tạo IngestionTask.

    Trả về task_id ngay lập tức, ingestion chạy ngầm trong background.
    Client dùng GET /status/{task_id} để poll tiến trình.
    """
    # ── Validation ───────────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    # Đọc nội dung file
    file_content = await file.read()
    max_file_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    if len(file_content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(file_content) > max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "File too large. "
                f"Maximum size: {settings.max_upload_size_mb} MB."
            ),
        )

    # ── Save file + create DB records ────────────────────────────────────
    document, task = await document_service.save_uploaded_file(
        db=db,
        filename=file.filename,
        file_content=file_content,
    )

    # Commit trước khi queue background task để tránh race condition:
    # worker đọc task/document khi transaction upload chưa được flush ra DB.
    await db.commit()

    # Validate MIME type sau khi detect (python-magic trong service)
    if (
        document.mime_type
        and document.mime_type not in settings.allowed_upload_mime_type_set
    ):
        logger.warning(
            "Unusual MIME type uploaded: %s for file %s — accepting anyway",
            document.mime_type,
            file.filename,
        )

    # ── Queue background ingestion ───────────────────────────────────────
    background_tasks.add_task(
        document_service.process_ingestion_task,
        task_id=task.id,
        document_id=document.id,
        embeddings=embeddings,
    )

    logger.info(
        "Upload accepted: file=%s, doc_id=%s, task_id=%s",
        file.filename, document.id, task.id,
    )

    return DocumentUploadResponse(
        document_id=document.id,
        task_id=task.id,
        filename=document.filename,
        file_size=document.file_size,
        mime_type=document.mime_type or "unknown",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Ingestion Status
# ═════════════════════════════════════════════════════════════════════════════

@router.get(
    "/status/{task_id}",
    response_model=IngestionTaskResponse,
    summary="Poll trạng thái ingestion task",
)
async def get_ingestion_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Kiểm tra tiến trình xử lý file (pending → processing → completed/failed)."""
    task = await document_service.get_task_status(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return IngestionTaskResponse(
        task_id=task.id,
        document_id=task.document_id,
        status=task.status,
        chunks_processed=task.chunks_processed or 0,
        error_message=task.error_message,
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


@router.post(
    "/{document_id}/process",
    response_model=DocumentIngestionTriggerResponse,
    summary="Queue processing cho 1 document",
)
async def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
):
    try:
        document, task, is_new_task = await document_service.queue_document_ingestion(
            db=db,
            document_id=document_id,
            force_reindex=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if is_new_task:
        await db.commit()
        background_tasks.add_task(
            document_service.process_ingestion_task,
            task_id=task.id,
            document_id=document.id,
            embeddings=embeddings,
        )
        message = "Ingestion task queued."
    else:
        message = "A processing task is already running for this document."

    return DocumentIngestionTriggerResponse(
        document_id=document.id,
        task_id=task.id,
        status="pending",
        message=message,
    )


@router.post(
    "/{document_id}/reindex",
    response_model=DocumentIngestionTriggerResponse,
    summary="Queue reindex cho 1 document",
)
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
):
    try:
        document, task, is_new_task = await document_service.queue_document_ingestion(
            db=db,
            document_id=document_id,
            force_reindex=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if is_new_task:
        await db.commit()
        background_tasks.add_task(
            document_service.process_ingestion_task,
            task_id=task.id,
            document_id=document.id,
            embeddings=embeddings,
        )
        message = "Reindex task queued."
    else:
        message = "A processing task is already running for this document."

    return DocumentIngestionTriggerResponse(
        document_id=document.id,
        task_id=task.id,
        status="pending",
        message=message,
    )


@router.post(
    "/process/batch",
    response_model=DocumentBatchProcessResponse,
    summary="Queue batch processing cho nhiều documents",
)
async def process_documents_batch(
    background_tasks: BackgroundTasks,
    body: DocumentBatchProcessRequest | None = None,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
):
    requested_ids = body.document_ids if body and body.document_ids else []
    force_reindex = body.force_reindex if body else False

    if requested_ids:
        target_ids = requested_ids
    else:
        documents, _total = await document_service.list_documents(db, skip=0, limit=1000)
        target_ids = [doc.id for doc in documents]

    tasks: list[DocumentBatchTaskItem] = []
    queued_tasks: list[tuple[str, str]] = []

    for document_id in target_ids:
        try:
            document, task, is_new_task = await document_service.queue_document_ingestion(
                db=db,
                document_id=document_id,
                force_reindex=force_reindex,
            )
        except ValueError:
            continue

        tasks.append(DocumentBatchTaskItem(document_id=document.id, task_id=task.id))
        if is_new_task:
            queued_tasks.append((task.id, document.id))

    if queued_tasks:
        await db.commit()
        for task_id, document_id in queued_tasks:
            background_tasks.add_task(
                document_service.process_ingestion_task,
                task_id=task_id,
                document_id=document_id,
                embeddings=embeddings,
            )

    return DocumentBatchProcessResponse(
        queued=len(queued_tasks),
        tasks=tasks,
    )


@router.get(
    "/{document_id}/markdown",
    summary="Render markdown/text preview for one document",
)
async def get_document_markdown(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        _document, markdown = await document_service.get_document_markdown(
            db=db,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=f"Cannot extract text: {exc}")

    return Response(content=markdown, media_type="text/markdown; charset=utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Document CRUD
# ═════════════════════════════════════════════════════════════════════════════

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="Liệt kê documents",
)
async def list_documents(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Danh sách documents đã upload với pagination."""
    documents, total = await document_service.list_documents(db, skip=skip, limit=limit)

    doc_responses = []
    for doc in documents:
        latest_task = await document_service.get_latest_task_for_document(db, doc.id)
        doc_responses.append(DocumentResponse(
            id=doc.id,
            filename=doc.filename,
            file_size=doc.file_size,
            mime_type=doc.mime_type or "unknown",
            chunk_count=doc.chunk_count,
            created_at=doc.created_at,
            latest_task_status=latest_task.status if latest_task else None,
        ))

    return DocumentListResponse(documents=doc_responses, total=total)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Chi tiết 1 document",
)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Lấy thông tin document theo ID."""
    document = await document_service.get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    latest_task = await document_service.get_latest_task_for_document(db, document.id)

    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        file_size=document.file_size,
        mime_type=document.mime_type or "unknown",
        chunk_count=document.chunk_count,
        created_at=document.created_at,
        latest_task_status=latest_task.status if latest_task else None,
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Xóa document + cleanup",
)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Xóa document: DB record + file vật lý + ChromaDB chunks."""
    document = await document_service.delete_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail=f"Document not found: {document_id}")

    return DocumentDeleteResponse(
        document_id=document.id,
        filename=document.filename,
    )
