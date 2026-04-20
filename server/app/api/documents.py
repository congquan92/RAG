from __future__ import annotations

import os
import re
import uuid
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import PlainTextResponse
from sqlalchemy.ext.asyncio import AsyncSession
import json
from sqlalchemy import select

from app.core.config import settings
from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentImage, DocumentStatus
from app.schemas.document import DocumentResponse, DocumentUploadResponse
from app.schemas.rag import DocumentImageResponse

logger = logging.getLogger(__name__)


def _inject_images_from_db(
    markdown: str,
    images: list[DocumentImage],
    workspace_id: int,
) -> str:
    """Thay các placeholder <!-- image --> còn lại bằng image markdown thực.

    Dùng như lớp safety net khi parser chưa inject trong lúc xử lý.
    Ảnh được ghép theo thứ tự insert (theo primary key), tương ứng với
    thứ tự ảnh trong document Docling gốc.
    """
    img_iter = iter(images)

    def _replacer(match):
        try:
            img = next(img_iter)
            url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
            caption = (img.caption or "").replace("[", "").replace("]", "")
            return f"\n![{caption}]({url})\n"
        except StopIteration:
            return ""

    return re.sub(r"<!--\s*image\s*-->", _replacer, markdown)

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = settings.BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@router.get("/workspace/{workspace_id}", response_model=list[DocumentResponse])
async def list_documents(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Liệt kê toàn bộ document trong một knowledge base."""
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    result = await db.execute(
        select(Document).where(Document.workspace_id == workspace_id).order_by(Document.created_at.desc())
    )
    return result.scalars().all()


async def process_document_background(document_id: int, file_path: str, workspace_id: int):
    """Background task xử lý document để RAG indexing."""
    from app.core.database import async_session_maker
    from app.services.rag_service import get_rag_service

    async with async_session_maker() as db:
        try:
            # Tải cấu hình KG ở cấp workspace
            from sqlalchemy import select as sa_select
            from app.models.knowledge_base import KnowledgeBase
            ws_result = await db.execute(
                sa_select(
                    KnowledgeBase.kg_language,
                    KnowledgeBase.kg_entity_types,
                    KnowledgeBase.chunk_size,
                    KnowledgeBase.chunk_overlap,
                )
                .where(KnowledgeBase.id == workspace_id)
            )
            ws_row = ws_result.one_or_none()
            kg_language = ws_row.kg_language if ws_row else None
            kg_entity_types = ws_row.kg_entity_types if ws_row else None
            chunk_size = ws_row.chunk_size if ws_row else None
            chunk_overlap = ws_row.chunk_overlap if ws_row else None

            rag_service = get_rag_service(
                db, workspace_id,
                kg_language=kg_language,
                kg_entity_types=kg_entity_types,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            await rag_service.process_document(document_id, file_path)
            logger.info(f"Document {document_id} processed successfully")
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            # Đảm bảo trạng thái FAILED ngay cả khi handler của process_document lỗi
            try:
                from sqlalchemy import select, update
                from app.models.document import Document, DocumentStatus
                result = await db.execute(
                    select(Document.status).where(Document.id == document_id)
                )
                current_status = result.scalar_one_or_none()
                if current_status and current_status != DocumentStatus.FAILED:
                    await db.execute(
                        update(Document)
                        .where(Document.id == document_id)
                        .values(
                            status=DocumentStatus.FAILED,
                            error_message=str(e)[:500],
                        )
                    )
                    await db.commit()
            except Exception as recovery_err:
                logger.error(f"Failed to set FAILED status for doc {document_id}: {recovery_err}")


@router.post("/upload/{workspace_id}", response_model=DocumentUploadResponse)
async def upload_document(
    workspace_id: int,
    file: UploadFile = File(...),
    custom_metadata: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload document vào knowledge base. Việc xử lý cần được trigger riêng."""
    
    parsed_metadata = None
    if custom_metadata:
        try:
            raw_metadata = json.loads(custom_metadata)
            if not isinstance(raw_metadata, list):
                raise ValueError("Metadata must be a list of key-value objects")
            
            parsed_metadata = {}
            for item in raw_metadata:
                if not isinstance(item, dict) or "key" not in item or "value" not in item:
                    raise ValueError("Each metadata item must contain 'key' and 'value' fields")
                parsed_metadata[item["key"]] = item["value"]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid custom_metadata format: {e}"
            )
            
    result = await db.execute(select(KnowledgeBase).where(KnowledgeBase.id == workspace_id))
    kb = result.scalar_one_or_none()

    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}"
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )

    filename = f"{uuid.uuid4()}{ext}"
    file_path = UPLOAD_DIR / filename

    import aiofiles
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    document = Document(
        workspace_id=workspace_id,
        filename=filename,
        original_filename=file.filename,
        file_type=ext[1:],
        file_size=len(content),
        status=DocumentStatus.PENDING,
        custom_metadata=parsed_metadata,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    return DocumentUploadResponse(
        id=document.id,
        filename=document.original_filename,
        status=document.status,
        message="Document uploaded. Click 'Process' to extract and index content."
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Lấy document theo ID"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    return document


@router.get("/{document_id}/markdown")
async def get_document_markdown(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Lấy toàn bộ structured markdown content của document (đã parse bởi NexusRAG)."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if not document.markdown_content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No markdown content available. Document may not have been processed with NexusRAG."
        )

    markdown = document.markdown_content

    # Lớp safety net: nếu còn image placeholder thì inject reference thực ngay lúc trả về
    if "<!-- image" in markdown:
        img_result = await db.execute(
            select(DocumentImage)
            .where(DocumentImage.document_id == document_id)
            .order_by(DocumentImage.id)
        )
        images = img_result.scalars().all()
        if images:
            markdown = _inject_images_from_db(markdown, images, document.workspace_id)

    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown",
    )


@router.get("/{document_id}/images", response_model=list[DocumentImageResponse])
async def get_document_images(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Liệt kê toàn bộ ảnh đã extract của document."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    result = await db.execute(
        select(DocumentImage)
        .where(DocumentImage.document_id == document_id)
        .order_by(DocumentImage.page_no)
    )
    images = result.scalars().all()

    return [
        DocumentImageResponse(
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            width=img.width,
            height=img.height,
            url=f"/static/doc-images/kb_{document.workspace_id}/images/{img.image_id}.png",
        )
        for img in images
    ]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Xóa document và các chunk của nó khỏi vector store"""
    result = await db.execute(select(Document).where(Document.id == document_id))
    document = result.scalar_one_or_none()

    if document is None:
        raise NotFoundError("Document", document_id)

    if document.status == DocumentStatus.INDEXED:
        try:
            from app.services.rag_service import get_rag_service
            rag_service = get_rag_service(db, document.workspace_id)
            await rag_service.delete_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to delete chunks from vector store: {e}")

    file_path = UPLOAD_DIR / document.filename
    try:
        if file_path.exists():
            os.remove(file_path)
    except OSError as e:
        logger.warning(f"Failed to remove uploaded file {file_path}: {e}")


    await db.delete(document)
    await db.commit()
