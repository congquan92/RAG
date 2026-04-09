"""
Document Service — Xử lý upload file và background ingestion.

Workflow:
  1. User upload file → save_uploaded_file() lưu file tạm + tạo Document record
  2. Background task → process_ingestion_task():
     a. Update status = "processing"
     b. Gọi ingestion.py parse + chunk
     c. Lưu chunks vào ChromaDB (vector store)
     d. Update Document.chunk_count + IngestionTask.status = "completed"
  3. User poll status → get_task_status() kiểm tra tiến trình
  4. User xem documents → list_documents(), get_document(), delete_document()
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import settings
from app.models.document import Document, IngestionTask

logger = logging.getLogger(__name__)


def _get_upload_dir() -> Path:
    """Resolve upload directory from environment-backed settings."""
    return Path(settings.upload_dir)


def _delete_document_chunks_from_chromadb(document_id: str, chunk_count: int) -> None:
    """Best-effort delete of existing document chunks in ChromaDB."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection_name = settings.chroma_collection_name
        existing = [c.name for c in client.list_collections()]

        if collection_name in existing:
            collection = client.get_collection(name=collection_name)
            # Primary path: delete by metadata filter, works even when chunk_count is stale.
            collection.delete(where={"document_id": document_id})

            # Fallback for legacy chunks without metadata.
            if chunk_count > 0:
                chunk_ids = [
                    f"{document_id}_chunk_{i}"
                    for i in range(chunk_count)
                ]
                collection.delete(ids=chunk_ids)

            logger.info("Deleted ChromaDB chunks for document %s", document_id)
    except Exception as exc:
        logger.warning("Failed to delete ChromaDB chunks: %s", exc)


def _cleanup_chroma_persist_dir_if_empty(remaining_documents: int) -> None:
    """Remove Chroma persist directory when there are no documents left."""
    if remaining_documents > 0:
        return

    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        return

    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection_name = settings.chroma_collection_name
        existing = [c.name for c in client.list_collections()]
        if collection_name in existing:
            client.delete_collection(name=collection_name)
            logger.info("Deleted empty ChromaDB collection: %s", collection_name)
    except Exception as exc:
        logger.warning("Failed to delete empty ChromaDB collection: %s", exc)

    try:
        shutil.rmtree(persist_dir)
        logger.info("Removed ChromaDB persist directory: %s", persist_dir)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("Failed to remove ChromaDB persist directory: %s", exc)


# ═════════════════════════════════════════════════════════════════════════════
# Upload & Save
# ═════════════════════════════════════════════════════════════════════════════

async def save_uploaded_file(
    db: AsyncSession,
    filename: str,
    file_content: bytes,
) -> tuple[Document, IngestionTask]:
    """
    Lưu file upload vào disk + tạo Document và IngestionTask records.

    Args:
        db: Async database session
        filename: Tên file gốc từ user
        file_content: Nội dung file (bytes)

    Returns:
        (Document, IngestionTask) — records vừa tạo
    """
    upload_dir = _get_upload_dir()

    # Tạo thư mục upload nếu chưa có
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Tạo unique filename tránh ghi đè
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{filename}"
    file_path = upload_dir / safe_filename

    # Ghi file ra disk
    file_path.write_bytes(file_content)
    file_size = len(file_content)

    # Detect MIME type
    from app.rag.ingestion import detect_mime_type

    mime_type = detect_mime_type(file_path)

    # Tạo Document record
    document = Document(
        filename=filename,
        file_path=str(file_path),
        file_size=file_size,
        mime_type=mime_type,
    )
    db.add(document)
    await db.flush()  # Để lấy document.id

    # Tạo IngestionTask record (status=pending)
    task = IngestionTask(
        document_id=document.id,
        status="pending",
    )
    db.add(task)
    await db.flush()

    logger.info(
        "File saved: %s → %s (size=%d, mime=%s, doc_id=%s, task_id=%s)",
        filename, file_path, file_size, mime_type, document.id, task.id,
    )

    return document, task


# ═════════════════════════════════════════════════════════════════════════════
# Background Ingestion — chạy trong FastAPI BackgroundTasks
# ═════════════════════════════════════════════════════════════════════════════

async def process_ingestion_task(
    task_id: str,
    document_id: str,
    embeddings: Any,
) -> None:
    """
    Background task: Parse file → chunk → lưu ChromaDB.

    Hàm này được gọi từ FastAPI BackgroundTasks, tự quản lý DB session riêng
    (không dùng session từ request vì request đã kết thúc).

    Args:
        task_id: ID của IngestionTask
        document_id: ID của Document
        embeddings: LangChain Embeddings instance (từ app.state)
    """
    from app.core.database import async_session_factory

    async with async_session_factory() as db:
        try:
            # Lấy task và document
            task = await db.get(IngestionTask, task_id)
            document = await db.get(Document, document_id)

            if not task or not document:
                logger.error("Task %s or Document %s not found", task_id, document_id)
                return

            # Update status → processing
            task.status = "processing"
            await db.commit()

            logger.info(
                "Ingestion started: task=%s, file=%s",
                task_id, document.filename,
            )

            # ── Step 1: Parse + Chunk ────────────────────────────────────
            from app.rag.ingestion import ingest_file

            extraction = ingest_file(document.file_path)

            if extraction.error:
                task.status = "failed"
                task.error_message = extraction.error
                await db.commit()
                logger.error("Ingestion failed: %s — %s", document.filename, extraction.error)
                return

            if not extraction.chunks:
                task.status = "failed"
                task.error_message = "No chunks extracted from file"
                await db.commit()
                logger.warning("No chunks from file: %s", document.filename)
                return

            # ── Step 2: Lưu chunks vào ChromaDB ─────────────────────────
            chunks_stored = await _store_chunks_chromadb(
                chunks=extraction.chunks,
                document_id=document.id,
                filename=document.filename,
                embeddings=embeddings,
            )

            if chunks_stored <= 0:
                task.status = "failed"
                task.error_message = (
                    "Không lưu được chunk vào vector index. "
                    "Vui lòng kiểm tra log ở server."
                )
                await db.commit()
                logger.error(
                    "Ingestion failed after extraction: task=%s, file=%s, chunks_extracted=%d",
                    task_id,
                    document.filename,
                    len(extraction.chunks),
                )
                return

            # ── Step 3: Update DB records ────────────────────────────────
            document.chunk_count = chunks_stored
            task.status = "completed"
            task.chunks_processed = chunks_stored
            task.error_message = None
            await db.commit()

            logger.info(
                "Ingestion completed: task=%s, file=%s, chunks=%d",
                task_id, document.filename, chunks_stored,
            )

        except Exception as exc:
            logger.error("Ingestion task %s crashed: %s", task_id, exc, exc_info=True)
            try:
                task = await db.get(IngestionTask, task_id)
                if task:
                    task.status = "failed"
                    task.error_message = f"Unexpected error: {exc}"
                    await db.commit()
            except Exception:
                await db.rollback()


async def _store_chunks_chromadb(
    chunks: list,
    document_id: str,
    filename: str,
    embeddings: Any,
) -> int:
    """
    Lưu chunks vào ChromaDB collection.

    Args:
        chunks: List[DocumentChunk] từ ingestion
        document_id: ID document trong SQLite
        filename: Tên file gốc
        embeddings: LangChain Embeddings instance

    Returns:
        Số chunks đã lưu thành công
    """
    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection_name = settings.chroma_collection_name

        # Tạo hoặc lấy collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "RAG document chunks"},
        )

        # Chuẩn bị data
        texts = [chunk.text for chunk in chunks]
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "document_id": document_id,
                "source_file": filename,
                "chunk_index": i,
                **{k: str(v) for k, v in chunk.metadata.items() if k not in ("document_id", "source_file", "chunk_index")},
            }
            for i, chunk in enumerate(chunks)
        ]

        # Embed texts
        text_embeddings = embeddings.embed_documents(texts)

        # Upsert vào ChromaDB (batch)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=text_embeddings,
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d chunks in ChromaDB for document %s (%s)",
            len(chunks), document_id, filename,
        )
        return len(chunks)

    except ImportError:
        logger.error("chromadb not installed, cannot store chunks")
        return 0
    except Exception as exc:
        logger.error("ChromaDB storage failed for document %s: %s", document_id, exc)
        return 0


# ═════════════════════════════════════════════════════════════════════════════
# Query helpers — dùng cho API controllers
# ═════════════════════════════════════════════════════════════════════════════

async def get_task_status(
    db: AsyncSession,
    task_id: str,
) -> Optional[IngestionTask]:
    """Lấy trạng thái ingestion task theo ID."""
    return await db.get(IngestionTask, task_id)


async def get_document(
    db: AsyncSession,
    document_id: str,
) -> Optional[Document]:
    """Lấy thông tin document theo ID."""
    return await db.get(Document, document_id)


async def list_documents(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[Document], int]:
    """
    Liệt kê documents với pagination.

    Returns:
        (documents, total_count)
    """
    # Count total
    count_stmt = select(func.count(Document.id))
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Fetch documents
    stmt = (
        select(Document)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    documents = list(result.scalars().all())

    return documents, total


async def delete_document(
    db: AsyncSession,
    document_id: str,
) -> Optional[Document]:
    """
    Xóa document + file vật lý + chunks trong ChromaDB.

    Returns:
        Document đã xóa (hoặc None nếu không tìm thấy)
    """
    document = await db.get(Document, document_id)
    if not document:
        return None

    # Xóa file vật lý
    file_path = Path(document.file_path)
    if file_path.exists():
        file_path.unlink()
        logger.info("Deleted file: %s", file_path)

    # Xóa chunks khỏi ChromaDB
    _delete_document_chunks_from_chromadb(document_id, document.chunk_count)

    # Xóa DB record (cascade xóa luôn ingestion_tasks)
    await db.delete(document)

    # Flush để count phản ánh state sau khi delete trong transaction hiện tại.
    await db.flush()

    total_result = await db.execute(select(func.count(Document.id)))
    remaining_documents = total_result.scalar() or 0
    _cleanup_chroma_persist_dir_if_empty(int(remaining_documents))

    logger.info(
        "Document deleted: id=%s, filename=%s, remaining_documents=%d",
        document_id,
        document.filename,
        remaining_documents,
    )
    return document
