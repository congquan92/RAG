"""
Service RAG (Retrieval-Augmented Generation).
Service chính điều phối xử lý tài liệu, indexing và retrieval.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import Document, DocumentStatus
from app.services.document_loader import load_document, LoadedDocument
from app.services.chunker import DocumentChunker, TextChunk
from app.services.embedder import EmbeddingService, get_embedding_service
from app.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Đại diện cho một chunk đã truy xuất cùng relevance score."""
    content: str
    metadata: dict
    score: float  # Giá trị càng thấp thì càng tương đồng (distance)
    chunk_id: str


@dataclass
class RAGQueryResult:
    """Kết quả của một truy vấn RAG."""
    chunks: list[RetrievedChunk]
    context: str  # Các chunk đã nối lại để làm context cho LLM
    query: str


class RAGService:
    """
    Service RAG chính xử lý document processing và retrieval.
    """

    def __init__(
        self,
        db: AsyncSession,
        workspace_id: int,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Khởi tạo service RAG.

        Args:
            db: Database session
            workspace_id: Knowledge base ID để tách biệt dữ liệu
            chunk_size: Kích thước text chunk
            chunk_overlap: Độ chồng lấp giữa các chunk
        """
        self.db = db
        self.workspace_id = workspace_id
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = get_embedding_service()
        self.vector_store = get_vector_store(workspace_id)

    async def process_document(self, document_id: int, file_path: str) -> int:
        """
        Xử lý một tài liệu: load, chunk, embed và lưu.

        Args:
            document_id: Document ID trong database
            file_path: Đường dẫn đến file tài liệu

        Returns:
            Số lượng chunk được tạo

        Raises:
            ValueError: Nếu xử lý tài liệu thất bại
        """
        # Lấy document từ DB
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if document is None:
            raise ValueError(f"Document {document_id} not found")

        try:
            # Cập nhật trạng thái sang processing
            document.status = DocumentStatus.PROCESSING
            await self.db.commit()

            import asyncio

            def _process_sync():
                # Load tài liệu
                logger.info(f"Loading document {document_id} from {file_path}")
                loaded = load_document(file_path)

                # Chia nhỏ text thành chunk
                logger.info(f"Chunking document {document_id}")
                chunks = self.chunker.split_text(
                    text=loaded.content,
                    source=document.original_filename,
                    extra_metadata={
                        "document_id": document_id,
                        "file_type": loaded.file_type,
                        "page_count": loaded.page_count
                    }
                )

                if not chunks:
                    return []

                # Tạo embeddings
                logger.info(f"Generating embeddings for {len(chunks)} chunks")
                chunk_texts = [c.content for c in chunks]
                embeddings = self.embedder.embed_texts(chunk_texts)

                # Chuẩn bị dữ liệu cho vector store
                ids = [f"doc_{document_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = []
                for c in chunks:
                    meta = {
                        "document_id": document_id,
                        "chunk_index": c.chunk_index,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "source": c.metadata.get("source", ""),
                        "file_type": c.metadata.get("file_type", "")
                    }
                    if document.custom_metadata:
                        meta.update(document.custom_metadata)
                    metadatas.append(meta)

                # Lưu vào vector database
                logger.info(f"Storing {len(chunks)} chunks in vector store")
                self.vector_store.add_documents(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunk_texts,
                    metadatas=metadatas
                )
                return chunks

            # Chạy phần code đồng bộ chặn CPU/IO trong thread pool
            chunks = await asyncio.to_thread(_process_sync)

            if not chunks:
                document.status = DocumentStatus.INDEXED
                document.chunk_count = 0
                await self.db.commit()
                logger.warning(f"Document {document_id} produced no chunks (empty content)")
                return 0


            # Cập nhật trạng thái document
            document.status = DocumentStatus.INDEXED
            document.chunk_count = len(chunks)
            await self.db.commit()

            logger.info(f"Successfully processed document {document_id}: {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)[:500]
            await self.db.commit()
            raise

    async def delete_document(self, document_id: int) -> None:
        """
        Xóa các chunk của tài liệu khỏi vector store.

        Args:
            document_id: Document ID trong database
        """
        self.vector_store.delete_by_document_id(document_id)
        logger.info(f"Deleted document {document_id} from vector store")

    def query(
        self,
        question: str,
        top_k: int = 5,
        document_ids: list[int] | None = None
    ) -> RAGQueryResult:
        """
        Truy vấn vector store để lấy các chunk liên quan.

        Args:
            question: Câu hỏi truy vấn
            top_k: Số lượng chunk cần truy xuất
            document_ids: Bộ lọc tùy chọn theo danh sách tài liệu

        Returns:
            RAGQueryResult chứa chunk đã truy xuất và context đã ghép
        """
        # Tạo query embedding
        query_embedding = self.embedder.embed_query(question)

        # Tạo bộ lọc
        where = None
        if document_ids:
            where = {"document_id": {"$in": document_ids}}

        # Truy vấn vector store
        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where
        )

        # Dựng danh sách chunk đã truy xuất
        chunks = []
        for i, doc in enumerate(results["documents"]):
            chunks.append(RetrievedChunk(
                content=doc,
                metadata=results["metadatas"][i] if results["metadatas"] else {},
                score=results["distances"][i] if results["distances"] else 0.0,
                chunk_id=results["ids"][i] if results["ids"] else ""
            ))

        # Sắp xếp theo score (distance thấp hơn = tương đồng hơn)
        chunks.sort(key=lambda x: x.score)

        # Ghép context
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", "Unknown")
            context_parts.append(f"[Source: {source}, Chunk {i+1}]\n{chunk.content}")

        context = "\n\n---\n\n".join(context_parts)

        return RAGQueryResult(
            chunks=chunks,
            context=context,
            query=question
        )

    def get_chunk_count(self) -> int:
        """Trả về tổng số chunk trong vector store của knowledge base."""
        return self.vector_store.count()


def get_rag_service(
    db: AsyncSession,
    workspace_id: int,
    kg_language: str | None = None,
    kg_entity_types: list[str] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> "RAGService | NexusRAGService":
    """Factory function: điều hướng tới NexusRAGService hoặc RAGService cũ theo config."""
    from app.core.config import settings

    if settings.NEXUSRAG_ENABLED:
        from app.services.nexus_rag_service import NexusRAGService
        return NexusRAGService(
            db=db,
            workspace_id=workspace_id,
            kg_language=kg_language,
            kg_entity_types=kg_entity_types,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    effective_chunk_size = chunk_size if chunk_size is not None else 500
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else 50
    return RAGService(
        db=db,
        workspace_id=workspace_id,
        chunk_size=effective_chunk_size,
        chunk_overlap=effective_chunk_overlap,
    )
