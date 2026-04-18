"""
Deep Retriever
==============

Hybrid retrieval kết hợp Knowledge Graph (LightRAG) + Vector Search (ChromaDB)
+ Cross-encoder Reranking (bge-reranker-v2-m3).

Pipeline:
    1. KG query  (song song) -> tóm tắt entity/relationship
    2. Vector search -> over-fetch top-N ứng viên (NEXUSRAG_VECTOR_PREFETCH)
    3. Cross-encoder rerank -> lọc chính xác tới top-K (NEXUSRAG_RERANKER_TOP_K)
    4. Gộp với citation + image references tùy chọn
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.document import Document, DocumentImage, DocumentTable
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.reranker import RerankerService, get_reranker_service
from app.services.models.parsed_document import (
    Citation,
    DeepRetrievalResult,
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
)

logger = logging.getLogger(__name__)


class DeepRetriever:
    """
    Hybrid retriever: KG traversal + vector similarity + cross-encoder reranking.
    """

    def __init__(
        self,
        workspace_id: int,
        kg_service: Optional[KnowledgeGraphService],
        vector_store: VectorStore,
        embedder: EmbeddingService,
        db: Optional[AsyncSession] = None,
        reranker: Optional[RerankerService] = None,
    ):
        self.workspace_id = workspace_id
        self.kg_service = kg_service
        self.vector_store = vector_store
        self.embedder = embedder
        self.db = db
        self.reranker = reranker or get_reranker_service()

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 5,
        document_ids: Optional[list[int]] = None,
        include_images: bool = True,
        metadata_filter: dict | None = None,
    ) -> DeepRetrievalResult:
        """
        Chạy hybrid retrieval với reranking.

                Luồng xử lý:
                    1. [song song] KG query + Vector over-fetch (NEXUSRAG_VECTOR_PREFETCH)
                    2. Cross-encoder rerank kết quả vector -> top_k cuối cùng
                    3. Tùy chọn tìm image liên quan từ các trang của chunk
                    4. Ghép context có cấu trúc cho LLM

        Args:
            question: Truy vấn ngôn ngữ tự nhiên
            mode: "hybrid" (default), "naive", "local", "global", "vector_only"
            top_k: Số lượng chunk cuối cùng trả về (sau reranking)
            document_ids: Bộ lọc tùy chọn theo tài liệu cụ thể
            include_images: Có tìm image liên quan hay không

        Returns:
            DeepRetrievalResult gồm chunk, citation, context và image tùy chọn
        """
        # Áp dụng trần rerank toàn cục từ settings để env NEXUSRAG_RERANKER_TOP_K
        # luôn tác động runtime, nhưng vẫn cho phép caller yêu cầu ít hơn.
        effective_top_k = min(top_k, settings.NEXUSRAG_RERANKER_TOP_K)
        if effective_top_k < top_k:
            logger.info(
                "Requested top_k=%s capped by NEXUSRAG_RERANKER_TOP_K=%s",
                top_k,
                settings.NEXUSRAG_RERANKER_TOP_K,
            )

        # Chạy KG và vector search song song
        kg_task = None
        if self.kg_service and mode != "vector_only":
            kg_task = asyncio.create_task(
                self._kg_query(question, mode)
            )

        # Over-fetch từ vector DB để reranking
        prefetch_k = max(settings.NEXUSRAG_VECTOR_PREFETCH, effective_top_k * 3)
        vector_task = asyncio.create_task(
            asyncio.to_thread(
                self._vector_query, question, prefetch_k, document_ids, metadata_filter
            )
        )

        # Đợi kết quả
        kg_summary = ""
        if kg_task:
            try:
                kg_summary = await kg_task
            except Exception as e:
                logger.warning(f"KG query failed, continuing with vector only: {e}")

        raw_chunks, raw_citations = await vector_task

        # Rerank: chấm điểm bằng cross-encoder để tăng độ chính xác
        chunks, citations = await asyncio.to_thread(
            self._rerank_chunks, question, raw_chunks, raw_citations, effective_top_k
        )

        # Tìm image và table liên quan
        image_refs = []
        table_refs = []
        if include_images and self.db and chunks:
            page_nos = {(c.document_id, c.page_no) for c in chunks if c.page_no > 0}
            if page_nos:
                image_refs, table_refs = await asyncio.gather(
                    self._find_related_images(page_nos),
                    self._find_related_tables(page_nos),
                )

        # Ghép context
        context = self._assemble_context(chunks, citations, kg_summary, image_refs, table_refs)

        return DeepRetrievalResult(
            chunks=chunks,
            citations=citations,
            context=context,
            query=question,
            mode=mode,
            knowledge_graph_summary=kg_summary,
            image_refs=image_refs,
            table_refs=table_refs,
        )

    async def _kg_query(self, question: str, mode: str) -> str:
        """Lấy KG context thô (entities + relationships) liên quan câu hỏi.

        Dùng dữ liệu graph factual thay vì narrative do LLM sinh để tránh
        hallucination từ aquery() của LightRAG.
        """
        if not self.kg_service:
            return ""
        try:
            return await asyncio.wait_for(
                self.kg_service.get_relevant_context(question),
                timeout=settings.NEXUSRAG_KG_QUERY_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("KG raw context retrieval timed out")
            return ""
        except Exception as e:
            logger.warning(f"KG raw context failed: {e}")
            return ""

    def _vector_query(
        self,
        question: str,
        top_k: int,
        document_ids: Optional[list[int]],
        metadata_filter: dict | None = None,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """Vector search đồng bộ qua ChromaDB (giai đoạn over-fetch)."""
        query_embedding = self.embedder.embed_query(question)

        # Gộp metadata_filter và document_ids
        where = metadata_filter.copy() if metadata_filter else {}
        if document_ids:
            where["document_id"] = {"$in": document_ids}
            
        if not where:
            where = None

        results = self.vector_store.query(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where,
        )

        chunks = []
        citations = []

        for i, doc_text in enumerate(results.get("documents", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}

            heading_path = []
            heading_str = meta.get("heading_path", "")
            if heading_str:
                heading_path = heading_str.split(" > ") if isinstance(heading_str, str) else []

            image_refs = []
            image_ids_str = meta.get("image_ids", "")
            if image_ids_str and isinstance(image_ids_str, str):
                image_refs = [iid for iid in image_ids_str.split("|") if iid]

            table_refs = []
            table_ids_str = meta.get("table_ids", "")
            if table_ids_str and isinstance(table_ids_str, str):
                table_refs = [tid for tid in table_ids_str.split("|") if tid]

            chunk = EnrichedChunk(
                content=doc_text,
                chunk_index=meta.get("chunk_index", i),
                source_file=meta.get("source", ""),
                document_id=meta.get("document_id", 0),
                page_no=meta.get("page_no", 0),
                heading_path=heading_path,
                image_refs=image_refs,
                table_refs=table_refs,
                has_table=meta.get("has_table", False),
                has_code=meta.get("has_code", False),
            )
            chunks.append(chunk)

            citations.append(Citation(
                source_file=meta.get("source", "Unknown"),
                document_id=meta.get("document_id", 0),
                page_no=meta.get("page_no", 0),
                heading_path=heading_path,
            ))

        return chunks, citations

    def _rerank_chunks(
        self,
        question: str,
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        top_k: int,
    ) -> tuple[list[EnrichedChunk], list[Citation]]:
        """
        Cross-encoder reranking: chấm điểm đồng thời từng cặp (query, chunk),
        sau đó lọc theo relevance threshold và trả về top_k.
        """
        if not chunks:
            return [], []

        # Trích xuất text cho bước reranking
        doc_texts = [c.content for c in chunks]

        reranked = self.reranker.rerank(
            query=question,
            documents=doc_texts,
            top_k=top_k,
            min_score=settings.NEXUSRAG_MIN_RELEVANCE_SCORE,
        )

        if not reranked:
            # Fallback: nếu reranker lọc hết, giữ top 3 theo thứ tự ban đầu
            fallback_k = min(top_k, 3)
            logger.warning(
                f"Reranker filtered all {len(chunks)} chunks below threshold "
                f"{settings.NEXUSRAG_MIN_RELEVANCE_SCORE}, falling back to top {fallback_k}"
            )
            return chunks[:min(fallback_k, len(chunks))], citations[:min(fallback_k, len(citations))]

        # Ánh xạ kết quả rerank về chunk/citation gốc
        reranked_chunks = [chunks[r.index] for r in reranked]
        reranked_citations = [citations[r.index] for r in reranked]

        logger.info(
            f"Reranked {len(chunks)} → {len(reranked)} chunks "
            f"(scores: {reranked[0].score:.3f} → {reranked[-1].score:.3f})"
        )

        return reranked_chunks, reranked_citations

    async def _find_related_images(
        self,
        page_refs: set[tuple[int, int]],  # (document_id, page_no)
    ) -> list[ExtractedImage]:
        """Tìm image ở đúng các trang trùng với chunk đã truy xuất."""
        if not self.db:
            return []

        images = []
        for doc_id, page_no in page_refs:
            result = await self.db.execute(
                select(DocumentImage).where(
                    DocumentImage.document_id == doc_id,
                    DocumentImage.page_no == page_no,
                )
            )
            for img in result.scalars().all():
                images.append(ExtractedImage(
                    image_id=img.image_id,
                    document_id=img.document_id,
                    page_no=img.page_no,
                    file_path=img.file_path,
                    caption=img.caption,
                    width=img.width,
                    height=img.height,
                    mime_type=img.mime_type,
                ))

        # Deduplicate theo image_id
        seen = set()
        unique = []
        for img in images:
            if img.image_id not in seen:
                seen.add(img.image_id)
                unique.append(img)

        return unique

    async def _find_related_tables(
        self,
        page_refs: set[tuple[int, int]],
    ) -> list[ExtractedTable]:
        """Tìm table ở đúng các trang trùng với chunk đã truy xuất."""
        if not self.db:
            return []

        tables = []
        for doc_id, page_no in page_refs:
            result = await self.db.execute(
                select(DocumentTable).where(
                    DocumentTable.document_id == doc_id,
                    DocumentTable.page_no == page_no,
                )
            )
            for tbl in result.scalars().all():
                tables.append(ExtractedTable(
                    table_id=tbl.table_id,
                    document_id=tbl.document_id,
                    page_no=tbl.page_no,
                    content_markdown=tbl.content_markdown,
                    caption=tbl.caption,
                    num_rows=tbl.num_rows,
                    num_cols=tbl.num_cols,
                ))

        # Deduplicate theo table_id
        seen = set()
        unique = []
        for tbl in tables:
            if tbl.table_id not in seen:
                seen.add(tbl.table_id)
                unique.append(tbl)

        return unique

    @staticmethod
    def _assemble_context(
        chunks: list[EnrichedChunk],
        citations: list[Citation],
        kg_summary: str,
        image_refs: list[ExtractedImage],
        table_refs: list[ExtractedTable] | None = None,
    ) -> str:
        """Ghép chuỗi context có cấu trúc cho LLM."""
        parts = []

        # KG insights
        if kg_summary:
            parts.append("## Knowledge Graph Insights")
            parts.append(kg_summary)
            parts.append("")

        # Chunk đã truy xuất kèm citation
        if chunks:
            parts.append("## Retrieved Document Sections")
            for i, (chunk, citation) in enumerate(zip(chunks, citations)):
                parts.append(f"### [{i + 1}] {citation.format()}")
                parts.append(chunk.content)
                parts.append("")

        # Image khả dụng
        if image_refs:
            parts.append("## Available Document Images")
            for img in image_refs:
                caption_str = f': "{img.caption}"' if img.caption else ""
                parts.append(
                    f"- Image p.{img.page_no}{caption_str} (id: {img.image_id})"
                )
            parts.append("")

        # Table khả dụng
        if table_refs:
            parts.append("## Available Document Tables")
            for tbl in table_refs:
                caption_str = f': "{tbl.caption}"' if tbl.caption else ""
                parts.append(
                    f"- Table p.{tbl.page_no} ({tbl.num_rows}x{tbl.num_cols}){caption_str}"
                )
            parts.append("")

        if not parts:
            return "No relevant documents found for this query."

        return "\n".join(parts)
