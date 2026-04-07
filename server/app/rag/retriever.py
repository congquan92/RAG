"""
Tri-Search Retriever — Tìm kiếm đa luồng + Reranking.

3 luồng search song song:
  1. Semantic Search  — ChromaDB vector similarity
  2. Keyword Search   — BM25 (term frequency)
  3. Graph Search     — LightRAG knowledge graph

Kết quả merge → FlashRank reranking → Top-K chunks.

Tất cả AI models (embeddings, reranker) được lấy từ app.state
(đã load sẵn 1 lần ở lifespan startup).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.settings import settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedChunk:
    """Một chunk đã retrieve, kèm score và metadata nguồn."""

    text: str
    score: float = 0.0
    source: str = ""       # "semantic" | "keyword" | "graph"
    document_id: str = ""
    filename: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Kết quả tổng hợp từ tri-search + reranking."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    raw_semantic: list[RetrievedChunk] = field(default_factory=list)
    raw_keyword: list[RetrievedChunk] = field(default_factory=list)
    raw_graph: list[RetrievedChunk] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Semantic Search — ChromaDB
# ═════════════════════════════════════════════════════════════════════════════

def search_semantic(
    query: str,
    embeddings: Any,
    top_k: int | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    """
    Vector similarity search qua ChromaDB.

    Args:
        query: Câu hỏi user
        embeddings: LangChain Embeddings instance (từ app.state)
        top_k: Số kết quả trả về
        chroma_dir: Thư mục persist ChromaDB
        collection_name: Tên collection trong ChromaDB
    """
    _top_k = top_k or settings.retrieval_top_k
    _chroma_dir = chroma_dir or settings.chroma_persist_dir
    _collection_name = collection_name or settings.chroma_collection_name

    try:
        import chromadb

        client = chromadb.PersistentClient(path=_chroma_dir)

        # Kiểm tra collection tồn tại
        existing = [c.name for c in client.list_collections()]
        if _collection_name not in existing:
            logger.info("ChromaDB collection '%s' not found, returning empty", _collection_name)
            return []

        collection = client.get_collection(name=_collection_name)

        # Embed query
        query_embedding = embeddings.embed_query(query)

        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=_top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB distance → similarity score (1 - normalized distance)
                similarity = max(0.0, 1.0 - dist)
                chunks.append(
                    RetrievedChunk(
                        text=doc,
                        score=similarity,
                        source="semantic",
                        document_id=meta.get("document_id", ""),
                        filename=meta.get("source_file", ""),
                        metadata=meta,
                    )
                )

        logger.info("Semantic search returned %d results for query", len(chunks))
        return chunks

    except ImportError:
        logger.error("chromadb not installed")
        return []
    except Exception as exc:
        logger.error("Semantic search failed: %s", exc)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# 2. Keyword Search — BM25
# ═════════════════════════════════════════════════════════════════════════════

def search_keyword(
    query: str,
    top_k: int | None = None,
    chroma_dir: str | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    """
    BM25-style keyword search.

    Sử dụng ChromaDB documents đã lưu, lấy toàn bộ rồi rank bằng BM25.
    Phù hợp cho tìm kiếm exact match / terminology cụ thể.
    """
    _top_k = top_k or settings.retrieval_top_k
    _chroma_dir = chroma_dir or settings.chroma_persist_dir
    _collection_name = collection_name or settings.chroma_collection_name

    try:
        import chromadb

        client = chromadb.PersistentClient(path=_chroma_dir)

        existing = [c.name for c in client.list_collections()]
        if _collection_name not in existing:
            logger.info("ChromaDB collection '%s' not found for keyword search", _collection_name)
            return []

        collection = client.get_collection(name=_collection_name)

        # Lấy tất cả documents từ collection
        all_docs = collection.get(include=["documents", "metadatas"])
        if not all_docs or not all_docs["documents"]:
            return []

        documents = all_docs["documents"]
        metadatas = all_docs["metadatas"]
        doc_ids = all_docs["ids"]

        # BM25 ranking
        from rank_bm25 import BM25Okapi

        tokenized_docs = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)

        # Lấy top-K theo score
        scored_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:_top_k]

        chunks: list[RetrievedChunk] = []
        for idx in scored_indices:
            if scores[idx] > 0:  # Chỉ lấy kết quả có match
                meta = metadatas[idx] if metadatas else {}
                chunks.append(
                    RetrievedChunk(
                        text=documents[idx],
                        score=float(scores[idx]),
                        source="keyword",
                        document_id=meta.get("document_id", doc_ids[idx]),
                        filename=meta.get("source_file", ""),
                        metadata=meta,
                    )
                )

        logger.info("Keyword (BM25) search returned %d results", len(chunks))
        return chunks

    except ImportError:
        logger.warning("rank_bm25 not installed, keyword search disabled")
        return []
    except Exception as exc:
        logger.error("Keyword search failed: %s", exc)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# 3. Graph Search — LightRAG
# ═════════════════════════════════════════════════════════════════════════════

def search_graph(
    query: str,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Knowledge graph search qua LightRAG.
    Tìm quan hệ giữa các entities trong tài liệu → context phong phú hơn.
    """
    _top_k = top_k or settings.retrieval_top_k

    try:
        from lightrag import LightRAG as LightRAGEngine
        from lightrag import QueryParam

        rag = LightRAGEngine(working_dir=settings.lightrag_working_dir)

        result = rag.query(
            query,
            param=QueryParam(mode="hybrid", top_k=_top_k),
        )

        # LightRAG trả về string, wrap thành chunk
        if result and result.strip():
            chunks = [
                RetrievedChunk(
                    text=result,
                    score=1.0,  # Graph result không có score cụ thể
                    source="graph",
                    filename="knowledge_graph",
                )
            ]
            logger.info("Graph search returned %d result(s)", len(chunks))
            return chunks

        logger.info("Graph search returned no results")
        return []

    except ImportError:
        logger.warning("lightrag not installed, graph search disabled")
        return []
    except Exception as exc:
        logger.error("Graph search failed: %s", exc)
        return []


# ═════════════════════════════════════════════════════════════════════════════
# FlashRank Reranking
# ═════════════════════════════════════════════════════════════════════════════

def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    reranker: Any,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Rerank merged results bằng FlashRank cross-encoder.

    Args:
        query: Câu hỏi gốc
        chunks: Danh sách chunks đã merge từ 3 nguồn
        reranker: FlashRank Ranker instance (từ app.state)
        top_k: Số kết quả cuối cùng giữ lại
    """
    _top_k = top_k or settings.reranker_top_k

    if not chunks:
        return []

    if reranker is None:
        logger.warning("Reranker not available, returning chunks sorted by original score")
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:_top_k]

    try:
        from flashrank import RerankRequest

        # Chuẩn bị input cho FlashRank
        passages = [
            {"id": str(i), "text": chunk.text, "meta": chunk.metadata}
            for i, chunk in enumerate(chunks)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked = reranker.rerank(rerank_request)

        # Map lại score từ reranker
        result: list[RetrievedChunk] = []
        for item in reranked[:_top_k]:
            original_idx = int(item["id"])
            chunk = chunks[original_idx]
            chunk.score = float(item["score"])
            result.append(chunk)

        logger.info(
            "Reranked %d → %d chunks (top_k=%d)",
            len(chunks), len(result), _top_k,
        )
        return result

    except ImportError:
        logger.warning("flashrank not installed, skipping reranking")
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:_top_k]
    except Exception as exc:
        logger.error("Reranking failed: %s", exc)
        return sorted(chunks, key=lambda c: c.score, reverse=True)[:_top_k]


# ═════════════════════════════════════════════════════════════════════════════
# Deduplicate — loại bỏ chunks trùng lặp giữa các nguồn
# ═════════════════════════════════════════════════════════════════════════════

def deduplicate_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Loại bỏ chunks trùng nội dung, giữ bản có score cao nhất.
    So sánh bằng stripped text content.
    """
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = chunk.text.strip()[:200]  # Dùng 200 ký tự đầu làm key
        if key not in seen or chunk.score > seen[key].score:
            seen[key] = chunk

    deduped = list(seen.values())
    if len(deduped) < len(chunks):
        logger.info("Deduplicated %d → %d chunks", len(chunks), len(deduped))
    return deduped


# ═════════════════════════════════════════════════════════════════════════════
# Full Retrieval Pipeline
# ═════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    embeddings: Any,
    reranker: Any = None,
    top_k: int | None = None,
    reranker_top_k: int | None = None,
    enable_semantic: bool = True,
    enable_keyword: bool = True,
    enable_graph: bool = True,
) -> RetrievalResult:
    """
    Full Tri-Search retrieval pipeline:
      1. Search song song 3 luồng (Semantic + Keyword + Graph)
      2. Merge + deduplicate
      3. FlashRank reranking → Top-K

    Args:
        query: Câu hỏi user
        embeddings: LangChain Embeddings instance (từ app.state)
        reranker: FlashRank Ranker instance (từ app.state, nullable)
        top_k: Số kết quả mỗi luồng search
        reranker_top_k: Số kết quả cuối cùng sau reranking
        enable_*: Bật/tắt từng luồng search

    Returns:
        RetrievalResult — chunks đã rerank + raw results từng nguồn
    """
    _top_k = top_k or settings.retrieval_top_k
    _reranker_top_k = reranker_top_k or settings.reranker_top_k

    result = RetrievalResult()

    # ── 1. Song song 3 luồng search ─────────────────────────────────
    if enable_semantic:
        result.raw_semantic = search_semantic(query, embeddings, top_k=_top_k)

    if enable_keyword:
        result.raw_keyword = search_keyword(query, top_k=_top_k)

    if enable_graph:
        result.raw_graph = search_graph(query, top_k=_top_k)

    # ── 2. Merge + deduplicate ───────────────────────────────────────
    all_chunks = result.raw_semantic + result.raw_keyword + result.raw_graph

    if not all_chunks:
        logger.info("No results from any search source")
        return result

    merged = deduplicate_chunks(all_chunks)

    logger.info(
        "Merged results: semantic=%d, keyword=%d, graph=%d → %d unique",
        len(result.raw_semantic),
        len(result.raw_keyword),
        len(result.raw_graph),
        len(merged),
    )

    # ── 3. Rerank ────────────────────────────────────────────────────
    result.chunks = rerank_chunks(
        query=query,
        chunks=merged,
        reranker=reranker,
        top_k=_reranker_top_k,
    )

    return result
