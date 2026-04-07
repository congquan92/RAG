"""
Test Script: Retriever Pipeline
Kiểm tra search functions và reranking logic.

Usage:
    cd server
    source .venv/bin/activate
    python scripts/test_retriever.py

Lưu ý: Test này chạy KHÔNG cần Ollama running hay ChromaDB có data.
Nó tạo mock data và test logic thuần túy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_data_structures():
    """Test RetrievedChunk và RetrievalResult."""
    from app.rag.retriever import RetrievedChunk, RetrievalResult

    print("=" * 60)
    print("  TEST: Retriever Data Structures")
    print("=" * 60)

    chunk = RetrievedChunk(
        text="Test chunk content",
        score=0.95,
        source="semantic",
        document_id="doc-001",
        filename="report.pdf",
        metadata={"page": 1},
    )
    print(f"  ✅ RetrievedChunk: text={chunk.text[:30]!r}, score={chunk.score}, source={chunk.source}")

    result = RetrievalResult(
        chunks=[chunk],
        raw_semantic=[chunk],
    )
    print(f"  ✅ RetrievalResult: chunks={len(result.chunks)}, raw_semantic={len(result.raw_semantic)}")
    print()


def test_deduplication():
    """Test deduplication logic."""
    from app.rag.retriever import RetrievedChunk, deduplicate_chunks

    print("=" * 60)
    print("  TEST: Deduplication")
    print("=" * 60)

    chunks = [
        RetrievedChunk(text="Same content here", score=0.9, source="semantic"),
        RetrievedChunk(text="Same content here", score=0.7, source="keyword"),
        RetrievedChunk(text="Different content", score=0.8, source="graph"),
    ]

    deduped = deduplicate_chunks(chunks)

    print(f"  Input:  {len(chunks)} chunks")
    print(f"  Output: {len(deduped)} chunks")

    assert len(deduped) == 2, f"Expected 2 after dedup, got {len(deduped)}"

    # Chunk trùng phải giữ score cao hơn (0.9)
    same_content = [c for c in deduped if c.text == "Same content here"][0]
    assert same_content.score == 0.9, f"Should keep highest score, got {same_content.score}"
    print(f"  ✅ Duplicate kept highest score: {same_content.score}")
    print("  ✅ Deduplication works!\n")


def test_rerank_without_reranker():
    """Test reranking khi không có FlashRank (fallback to score sort)."""
    from app.rag.retriever import RetrievedChunk, rerank_chunks

    print("=" * 60)
    print("  TEST: Reranking (No FlashRank - Fallback)")
    print("=" * 60)

    chunks = [
        RetrievedChunk(text="Low score", score=0.3, source="keyword"),
        RetrievedChunk(text="High score", score=0.95, source="semantic"),
        RetrievedChunk(text="Mid score", score=0.6, source="graph"),
    ]

    reranked = rerank_chunks(
        query="test query",
        chunks=chunks,
        reranker=None,  # Không có reranker
        top_k=2,
    )

    print(f"  Input:  {len(chunks)} chunks")
    print(f"  Output: {len(reranked)} chunks (top_k=2)")

    assert len(reranked) == 2, f"Expected 2 chunks, got {len(reranked)}"
    assert reranked[0].score >= reranked[1].score, "Should be sorted by score desc"
    print(f"  ✅ Top result: '{reranked[0].text}' (score={reranked[0].score})")
    print("  ✅ Fallback reranking works!\n")


def test_empty_search_results():
    """Test retriever khi không có data."""
    from app.rag.retriever import search_semantic, search_keyword, search_graph

    print("=" * 60)
    print("  TEST: Empty Search Results")
    print("=" * 60)

    # Semantic: ChromaDB collection chưa tồn tại → empty
    semantic = search_semantic(
        query="test",
        embeddings=None,  # Sẽ fail gracefully
        chroma_dir="./data/nonexistent_chroma",
    )
    print(f"  Semantic (no collection): {len(semantic)} results")

    # Keyword: ChromaDB trống → empty
    keyword = search_keyword(
        query="test",
        chroma_dir="./data/nonexistent_chroma",
    )
    print(f"  Keyword  (no collection): {len(keyword)} results")

    # Graph: LightRAG chưa init → empty (sẽ fail gracefully)
    graph = search_graph(query="test")
    print(f"  Graph    (no data):       {len(graph)} results")

    print("  ✅ All search functions handle empty state gracefully!\n")


def test_full_retrieve_pipeline():
    """Test full retrieve() pipeline với mock data."""
    from app.rag.retriever import retrieve

    print("=" * 60)
    print("  TEST: Full Retrieve Pipeline (All Sources Empty)")
    print("=" * 60)

    result = retrieve(
        query="test query",
        embeddings=None,  # Sẽ fail gracefully cho semantic
        reranker=None,
        enable_semantic=False,  # Tắt để tránh cần embeddings
        enable_keyword=False,   # Tắt vì chưa có data
        enable_graph=False,     # Tắt vì chưa có data
    )

    print(f"  Final chunks:   {len(result.chunks)}")
    print(f"  Raw semantic:   {len(result.raw_semantic)}")
    print(f"  Raw keyword:    {len(result.raw_keyword)}")
    print(f"  Raw graph:      {len(result.raw_graph)}")

    assert len(result.chunks) == 0, "Should return empty when all sources off"
    print("  ✅ Pipeline handles no-data scenario cleanly!\n")


def test_context_building():
    """Test context building cho generator."""
    from app.rag.retriever import RetrievedChunk
    from app.rag.generator import build_context, extract_citations

    print("=" * 60)
    print("  TEST: Context Building & Citation Extraction")
    print("=" * 60)

    chunks = [
        RetrievedChunk(
            text="Python là ngôn ngữ lập trình phổ biến.",
            score=0.95, source="semantic",
            document_id="doc-001", filename="python_guide.pdf",
        ),
        RetrievedChunk(
            text="FastAPI là framework web hiệu năng cao.",
            score=0.88, source="keyword",
            document_id="doc-002", filename="fastapi_docs.pdf",
        ),
    ]

    context = build_context(chunks)
    citations = extract_citations(chunks)

    print(f"  Context length: {len(context)} chars")
    print(f"  Citations:      {len(citations)} items")
    for cit in citations:
        print(f"    - {cit['filename']} (score={cit['relevance_score']})")

    assert len(context) > 0, "Context should not be empty"
    assert len(citations) == 2, f"Expected 2 citations, got {len(citations)}"
    assert "python_guide.pdf" in context, "Context should contain filename"
    print("  ✅ Context building and citation extraction work!\n")


if __name__ == "__main__":
    test_data_structures()
    test_deduplication()
    test_rerank_without_reranker()
    test_empty_search_results()
    test_full_retrieve_pipeline()
    test_context_building()
    print("🎉 All retriever tests passed!")
