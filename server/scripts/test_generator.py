"""
Test Script: Generator Pipeline
Kiểm tra context building, prompt formatting, và generator logic.

Usage:
    cd server
    source .venv/bin/activate
    python scripts/test_generator.py

Lưu ý: Test cơ bản chạy KHÔNG cần LLM running.
Test với LLM thật cần Ollama hoặc Gemini API key.
Chạy với flag --live để test với LLM thật.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_prompts():
    """Test prompt templates load đúng."""
    from app.rag.prompts import (
        QA_SYSTEM_PROMPT,
        QA_HUMAN_TEMPLATE,
        CONDENSE_QUESTION_PROMPT,
        NO_CONTEXT_RESPONSE,
    )

    print("=" * 60)
    print("  TEST: Prompt Templates")
    print("=" * 60)

    assert "{context}" in QA_SYSTEM_PROMPT, "System prompt must have {context} placeholder"
    assert "{question}" in QA_HUMAN_TEMPLATE, "Human template must have {question} placeholder"
    assert "{chat_history}" in CONDENSE_QUESTION_PROMPT, "Condense prompt must have {chat_history}"
    assert "{question}" in CONDENSE_QUESTION_PROMPT, "Condense prompt must have {question}"
    assert len(NO_CONTEXT_RESPONSE) > 0, "Fallback response must not be empty"

    # Test formatting
    formatted_system = QA_SYSTEM_PROMPT.format(context="Test context here")
    assert "Test context here" in formatted_system
    print(f"  ✅ QA_SYSTEM_PROMPT:       {len(QA_SYSTEM_PROMPT)} chars, placeholder OK")
    print(f"  ✅ QA_HUMAN_TEMPLATE:      {len(QA_HUMAN_TEMPLATE)} chars, placeholder OK")
    print(f"  ✅ CONDENSE_QUESTION_PROMPT: {len(CONDENSE_QUESTION_PROMPT)} chars, placeholder OK")
    print(f"  ✅ NO_CONTEXT_RESPONSE:    '{NO_CONTEXT_RESPONSE[:60]}...'")
    print()


def test_context_building():
    """Test build_context() với mock chunks."""
    from app.rag.retriever import RetrievedChunk
    from app.rag.generator import build_context

    print("=" * 60)
    print("  TEST: Context Building")
    print("=" * 60)

    # Test empty
    empty_context = build_context([])
    assert empty_context == "", "Empty chunks should produce empty context"
    print("  ✅ Empty chunks → empty context")

    # Test with chunks
    chunks = [
        RetrievedChunk(
            text="Machine Learning là lĩnh vực AI sử dụng dữ liệu để học.",
            score=0.95, source="semantic",
            document_id="doc-1", filename="ml_intro.pdf",
        ),
        RetrievedChunk(
            text="Deep Learning là nhánh của ML sử dụng neural networks nhiều lớp.",
            score=0.87, source="keyword",
            document_id="doc-2", filename="dl_guide.pdf",
        ),
    ]

    context = build_context(chunks)
    assert "ml_intro.pdf" in context, "Context must contain source filename"
    assert "0.95" in context, "Context must contain relevance score"
    assert "Machine Learning" in context, "Context must contain chunk text"
    print(f"  ✅ Context with 2 chunks: {len(context)} chars")
    print(f"  ✅ Contains filenames and scores")
    print()


def test_citation_extraction():
    """Test extract_citations() với dedup."""
    from app.rag.retriever import RetrievedChunk
    from app.rag.generator import extract_citations

    print("=" * 60)
    print("  TEST: Citation Extraction")
    print("=" * 60)

    chunks = [
        RetrievedChunk(text="Chunk 1 from doc A", score=0.9, document_id="doc-A", filename="a.pdf"),
        RetrievedChunk(text="Chunk 2 from doc A", score=0.8, document_id="doc-A", filename="a.pdf"),
        RetrievedChunk(text="Chunk from doc B", score=0.7, document_id="doc-B", filename="b.pdf"),
    ]

    citations = extract_citations(chunks)

    # Phải deduplicate theo document_id
    assert len(citations) == 2, f"Expected 2 unique citations, got {len(citations)}"
    print(f"  ✅ Deduped citations: 3 chunks → {len(citations)} citations")

    for cit in citations:
        print(f"    - {cit['filename']} (doc_id={cit['document_id']}, score={cit['relevance_score']})")

    print()


async def test_generate_no_context():
    """Test generator với empty context (fallback response)."""
    from app.rag.generator import generate
    from app.rag.prompts import NO_CONTEXT_RESPONSE

    print("=" * 60)
    print("  TEST: Generate — No Context (Fallback)")
    print("=" * 60)

    result = await generate(
        query="What is Python?",
        chunks=[],
        llm=None,  # Không cần LLM khi chunks trống
    )

    assert result["answer"] == NO_CONTEXT_RESPONSE
    assert result["citations"] == []
    assert result["context_used"] is False
    print(f"  ✅ Fallback response: '{result['answer'][:60]}...'")
    print(f"  ✅ No citations, context_used=False")
    print()


async def test_generate_stream_no_context():
    """Test streaming generator với empty context."""
    import json
    from app.rag.generator import generate_stream

    print("=" * 60)
    print("  TEST: Generate Stream — No Context (Fallback)")
    print("=" * 60)

    events: list[dict] = []
    async for event_str in generate_stream(query="test?", chunks=[], llm=None):
        events.append(json.loads(event_str))

    print(f"  Events received: {len(events)}")
    for evt in events:
        print(f"    - type={evt['type']}, data={str(evt.get('data', ''))[:60]}")

    assert any(e["type"] == "token" for e in events), "Should have token events"
    assert any(e["type"] == "done" for e in events), "Should have done event"
    print("  ✅ Streaming fallback works!\n")


async def test_generate_live():
    """Test với LLM thật (Ollama). Chỉ chạy khi có --live flag."""
    from app.core.settings import settings
    from app.core.llm_factory import get_llm
    from app.rag.retriever import RetrievedChunk
    from app.rag.generator import generate

    print("=" * 60)
    print("  TEST: Generate — LIVE with LLM")
    print("=" * 60)

    llm = get_llm(settings)

    chunks = [
        RetrievedChunk(
            text="Python là ngôn ngữ lập trình bậc cao, ra đời năm 1991 bởi Guido van Rossum. "
                 "Python nổi tiếng với cú pháp đơn giản, dễ đọc, được dùng rộng rãi trong "
                 "web development, data science, và AI.",
            score=0.95, source="semantic",
            document_id="doc-1", filename="python_intro.pdf",
        ),
    ]

    result = await generate(
        query="Python là gì?",
        chunks=chunks,
        llm=llm,
    )

    print(f"  Answer ({len(result['answer'])} chars):")
    print(f"  {result['answer'][:300]}...")
    print(f"  Citations: {len(result['citations'])}")
    print(f"  Context used: {result['context_used']}")
    print("  ✅ Live generation works!\n")


if __name__ == "__main__":
    test_prompts()
    test_context_building()
    test_citation_extraction()

    # Async tests
    asyncio.run(test_generate_no_context())
    asyncio.run(test_generate_stream_no_context())

    # Live test only khi có --live flag
    if "--live" in sys.argv:
        print("\n⚡ Running LIVE tests (requires Ollama running)...\n")
        asyncio.run(test_generate_live())
    else:
        print("ℹ️  Skip live tests. Run with --live to test with actual LLM.\n")

    print("🎉 All generator tests passed!")
