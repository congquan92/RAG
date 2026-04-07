"""
Test Script: Admin UI & Evaluator (Vòng 6)
Kiểm tra logic evaluator: fetch Q&A, build dataset, quick check.

KHÔNG test Streamlit UI trực tiếp (cần browser).
KHÔNG cần Gemini API key — mock critic LLM.
Dùng test DB riêng, seed data giả lập Q&A history.

Usage:
    cd server
    source .venv/bin/activate
    python scripts/test_evaluator.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.database import Base
from app.models.chat import ChatSession, ChatMessage
from app.models.document import Document, IngestionTask  # noqa: F401

# ── Test DB ──────────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///./data/test_eval.db"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ═════════════════════════════════════════════════════════════════════════════
# Seed Data — tạo Q&A history giả lập trong test DB
# ═════════════════════════════════════════════════════════════════════════════

async def seed_test_data():
    """Tạo sessions + messages giả lập cho evaluator test."""
    async with test_session_factory() as db:
        # Session 1: có 2 cặp Q&A với citations
        s1 = ChatSession(title="Python Q&A")
        db.add(s1)
        await db.flush()

        db.add(ChatMessage(
            session_id=s1.id, role="user",
            content="Python là gì?",
        ))
        await db.flush()

        db.add(ChatMessage(
            session_id=s1.id, role="assistant",
            content="Python là ngôn ngữ lập trình bậc cao, dễ học.",
            citations=json.dumps([
                {
                    "document_id": "doc-001",
                    "filename": "python_intro.pdf",
                    "chunk_text": "Python is a high-level programming language...",
                    "relevance_score": 0.92,
                },
                {
                    "document_id": "doc-002",
                    "filename": "languages.pdf",
                    "chunk_text": "Python was created by Guido van Rossum...",
                    "relevance_score": 0.85,
                },
            ], ensure_ascii=False),
        ))
        await db.flush()

        db.add(ChatMessage(
            session_id=s1.id, role="user",
            content="FastAPI khác Flask như nào?",
        ))
        await db.flush()

        db.add(ChatMessage(
            session_id=s1.id, role="assistant",
            content="FastAPI nhanh hơn, hỗ trợ async, tự gen docs.",
            citations=json.dumps([
                {
                    "document_id": "doc-003",
                    "filename": "web_frameworks.pdf",
                    "chunk_text": "FastAPI is a modern web framework for building APIs...",
                    "relevance_score": 0.88,
                },
            ], ensure_ascii=False),
        ))
        await db.flush()

        # Session 2: 1 cặp Q&A KHÔNG có citations
        s2 = ChatSession(title="General Chat")
        db.add(s2)
        await db.flush()

        db.add(ChatMessage(
            session_id=s2.id, role="user",
            content="RAG là gì?",
        ))
        await db.flush()

        db.add(ChatMessage(
            session_id=s2.id, role="assistant",
            content="RAG là Retrieval-Augmented Generation.",
            citations=None,  # Không có citations
        ))
        await db.flush()

        # Session 3: rỗng (không có messages)
        s3 = ChatSession(title="Empty Session")
        db.add(s3)
        await db.flush()

        await db.commit()

        print(f"  Seeded: session1={s1.id[:8]}... (4 msgs)")
        print(f"  Seeded: session2={s2.id[:8]}... (2 msgs)")
        print(f"  Seeded: session3={s3.id[:8]}... (0 msgs)")

        return s1.id, s2.id, s3.id


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

async def test_fetch_qa_pairs_all():
    """Test lấy tất cả Q&A pairs từ DB."""
    print("=" * 60)
    print("  TEST: _fetch_qa_pairs (all sessions)")
    print("=" * 60)

    from admin_ui.evaluator import _fetch_qa_pairs

    # Monkey-patch session factory để dùng test DB
    import app.core.database as db_module
    original_factory = db_module.async_session_factory
    db_module.async_session_factory = test_session_factory

    try:
        qa_pairs = await _fetch_qa_pairs(limit=50)

        print(f"  Total Q&A pairs: {len(qa_pairs)}")
        for i, qa in enumerate(qa_pairs):
            print(f"    [{i}] Q: {qa['question'][:40]}...")
            print(f"        A: {qa['answer'][:40]}...")
            print(f"        Contexts: {len(qa['contexts'])}")

        assert len(qa_pairs) == 3, f"Expected 3 Q&A pairs, got {len(qa_pairs)}"

        # Thứ tự: sessions ordered by updated_at DESC
        # → Session 2 (RAG) trước, rồi Session 1 (Python, FastAPI)
        questions = [qa["question"] for qa in qa_pairs]
        print(f"  Questions order: {questions}")

        # Tìm pair theo content thay vì index cứng
        python_pair = next(qa for qa in qa_pairs if "Python" in qa["question"])
        fastapi_pair = next(qa for qa in qa_pairs if "FastAPI" in qa["question"])
        rag_pair = next(qa for qa in qa_pairs if "RAG" in qa["question"])

        # Python pair: có 2 citations
        assert len(python_pair["contexts"]) == 2
        assert "high-level" in python_pair["contexts"][0]

        # FastAPI pair: có 1 citation
        assert len(fastapi_pair["contexts"]) == 1

        # RAG pair: không có citations → fallback
        assert rag_pair["contexts"] == ["No context retrieved."]

        print("  ✅ Fetch Q&A pairs OK\n")
    finally:
        db_module.async_session_factory = original_factory


async def test_fetch_qa_pairs_by_session(session_ids: list[str]):
    """Test lấy Q&A pairs cho sessions cụ thể."""
    print("=" * 60)
    print("  TEST: _fetch_qa_pairs (specific sessions)")
    print("=" * 60)

    from admin_ui.evaluator import _fetch_qa_pairs
    import app.core.database as db_module
    original_factory = db_module.async_session_factory
    db_module.async_session_factory = test_session_factory

    try:
        # Chỉ lấy session 1 (2 pairs)
        qa_pairs = await _fetch_qa_pairs(session_ids=[session_ids[0]])
        assert len(qa_pairs) == 2, f"Expected 2 pairs from session 1, got {len(qa_pairs)}"
        print(f"  Session 1 only: {len(qa_pairs)} pairs")

        # Chỉ lấy session 2 (1 pair)
        qa_pairs = await _fetch_qa_pairs(session_ids=[session_ids[1]])
        assert len(qa_pairs) == 1, f"Expected 1 pair from session 2, got {len(qa_pairs)}"
        print(f"  Session 2 only: {len(qa_pairs)} pair")

        # Session rỗng (0 pairs)
        qa_pairs = await _fetch_qa_pairs(session_ids=[session_ids[2]])
        assert len(qa_pairs) == 0, f"Expected 0 pairs from empty session, got {len(qa_pairs)}"
        print(f"  Empty session:  {len(qa_pairs)} pairs")

        print("  ✅ Filtered fetch OK\n")
    finally:
        db_module.async_session_factory = original_factory


async def test_fetch_qa_empty_db():
    """Test fetch khi DB hoàn toàn trống."""
    print("=" * 60)
    print("  TEST: _fetch_qa_pairs (empty DB)")
    print("=" * 60)

    from admin_ui.evaluator import _fetch_qa_pairs
    import app.core.database as db_module

    # Tạo DB riêng hoàn toàn trống
    empty_engine = create_async_engine("sqlite+aiosqlite:///./data/test_eval_empty.db")
    empty_factory = async_sessionmaker(bind=empty_engine, class_=AsyncSession, expire_on_commit=False)

    async with empty_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    original_factory = db_module.async_session_factory
    db_module.async_session_factory = empty_factory

    try:
        qa_pairs = await _fetch_qa_pairs()
        assert len(qa_pairs) == 0
        print(f"  Empty DB: {len(qa_pairs)} pairs")
        print("  ✅ Empty DB handled\n")
    finally:
        db_module.async_session_factory = original_factory
        async with empty_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await empty_engine.dispose()
        Path("./data/test_eval_empty.db").unlink(missing_ok=True)


async def test_build_ragas_dataset():
    """Test chuyển Q&A pairs sang Ragas Dataset format."""
    print("=" * 60)
    print("  TEST: _build_ragas_dataset")
    print("=" * 60)

    from admin_ui.evaluator import _build_ragas_dataset

    qa_pairs = [
        {
            "question": "What is Python?",
            "answer": "A programming language.",
            "contexts": ["Python is high-level...", "Created by Guido..."],
            "session_id": "s1",
            "session_title": "Test",
        },
        {
            "question": "What is RAG?",
            "answer": "Retrieval-Augmented Generation.",
            "contexts": ["No context retrieved."],
            "session_id": "s2",
            "session_title": "Test 2",
        },
    ]

    dataset = _build_ragas_dataset(qa_pairs)

    assert len(dataset) == 2
    assert dataset[0]["question"] == "What is Python?"
    assert dataset[0]["answer"] == "A programming language."
    assert len(dataset[0]["contexts"]) == 2
    assert dataset[1]["contexts"] == ["No context retrieved."]

    print(f"  Dataset rows:    {len(dataset)}")
    print(f"  Columns:         {list(dataset.column_names)}")
    print(f"  Row 0 question:  {dataset[0]['question']}")
    print(f"  Row 0 contexts:  {len(dataset[0]['contexts'])}")
    print("  ✅ Build dataset OK\n")


async def test_quick_check():
    """Test quick_check preview function."""
    print("=" * 60)
    print("  TEST: quick_check")
    print("=" * 60)

    from admin_ui.evaluator import quick_check
    import app.core.database as db_module
    original_factory = db_module.async_session_factory
    db_module.async_session_factory = test_session_factory

    try:
        info = await quick_check()

        print(f"  Total pairs:      {info['total_pairs']}")
        print(f"  Sessions:         {len(info['sessions'])}")
        print(f"  Has contexts:     {info['has_contexts']}")
        print(f"  Sample questions: {info['sample_questions']}")

        assert info["total_pairs"] == 3
        assert len(info["sessions"]) == 2  # session 1 + session 2 (session 3 rỗng)
        assert info["has_contexts"] == 2   # 2 pairs có real contexts
        assert len(info["sample_questions"]) == 3

        print("  ✅ Quick check OK\n")
    finally:
        db_module.async_session_factory = original_factory


async def test_evaluate_no_gemini_key():
    """Test evaluate_sessions khi không có GEMINI_API_KEY → error graceful."""
    print("=" * 60)
    print("  TEST: evaluate_sessions (no Gemini key)")
    print("=" * 60)

    from admin_ui.evaluator import evaluate_sessions
    import app.core.database as db_module
    original_factory = db_module.async_session_factory
    db_module.async_session_factory = test_session_factory

    # Mock settings để không có Gemini key
    from unittest.mock import patch
    with patch("admin_ui.evaluator.settings", create=True) as mock_settings:
        mock_settings.gemini_api_key = ""

        try:
            result = await evaluate_sessions(limit=10)

            print(f"  Total questions: {result['total_questions']}")
            print(f"  Error:           {result['error']}")
            print(f"  Scores:          {result['scores']}")

            # Phải có error message về missing Gemini key
            assert result["error"] is not None
            assert "GEMINI_API_KEY" in result["error"] or "dependency" in result["error"].lower()

            print("  ✅ No Gemini key handled gracefully\n")
        finally:
            db_module.async_session_factory = original_factory


async def test_evaluate_empty_data():
    """Test evaluate_sessions khi DB trống → error 'no data'."""
    print("=" * 60)
    print("  TEST: evaluate_sessions (empty data)")
    print("=" * 60)

    from admin_ui.evaluator import evaluate_sessions
    import app.core.database as db_module

    # DB trống riêng
    empty_engine = create_async_engine("sqlite+aiosqlite:///./data/test_eval_empty2.db")
    empty_factory = async_sessionmaker(bind=empty_engine, class_=AsyncSession, expire_on_commit=False)

    async with empty_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    original_factory = db_module.async_session_factory
    db_module.async_session_factory = empty_factory

    try:
        result = await evaluate_sessions(limit=10)

        print(f"  Error: {result['error']}")
        assert result["error"] is not None
        assert "No Q&A" in result["error"]
        assert result["total_questions"] == 0

        print("  ✅ Empty data handled\n")
    finally:
        db_module.async_session_factory = original_factory
        async with empty_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await empty_engine.dispose()
        Path("./data/test_eval_empty2.db").unlink(missing_ok=True)


async def test_citations_edge_cases():
    """Test parsing citations edge cases (malformed JSON, empty, etc)."""
    print("=" * 60)
    print("  TEST: Citations edge cases")
    print("=" * 60)

    from admin_ui.evaluator import _fetch_qa_pairs
    import app.core.database as db_module

    # DB riêng với edge case data
    edge_engine = create_async_engine("sqlite+aiosqlite:///./data/test_eval_edge.db")
    edge_factory = async_sessionmaker(bind=edge_engine, class_=AsyncSession, expire_on_commit=False)

    async with edge_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with edge_factory() as db:
        s = ChatSession(title="Edge Cases")
        db.add(s)
        await db.flush()

        # Pair 1: citations = invalid JSON
        db.add(ChatMessage(session_id=s.id, role="user", content="Q1"))
        await db.flush()
        db.add(ChatMessage(
            session_id=s.id, role="assistant", content="A1",
            citations="NOT VALID JSON {{{}}}",
        ))
        await db.flush()

        # Pair 2: citations = empty list JSON
        db.add(ChatMessage(session_id=s.id, role="user", content="Q2"))
        await db.flush()
        db.add(ChatMessage(
            session_id=s.id, role="assistant", content="A2",
            citations="[]",
        ))
        await db.flush()

        # Pair 3: citations with missing chunk_text
        db.add(ChatMessage(session_id=s.id, role="user", content="Q3"))
        await db.flush()
        db.add(ChatMessage(
            session_id=s.id, role="assistant", content="A3",
            citations=json.dumps([{"document_id": "d1", "filename": "f.pdf"}]),
        ))
        await db.flush()

        await db.commit()

    original_factory = db_module.async_session_factory
    db_module.async_session_factory = edge_factory

    try:
        qa_pairs = await _fetch_qa_pairs()

        assert len(qa_pairs) == 3

        # Invalid JSON → fallback
        assert qa_pairs[0]["contexts"] == ["No context retrieved."]
        print(f"  Invalid JSON:    contexts={qa_pairs[0]['contexts']}")

        # Empty list → fallback
        assert qa_pairs[1]["contexts"] == ["No context retrieved."]
        print(f"  Empty list:      contexts={qa_pairs[1]['contexts']}")

        # Missing chunk_text → fallback
        assert qa_pairs[2]["contexts"] == ["No context retrieved."]
        print(f"  No chunk_text:   contexts={qa_pairs[2]['contexts']}")

        print("  ✅ Edge cases handled\n")
    finally:
        db_module.async_session_factory = original_factory
        async with edge_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await edge_engine.dispose()
        Path("./data/test_eval_edge.db").unlink(missing_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    Path("./data").mkdir(parents=True, exist_ok=True)
    test_db_path = Path("./data/test_eval.db")
    test_db_path.unlink(missing_ok=True)

    print("\n" + "━" * 60)
    print("  🧪 Evaluator Tests — Vòng 6")
    print("━" * 60 + "\n")

    # Setup: create tables + seed data
    print("=" * 60)
    print("  SETUP: Create tables + seed data")
    print("=" * 60)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_ids = await seed_test_data()
    print()

    # Run tests
    await test_fetch_qa_pairs_all()
    await test_fetch_qa_pairs_by_session(list(session_ids))
    await test_fetch_qa_empty_db()
    await test_build_ragas_dataset()
    await test_quick_check()
    await test_evaluate_no_gemini_key()
    await test_evaluate_empty_data()
    await test_citations_edge_cases()

    # Cleanup
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()
    test_db_path.unlink(missing_ok=True)

    print("🎉 All 8 evaluator tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
