"""
Chat Service — Nhận câu hỏi → Retrieve → Generate → Lưu lịch sử.

Workflow chính (ask_question):
  1. Lưu user message vào DB
  2. Gọi retriever.retrieve() → Tri-Search + Rerank
  3. Gọi generator.generate() → LLM sinh câu trả lời
  4. Lưu assistant message + citations vào DB
  5. Trả kết quả cho controller

Quản lý sessions:
  - create_session / get_session / list_sessions / delete_session
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession
from app.rag.generator import generate, generate_stream
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Session Management
# ═════════════════════════════════════════════════════════════════════════════

async def create_session(
    db: AsyncSession,
    title: str = "New Conversation",
) -> ChatSession:
    """Tạo phiên hội thoại mới."""
    session = ChatSession(title=title)
    db.add(session)
    await db.flush()
    logger.info("Created chat session: id=%s, title=%s", session.id, title)
    return session


async def get_session(
    db: AsyncSession,
    session_id: str,
) -> Optional[ChatSession]:
    """Lấy phiên chat theo ID (kèm messages qua selectin)."""
    return await db.get(ChatSession, session_id)


async def list_sessions(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 50,
) -> tuple[list[dict], int]:
    """
    Liệt kê sessions với message count + pagination.

    Returns:
        (sessions_data, total_count)
    """
    # Count total
    count_stmt = select(func.count(ChatSession.id))
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Fetch sessions
    stmt = (
        select(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    sessions = list(result.scalars().all())

    # Build response với message_count
    sessions_data: list[dict] = []
    for s in sessions:
        msg_count_stmt = select(func.count(ChatMessage.id)).where(
            ChatMessage.session_id == s.id
        )
        msg_result = await db.execute(msg_count_stmt)
        msg_count = msg_result.scalar() or 0

        sessions_data.append({
            "id": s.id,
            "title": s.title,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "message_count": msg_count,
        })

    return sessions_data, total


async def delete_session(
    db: AsyncSession,
    session_id: str,
) -> Optional[ChatSession]:
    """Xóa phiên chat (cascade xóa messages)."""
    session = await db.get(ChatSession, session_id)
    if not session:
        return None

    await db.delete(session)
    logger.info("Deleted chat session: id=%s", session_id)
    return session


# ═════════════════════════════════════════════════════════════════════════════
# Core: Ask Question (Sync)
# ═════════════════════════════════════════════════════════════════════════════

async def ask_question(
    db: AsyncSession,
    session_id: str,
    query: str,
    embeddings: Any,
    reranker: Any,
    llm: Any,
) -> dict[str, Any]:
    """
    Full RAG pipeline: Query → Retrieve → Generate → Save History.

    Args:
        db: Async database session
        session_id: ID phiên chat
        query: Câu hỏi user
        embeddings: LangChain Embeddings (từ app.state)
        reranker: FlashRank Ranker (từ app.state, nullable)
        llm: LangChain Chat model (từ factory)

    Returns:
        dict compatible với ChatAnswerResponse schema

    Raises:
        ValueError: nếu session_id không tồn tại
    """
    # Validate session
    session = await db.get(ChatSession, session_id)
    if not session:
        raise ValueError(f"Chat session not found: {session_id}")

    # ── 1. Lưu user message ─────────────────────────────────────────
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=query,
    )
    db.add(user_message)
    await db.flush()

    logger.info("User message saved: session=%s, msg_id=%s", session_id, user_message.id)

    # ── 2. Retrieve — Tri-Search + Rerank ────────────────────────────
    retrieval_result = retrieve(
        query=query,
        embeddings=embeddings,
        reranker=reranker,
    )

    logger.info(
        "Retrieved: semantic=%d, keyword=%d, graph=%d → final=%d chunks",
        len(retrieval_result.raw_semantic),
        len(retrieval_result.raw_keyword),
        len(retrieval_result.raw_graph),
        len(retrieval_result.chunks),
    )

    # ── 3. Generate — LLM sinh câu trả lời ──────────────────────────
    gen_result = await generate(
        query=query,
        chunks=retrieval_result.chunks,
        llm=llm,
    )

    # ── 4. Lưu assistant message + citations ─────────────────────────
    citations_json = json.dumps(gen_result["citations"], ensure_ascii=False)

    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=gen_result["answer"],
        citations=citations_json,
    )
    db.add(assistant_message)
    await db.flush()

    # Auto-update session title từ câu hỏi đầu tiên
    if session.title == "New Conversation":
        session.title = query[:100]  # Cắt 100 ký tự đầu làm title

    logger.info(
        "Answer generated: session=%s, assistant_msg=%s, citations=%d",
        session_id, assistant_message.id, len(gen_result["citations"]),
    )

    # ── 5. Trả kết quả ──────────────────────────────────────────────
    return {
        "session_id": session_id,
        "answer": gen_result["answer"],
        "citations": gen_result["citations"],
        "user_message_id": user_message.id,
        "assistant_message_id": assistant_message.id,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Core: Ask Question (Streaming)
# ═════════════════════════════════════════════════════════════════════════════

async def ask_question_stream(
    db: AsyncSession,
    session_id: str,
    query: str,
    embeddings: Any,
    reranker: Any,
    llm: Any,
) -> AsyncGenerator[str, None]:
    """
    Streaming RAG pipeline: Query → Retrieve → Stream Generate → Save History.

    Yields:
        JSON strings cho SSE events:
          - {type: "message_ids", data: {user_id, assistant_id}}
          - {type: "token", data: <text>}
          - {type: "citations", data: [...]}
          - {type: "done"}
          - {type: "error", data: <message>}
    """
    # Validate session
    session = await db.get(ChatSession, session_id)
    if not session:
        yield json.dumps({"type": "error", "data": f"Session not found: {session_id}"})
        return

    # ── 1. Lưu user message ──────────────────────────────────────────
    user_message = ChatMessage(
        session_id=session_id,
        role="user",
        content=query,
    )
    db.add(user_message)
    await db.flush()

    # Tạo assistant message placeholder (sẽ update content sau)
    assistant_message = ChatMessage(
        session_id=session_id,
        role="assistant",
        content="",  # Sẽ update sau khi stream xong
    )
    db.add(assistant_message)
    await db.flush()

    # Gửi message IDs cho frontend
    yield json.dumps({
        "type": "message_ids",
        "data": {
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
        },
    })

    # ── 2. Retrieve ──────────────────────────────────────────────────
    retrieval_result = retrieve(
        query=query,
        embeddings=embeddings,
        reranker=reranker,
    )

    # ── 3. Stream Generate ───────────────────────────────────────────
    full_answer = ""
    citations_data: list[dict] = []

    async for event_str in generate_stream(
        query=query,
        chunks=retrieval_result.chunks,
        llm=llm,
    ):
        event = json.loads(event_str)

        if event["type"] == "token":
            full_answer += event["data"]
            yield event_str  # Forward token event

        elif event["type"] == "citations":
            citations_data = event["data"]
            yield event_str  # Forward citations event

        elif event["type"] == "done":
            # ── 4. Update assistant message with full content ─────────
            assistant_message.content = full_answer
            assistant_message.citations = json.dumps(citations_data, ensure_ascii=False)

            # Auto-update session title
            if session.title == "New Conversation":
                session.title = query[:100]

            await db.flush()

            logger.info(
                "Stream saved: session=%s, answer=%d chars, citations=%d",
                session_id, len(full_answer), len(citations_data),
            )
            yield event_str  # Forward done event

        elif event["type"] == "error":
            assistant_message.content = f"Error: {event.get('data', 'Unknown')}"
            await db.flush()
            yield event_str


# ═════════════════════════════════════════════════════════════════════════════
# Message History
# ═════════════════════════════════════════════════════════════════════════════

async def get_session_messages(
    db: AsyncSession,
    session_id: str,
) -> Optional[list[ChatMessage]]:
    """
    Lấy toàn bộ messages trong session (ordered by created_at).
    Returns None nếu session không tồn tại.
    """
    session = await db.get(ChatSession, session_id)
    if not session:
        return None

    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())
