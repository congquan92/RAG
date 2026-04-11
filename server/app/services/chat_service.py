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

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat import ChatMessage, ChatSession
from app.rag.generator import generate, generate_stream
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


async def _get_message_count(db: AsyncSession, session_id: str) -> int:
    stmt = select(func.count(ChatMessage.id)).where(ChatMessage.session_id == session_id)
    result = await db.execute(stmt)
    return result.scalar() or 0


async def _build_session_payload(db: AsyncSession, session: ChatSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "description": session.description,
        "system_prompt": session.system_prompt,
        "message_count": await _get_message_count(db, session.id),
    }


async def get_session_payload(
    db: AsyncSession,
    session_id: str,
) -> Optional[dict[str, Any]]:
    session = await db.get(ChatSession, session_id)
    if not session:
        return None
    return await _build_session_payload(db, session)


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

    # Build response với message_count + settings
    sessions_data: list[dict] = []
    for s in sessions:
        sessions_data.append(await _build_session_payload(db, s))

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


async def clear_session_messages(
    db: AsyncSession,
    session_id: str,
) -> bool:
    """Xóa toàn bộ messages trong một phiên chat, giữ lại session."""
    session = await db.get(ChatSession, session_id)
    if not session:
        return False

    await db.execute(
        delete(ChatMessage).where(ChatMessage.session_id == session_id)
    )
    session.updated_at = datetime.now(timezone.utc)
    logger.info("Cleared chat history for session: id=%s", session_id)
    return True


async def update_session(
    db: AsyncSession,
    session_id: str,
    updates: dict[str, Any],
) -> Optional[ChatSession]:
    """Cập nhật metadata session/workspace theo PATCH payload."""
    session = await db.get(ChatSession, session_id)
    if not session:
        return None

    if "title" in updates and updates["title"] is not None:
        session.title = str(updates["title"]).strip() or session.title

    if "description" in updates:
        session.description = updates["description"]

    if "system_prompt" in updates:
        session.system_prompt = updates["system_prompt"]

    session.updated_at = datetime.now(timezone.utc)
    await db.flush()
    logger.info("Updated session metadata: id=%s", session_id)
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
    enable_graph_search: bool = True,
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
        enable_graph=enable_graph_search,
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
    enable_graph_search: bool = True,
    is_client_disconnected: Callable[[], Awaitable[bool]] | None = None,
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
    if is_client_disconnected is not None and await is_client_disconnected():
        logger.info("Stream request cancelled before session validation")
        return

    # Validate session
    session = await db.get(ChatSession, session_id)
    if not session:
        yield json.dumps({"type": "error", "data": f"Session not found: {session_id}"})
        return

    if is_client_disconnected is not None and await is_client_disconnected():
        logger.info("Stream request cancelled before message persistence")
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

    async def _delete_placeholder_if_exists() -> None:
        """Xóa placeholder assistant khi stream bị hủy quá sớm."""
        if not assistant_message.id:
            return
        existing = await db.get(ChatMessage, assistant_message.id)
        if existing is not None:
            await db.delete(existing)
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
        enable_graph=enable_graph_search,
    )

    if is_client_disconnected is not None and await is_client_disconnected():
        await _delete_placeholder_if_exists()
        logger.info("Stream request cancelled before token generation")
        return

    # ── 3. Stream Generate ───────────────────────────────────────────
    full_answer = ""
    citations_data: list[dict] = []

    try:
        async for event_str in generate_stream(
            query=query,
            chunks=retrieval_result.chunks,
            llm=llm,
            should_stop=is_client_disconnected,
        ):
            event = json.loads(event_str)

            if event["type"] == "token":
                full_answer += event["data"]
                yield event_str
                continue

            if event["type"] == "citations":
                citations_data = event["data"]
                yield event_str
                continue

            if event["type"] == "done":
                assistant_message.content = full_answer
                assistant_message.citations = json.dumps(citations_data, ensure_ascii=False)

                if session.title == "New Conversation":
                    session.title = query[:100]

                await db.flush()
                logger.info(
                    "Stream saved: session=%s, answer=%d chars, citations=%d",
                    session_id, len(full_answer), len(citations_data),
                )
                yield event_str
                return

            if event["type"] == "error":
                assistant_message.content = f"Error: {event.get('data', 'Unknown')}"
                await db.flush()
                yield event_str
                return
    except asyncio.CancelledError:
        if full_answer.strip():
            assistant_message.content = full_answer
            assistant_message.citations = json.dumps(citations_data, ensure_ascii=False)
            await db.flush()
            logger.info(
                "Stream cancelled with partial answer persisted: session=%s, chars=%d",
                session_id,
                len(full_answer),
            )
        else:
            await _delete_placeholder_if_exists()
            logger.info("Stream cancelled before first token: placeholder removed session=%s", session_id)
        return


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
