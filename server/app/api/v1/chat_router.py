"""
Chat Router — API endpoints cho luồng hỏi-đáp RAG.

Endpoints:
  POST   /chat/sessions          → Tạo phiên hội thoại mới
  GET    /chat/sessions          → Liệt kê sessions (pagination)
  GET    /chat/sessions/{id}     → Chi tiết session + messages
  DELETE /chat/sessions/{id}     → Xóa session (cascade messages)
  POST   /chat                   → Gửi câu hỏi → RAG pipeline → trả answer
  POST   /chat/stream            → Gửi câu hỏi → SSE streaming response

Route handlers giữ mỏng: parse request → gọi service → trả response.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_embeddings, get_llm, get_reranker
from app.schemas.chat_schema import (
    ChatAnswerResponse,
    ChatMessageResponse,
    ChatQueryRequest,
    ChatSessionCreate,
    ChatSessionDetailResponse,
    ChatSessionResponse,
)
from app.services import chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


# ═════════════════════════════════════════════════════════════════════════════
# Session CRUD
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/sessions",
    response_model=ChatSessionResponse,
    status_code=201,
    summary="Tạo phiên hội thoại mới",
)
async def create_session(
    body: ChatSessionCreate,
    db: AsyncSession = Depends(get_db),
):
    """Tạo session rỗng, trả về metadata (id, title, timestamps)."""
    session = await chat_service.create_session(db, title=body.title)
    return ChatSessionResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.get(
    "/sessions",
    response_model=list[ChatSessionResponse],
    summary="Liệt kê sessions",
)
async def list_sessions(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Danh sách sessions với message_count + pagination."""
    sessions_data, _total = await chat_service.list_sessions(db, skip=skip, limit=limit)
    return [ChatSessionResponse(**s) for s in sessions_data]


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionDetailResponse,
    summary="Chi tiết session + lịch sử messages",
)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Lấy session kèm toàn bộ messages (ordered by created_at)."""
    session = await chat_service.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    messages = await chat_service.get_session_messages(db, session_id)

    # Chuyển đổi messages sang response format
    msg_responses = []
    for msg in (messages or []):
        citations = None
        if msg.citations:
            try:
                citations = json.loads(msg.citations)
            except (json.JSONDecodeError, TypeError):
                citations = None

        msg_responses.append(ChatMessageResponse(
            id=msg.id,
            session_id=msg.session_id,
            role=msg.role,
            content=msg.content,
            citations=citations,
            created_at=msg.created_at,
        ))

    return ChatSessionDetailResponse(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=msg_responses,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=204,
    summary="Xóa phiên hội thoại",
)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Xóa session + cascade xóa tất cả messages."""
    result = await chat_service.delete_session(db, session_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Core: Ask Question (Sync)
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "",
    response_model=ChatAnswerResponse,
    summary="Gửi câu hỏi → RAG pipeline → trả answer",
)
async def ask_question(
    body: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
    reranker: Any = Depends(get_reranker),
    llm: Any = Depends(get_llm),
):
    """
    Full RAG pipeline: Query → Tri-Search → Rerank → Generate → Save.

    Nếu body.stream=True → redirect sang /chat/stream (client nên gọi thẳng).
    """
    if body.stream:
        raise HTTPException(
            status_code=400,
            detail="Use POST /api/v1/chat/stream for streaming responses.",
        )

    try:
        result = await chat_service.ask_question(
            db=db,
            session_id=body.session_id,
            query=body.query,
            embeddings=embeddings,
            reranker=reranker,
            llm=llm,
        )
        return ChatAnswerResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during RAG pipeline.")


# ═════════════════════════════════════════════════════════════════════════════
# Core: Ask Question (SSE Streaming)
# ═════════════════════════════════════════════════════════════════════════════

@router.post(
    "/stream",
    summary="Gửi câu hỏi → SSE streaming response",
)
async def ask_question_stream(
    body: ChatQueryRequest,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
    reranker: Any = Depends(get_reranker),
    llm: Any = Depends(get_llm),
):
    """
    Streaming RAG pipeline qua Server-Sent Events.

    Events:
      - {type: "message_ids", data: {user_id, assistant_id}}
      - {type: "token", data: <text>}
      - {type: "citations", data: [...]}
      - {type: "done"}
      - {type: "error", data: <message>}
    """
    async def event_generator():
        try:
            async for event_str in chat_service.ask_question_stream(
                db=db,
                session_id=body.session_id,
                query=body.query,
                embeddings=embeddings,
                reranker=reranker,
                llm=llm,
            ):
                yield f"data: {event_str}\n\n"
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            error_event = json.dumps({"type": "error", "data": str(exc)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx proxy buffering off
        },
    )
