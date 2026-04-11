"""
Chat Router — API endpoints cho luồng hỏi-đáp RAG.

Endpoints:
  POST   /chat/sessions          → Tạo phiên hội thoại mới
  GET    /chat/sessions          → Liệt kê sessions (pagination)
  GET    /chat/sessions/{id}     → Chi tiết session + messages
    DELETE /chat/sessions/{id}/messages → Xóa toàn bộ messages của session
  DELETE /chat/sessions/{id}     → Xóa session (cascade messages)
  POST   /chat                   → Gửi câu hỏi → RAG pipeline → trả answer
  POST   /chat/stream            → Gửi câu hỏi → SSE streaming response

Route handlers giữ mỏng: parse request → gọi service → trả response.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_embeddings, get_reranker, get_settings
from app.core.settings import Settings
from app.schemas.chat_schema import (
    ChatAnswerResponse,
    ChatMessageResponse,
    ChatQueryRequest,
    ChatSessionCreate,
    ChatSessionDetailResponse,
    ChatSessionResponse,
    ChatSessionUpdate,
)
from app.services import chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def _resolve_runtime_llm(
    body: ChatQueryRequest,
    settings: Settings,
) -> tuple[Any, bool]:
    """
    Resolve runtime chat mode + LLM instance.

    Returns:
        (llm, enable_graph_search)
    """
    from app.core.llm_factory import get_llm, get_llm_with_overrides

    if body.rag_mode == "graphrag_gemini":
        runtime_key = (body.gemini_api_key or settings.gemini_api_key or "").strip()
        if not runtime_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Gemini API key is required when rag_mode='graphrag_gemini' "
                    "(pass from UI or configure GEMINI_API_KEY in server/.env)."
                ),
            )

        runtime_model = (body.gemini_model or settings.gemini_model).strip()
        try:
            llm = get_llm_with_overrides(
                settings=settings,
                provider_override="gemini",
                model_override=runtime_model,
                gemini_api_key_override=runtime_key,
            )
            return llm, True
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Runtime Gemini initialization failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Gemini runtime unavailable. Check key/model and server logs.",
            )

    try:
        return get_llm(settings), False
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM initialization failed: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error creating default LLM: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Check server configuration.",
        )


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
    created = await chat_service.get_session_payload(db, session.id)
    if created is None:
        raise HTTPException(status_code=500, detail="Failed to create session payload.")
    return ChatSessionResponse(**created)


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
        description=session.description,
        system_prompt=session.system_prompt,
        messages=msg_responses,
    )


@router.patch(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    summary="Cập nhật metadata phiên hội thoại",
)
async def update_session(
    session_id: str,
    body: ChatSessionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Cập nhật metadata workspace/session."""
    updates = body.model_dump(exclude_unset=True)
    session = await chat_service.update_session(db, session_id, updates)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    payload = await chat_service.get_session_payload(db, session.id)
    if payload is None:
        raise HTTPException(status_code=500, detail="Failed to build session payload.")
    return ChatSessionResponse(**payload)


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


@router.delete(
    "/sessions/{session_id}/messages",
    status_code=204,
    summary="Xóa lịch sử chat trong phiên",
)
async def clear_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Xóa tất cả messages của session nhưng giữ nguyên session."""
    result = await chat_service.clear_session_messages(db, session_id)
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
    request: Request,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
    reranker: Any = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
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
        settings = request.app.state.settings if hasattr(request.app.state, "settings") else settings
        llm, enable_graph_search = _resolve_runtime_llm(body, settings)

        result = await chat_service.ask_question(
            db=db,
            session_id=body.session_id,
            query=body.query,
            embeddings=embeddings,
            reranker=reranker,
            llm=llm,
            enable_graph_search=enable_graph_search,
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
    request: Request,
    db: AsyncSession = Depends(get_db),
    embeddings: Any = Depends(get_embeddings),
    reranker: Any = Depends(get_reranker),
    settings: Settings = Depends(get_settings),
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
    settings = request.app.state.settings if hasattr(request.app.state, "settings") else settings
    llm, enable_graph_search = _resolve_runtime_llm(body, settings)

    async def event_generator():
        try:
            async for event_str in chat_service.ask_question_stream(
                db=db,
                session_id=body.session_id,
                query=body.query,
                embeddings=embeddings,
                reranker=reranker,
                llm=llm,
                enable_graph_search=enable_graph_search,
                is_client_disconnected=request.is_disconnected,
            ):
                if await request.is_disconnected():
                    logger.info("Client disconnected; stop streaming session=%s", body.session_id)
                    break
                yield f"data: {event_str}\n\n"
        except asyncio.CancelledError:
            logger.info("SSE task cancelled by disconnect: session=%s", body.session_id)
            return
        except Exception as exc:
            logger.error("Stream error: %s", exc, exc_info=True)
            if not await request.is_disconnected():
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
