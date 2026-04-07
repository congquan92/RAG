"""
API Dependencies — FastAPI Dependency Injection functions.

Cung cấp các hàm Depends() cho route handlers:
  - get_db          → AsyncSession (auto commit/rollback)
  - get_embeddings  → LangChain Embeddings từ app.state
  - get_reranker    → FlashRank Ranker từ app.state (nullable)
  - get_llm         → LangChain Chat model (tạo mới mỗi request qua factory)
  - get_settings    → Settings instance từ app.state

Routers chỉ cần khai báo Depends(dep_func), KHÔNG construct model trực tiếp.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.settings import Settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Database Session
# ═════════════════════════════════════════════════════════════════════════════

async def get_db(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[AsyncSession, None]:
    """
    Proxy dependency — forward từ database.get_db_session.
    Giữ lại để tập trung mọi dependency ở 1 file duy nhất.
    """
    yield session


# ═════════════════════════════════════════════════════════════════════════════
# App State Dependencies (loaded once at startup via lifespan)
# ═════════════════════════════════════════════════════════════════════════════

def get_settings(request: Request) -> Settings:
    """Lấy Settings instance đã lưu trong app.state."""
    return request.app.state.settings


def get_embeddings(request: Request) -> Any:
    """
    Lấy Embedding model từ app.state (loaded 1 lần trong lifespan).
    Raise 503 nếu model chưa sẵn sàng.
    """
    embeddings = getattr(request.app.state, "embeddings", None)
    if embeddings is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not available. Check server startup logs.",
        )
    return embeddings


def get_reranker(request: Request) -> Any | None:
    """
    Lấy FlashRank Reranker từ app.state (nullable — disabled nếu không cài).
    Trả về None nếu reranker không được load.
    """
    return getattr(request.app.state, "reranker", None)


def get_llm(request: Request) -> Any:
    """
    Tạo LLM instance qua factory (stateless, tạo mới mỗi request).
    Factory đọc settings để quyết định provider (ollama/gemini).
    """
    from app.core.llm_factory import get_llm as create_llm

    settings: Settings = request.app.state.settings
    try:
        return create_llm(settings)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"LLM initialization failed: {exc}",
        )
    except Exception as exc:
        logger.error("Unexpected error creating LLM: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable. Check server configuration.",
        )
