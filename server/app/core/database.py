"""
Async Database Setup — SQLAlchemy async engine & session factory.

Sử dụng aiosqlite làm backend cho SQLite async. Engine và session factory
được tạo 1 lần, dùng lại trong toàn bộ app thông qua dependency injection.

Workflow:
  - Startup (lifespan): gọi init_db() để tạo tables
  - Request:           dùng get_db_session() làm FastAPI dependency
  - Shutdown:          gọi dispose_db() để đóng connection pool
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.settings import settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Declarative Base — tất cả models kế thừa từ đây
# ═════════════════════════════════════════════════════════════════════════════

class Base(DeclarativeBase):
    """Base class cho tất cả SQLAlchemy ORM models."""


# ═════════════════════════════════════════════════════════════════════════════
# Engine & Session Factory
# ═════════════════════════════════════════════════════════════════════════════

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    echo=False,  # True để debug SQL queries
    pool_pre_ping=True,
)

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ═════════════════════════════════════════════════════════════════════════════
# Lifecycle helpers (gọi từ main.py lifespan)
# ═════════════════════════════════════════════════════════════════════════════

async def init_db() -> None:
    """
    Tạo tất cả tables nếu chưa tồn tại.
    Gọi 1 lần trong lifespan startup sau khi import models.
    """
    # Tự tạo thư mục data/ nếu chưa có (tránh lỗi "unable to open database file")
    from pathlib import Path

    db_url = settings.database_url
    if "sqlite" in db_url:
        # Parse path từ URL: "sqlite+aiosqlite:///./data/history.db" → "./data/history.db"
        db_path = db_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized: %s", settings.database_url)


async def dispose_db() -> None:
    """Đóng connection pool. Gọi trong lifespan shutdown."""
    await engine.dispose()
    logger.info("Database engine disposed")


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI Dependency — inject session vào route handlers
# ═════════════════════════════════════════════════════════════════════════════

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency cho FastAPI routes. Tự động commit/rollback/close.

    Usage trong router:
        @router.post("/chat")
        async def chat(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
