"""
FastAPI Entry Point — Application bootstrap with Lifespan management.

Startup sequence:
  1. Start Phoenix tracer (if enabled)
  2. Initialize database (create tables)
  3. Load Embedding model into app.state (based on EMBEDDING_DEVICE)
  4. Load FlashRank reranker into app.state (based on RERANKER_DEVICE)
  5. Yield → application serves requests
  6. Shutdown: graceful cleanup of DB, Phoenix, models

All heavy models are loaded ONCE at startup and reused via app.state.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.settings import settings

# ── Logging setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("rag.server")


# ═════════════════════════════════════════════════════════════════════════════
# Lifespan — startup/shutdown resource management
# ═════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Quản lý vòng đời ứng dụng:
    - Startup: khởi tạo Phoenix, load AI models vào app.state
    - Shutdown: giải phóng tài nguyên, đóng kết nối
    """
    logger.info("═" * 60)
    logger.info("  RAG Server — Starting up...")
    logger.info("═" * 60)

    # ── 1. Phoenix Tracing (Observability) ───────────────────────────────
    phoenix_session = None
    if settings.phoenix_enabled:
        try:
            import phoenix as px

            phoenix_session = px.launch_app(port=settings.phoenix_port)
            logger.info(
                "Phoenix tracing started on port %d", settings.phoenix_port
            )

            # Auto-instrument LangChain calls cho tracing
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument()
            logger.info("LangChain auto-instrumentation enabled")
        except ImportError:
            logger.warning(
                "Phoenix or instrumentation packages not installed. "
                "Tracing disabled. Install: arize-phoenix, "
                "openinference-instrumentation-langchain"
            )
        except Exception as exc:
            logger.warning("Failed to start Phoenix tracing: %s", exc)

    # ── 2. Database Initialization ────────────────────────────────────────
    from app.models import Base  # noqa: F401 — đảm bảo tất cả models registered
    from app.core.database import init_db, dispose_db

    await init_db()
    logger.info("Database initialized successfully")

    # ── 3. Load Embedding Model ──────────────────────────────────────────
    try:
        from app.core.llm_factory import get_embeddings

        embeddings = get_embeddings(settings)
        app.state.embeddings = embeddings
        logger.info(
            "Embedding model loaded: provider=%s, device=%s",
            settings.embedding_provider,
            settings.embedding_device,
        )
    except Exception as exc:
        logger.error("Failed to load embedding model: %s", exc)
        app.state.embeddings = None

    # ── 4. Load Reranker (FlashRank) ─────────────────────────────────────
    try:
        from flashrank import Ranker

        reranker = Ranker(model_name=settings.reranker_model)
        app.state.reranker = reranker
        logger.info(
            "FlashRank reranker loaded: model=%s, device=%s",
            settings.reranker_model,
            settings.reranker_device,
        )
    except ImportError:
        logger.warning("flashrank not installed. Reranker disabled.")
        app.state.reranker = None
    except Exception as exc:
        logger.warning("Failed to load reranker: %s", exc)
        app.state.reranker = None

    # ── 5. Store settings reference ──────────────────────────────────────
    app.state.settings = settings

    logger.info("═" * 60)
    logger.info("  RAG Server — Ready to serve requests!")
    logger.info("  LLM Provider : %s", settings.llm_provider)
    logger.info("  Embed Provider: %s", settings.embedding_provider)
    logger.info("  CORS Origins  : %s", settings.cors_origin_list)
    logger.info("═" * 60)

    # ── Yield — App is running ───────────────────────────────────────────
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("RAG Server — Shutting down...")

    # Đóng database connection pool
    await dispose_db()

    # Đóng Phoenix session
    if phoenix_session is not None:
        try:
            phoenix_session.close()
            logger.info("Phoenix tracing stopped")
        except Exception as exc:
            logger.warning("Error closing Phoenix: %s", exc)

    # Cleanup embedding model (giải phóng GPU memory nếu có)
    if hasattr(app.state, "embeddings") and app.state.embeddings is not None:
        del app.state.embeddings
        logger.info("Embedding model released")

    if hasattr(app.state, "reranker") and app.state.reranker is not None:
        del app.state.reranker
        logger.info("Reranker model released")

    logger.info("RAG Server — Shutdown complete")


# ═════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Advanced RAG Server",
    description="Enterprise-grade Retrieval-Augmented Generation API "
    "with Hot-Swap LLM providers and Dynamic Hardware Optimization.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ─────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Endpoint kiểm tra trạng thái server và các component đã load."""
    return {
        "status": "healthy",
        "llm_provider": settings.llm_provider,
        "embedding_provider": settings.embedding_provider,
        "embedding_loaded": getattr(app.state, "embeddings", None) is not None,
        "reranker_loaded": getattr(app.state, "reranker", None) is not None,
        "phoenix_enabled": settings.phoenix_enabled,
    }


# ── API v1 Routers ──────────────────────────────────────────────────────
from app.api.v1 import router as api_v1_router

app.include_router(api_v1_router, prefix="/api/v1")


if __name__ == "__main__":
    import subprocess
    import sys
    import uvicorn

    check_proc = subprocess.run(
        [sys.executable, "scripts/check_env.py", "--quiet"],
        check=False,
    )
    if check_proc.returncode != 0:
        raise SystemExit(
            "Environment validation failed. Fix .env before starting the server."
        )

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
