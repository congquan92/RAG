"""
API v1 Router Aggregation — Gom tất cả sub-routers của version 1.

Import file này trong main.py để mount toàn bộ endpoints v1:
    from app.api.v1 import router as api_v1_router
    app.include_router(api_v1_router, prefix="/api/v1")
"""

from fastapi import APIRouter

from app.api.v1.chat_router import router as chat_router
from app.api.v1.document_router import router as document_router

router = APIRouter()

router.include_router(chat_router)
router.include_router(document_router)
