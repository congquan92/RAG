"""
Test Script: API Controllers (Vòng 5)
Kiểm tra toàn bộ HTTP endpoints qua ASGI TestClient.

Sử dụng httpx.AsyncClient + FastAPI test transport.
KHÔNG cần chạy uvicorn, KHÔNG cần AI models thật — mock toàn bộ app.state.
Dùng test DB riêng (in-memory SQLite) tránh ảnh hưởng data thật.

Usage:
    cd server
    source .venv/bin/activate
    python scripts/test_api.py
"""

import asyncio
import inspect
import json
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Import models + DB components ngay từ đầu ────────────────────────────

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Import Base và models để register tables vào metadata
from app.core.database import Base
from app.models.chat import ChatSession, ChatMessage  # noqa: F401
from app.models.document import Document, IngestionTask  # noqa: F401
from app.core.settings import settings

# ── Test DB riêng (in-memory hoặc file tạm) ─────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///./data/test_api.db"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
test_session_factory = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_test_db():
    """Dependency override: dùng test DB thay vì production DB."""
    async with test_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def _create_test_app():
    """Tạo FastAPI app instance với test DB + mocked AI models."""
    from fastapi import FastAPI
    from app.api.v1 import router as api_v1_router
    from app.api.deps import get_db, get_llm
    from app.core.database import get_db_session

    def get_test_llm():
        """Dependency override: trả về mock LLM, không load provider thật."""
        return MagicMock(name="mock_llm")

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        """Lifespan: tạo tables trong test DB, mock AI models."""
        # Tạo tables
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Mock embedding model
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents = MagicMock(
            return_value=[[0.1] * 384]
        )
        mock_embeddings.embed_query = MagicMock(
            return_value=[0.1] * 384
        )
        app.state.embeddings = mock_embeddings
        app.state.reranker = None
        app.state.settings = settings

        yield

        # Cleanup: drop tables + dispose engine
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await test_engine.dispose()

    app = FastAPI(title="Test RAG Server", lifespan=test_lifespan)

    # Override DB dependency → dùng test DB
    app.dependency_overrides[get_db_session] = get_test_db
    app.dependency_overrides[get_db] = get_test_db
    app.dependency_overrides[get_llm] = get_test_llm

    app.include_router(api_v1_router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    return app


# ═════════════════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════════════════

async def test_health_check(client):
    """Test health endpoint."""
    print("=" * 60)
    print("  TEST: GET /health")
    print("=" * 60)

    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    print(f"  Response: {data}")
    print("  ✅ Health check OK\n")


async def test_create_session(client) -> str:
    """Test tạo session mới → trả về session_id."""
    print("=" * 60)
    print("  TEST: POST /api/v1/chat/sessions")
    print("=" * 60)

    resp = await client.post(
        "/api/v1/chat/sessions",
        json={"title": "Test API Session"},
    )
    assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert "id" in data
    assert data["title"] == "Test API Session"
    assert data["message_count"] == 0
    print(f"  Session ID:  {data['id'][:8]}...")
    print(f"  Title:       {data['title']}")
    print(f"  Created:     {data['created_at']}")
    print("  ✅ Create session OK\n")
    return data["id"]


async def test_create_session_default_title(client) -> str:
    """Test tạo session với title mặc định."""
    print("=" * 60)
    print("  TEST: POST /api/v1/chat/sessions (default title)")
    print("=" * 60)

    resp = await client.post("/api/v1/chat/sessions", json={})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "New Conversation"
    print(f"  Title: {data['title']}")
    print("  ✅ Default title OK\n")
    return data["id"]


async def test_list_sessions(client):
    """Test liệt kê sessions."""
    print("=" * 60)
    print("  TEST: GET /api/v1/chat/sessions")
    print("=" * 60)

    resp = await client.get("/api/v1/chat/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1, "Should have at least 1 session"
    print(f"  Sessions found: {len(data)}")
    for s in data[:3]:
        print(f"    - {s['title']} (id={s['id'][:8]}..., msgs={s['message_count']})")
    print("  ✅ List sessions OK\n")


async def test_list_sessions_pagination(client):
    """Test pagination query params."""
    print("=" * 60)
    print("  TEST: GET /api/v1/chat/sessions?skip=0&limit=1")
    print("=" * 60)

    resp = await client.get("/api/v1/chat/sessions", params={"skip": 0, "limit": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) <= 1
    print(f"  Sessions (limit=1): {len(data)}")
    print("  ✅ Pagination OK\n")


async def test_get_session_detail(client, session_id: str):
    """Test lấy chi tiết 1 session + messages."""
    print("=" * 60)
    print("  TEST: GET /api/v1/chat/sessions/{id}")
    print("=" * 60)

    resp = await client.get(f"/api/v1/chat/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == session_id
    assert "messages" in data
    print(f"  Session:   {data['title']}")
    print(f"  Messages:  {len(data['messages'])}")
    print("  ✅ Get session detail OK\n")


async def test_get_session_not_found(client):
    """Test 404 khi session không tồn tại."""
    print("=" * 60)
    print("  TEST: GET /api/v1/chat/sessions/{invalid} → 404")
    print("=" * 60)

    resp = await client.get("/api/v1/chat/sessions/nonexistent-id-12345")
    assert resp.status_code == 404
    print(f"  Status: {resp.status_code}")
    print(f"  Detail: {resp.json()['detail']}")
    print("  ✅ 404 handled correctly\n")


async def test_delete_session(client, session_id: str):
    """Test xóa session."""
    print("=" * 60)
    print("  TEST: DELETE /api/v1/chat/sessions/{id}")
    print("=" * 60)

    resp = await client.delete(f"/api/v1/chat/sessions/{session_id}")
    assert resp.status_code == 204
    print(f"  Deleted session: {session_id[:8]}...")

    # Verify đã xóa
    verify = await client.get(f"/api/v1/chat/sessions/{session_id}")
    assert verify.status_code == 404
    print("  Verified: session no longer exists")
    print("  ✅ Delete session OK\n")


async def test_delete_session_not_found(client):
    """Test 404 khi xóa session không tồn tại."""
    print("=" * 60)
    print("  TEST: DELETE /api/v1/chat/sessions/{invalid} → 404")
    print("=" * 60)

    resp = await client.delete("/api/v1/chat/sessions/ghost-session-999")
    assert resp.status_code == 404
    print(f"  Status: {resp.status_code}")
    print("  ✅ 404 on delete handled\n")


async def test_upload_document(client, workspace_id: str) -> dict:
    """Test upload file → nhận task_id."""
    print("=" * 60)
    print("  TEST: POST /api/v1/documents/upload")
    print("=" * 60)

    file_content = b"This is a test document for API upload testing. " * 10

    # Mock background ingestion để test API contract không bị nhiễu bởi worker thực.
    with patch("app.services.document_service.process_ingestion_task", new=AsyncMock(return_value=None)):
        resp = await client.post(
            "/api/v1/documents/upload",
            data={"workspace_id": workspace_id},
            files={"file": ("test_api_upload.txt", file_content, "text/plain")},
        )
    assert resp.status_code == 202, f"Expected 202, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert "document_id" in data
    assert "task_id" in data
    assert data["filename"] == "test_api_upload.txt"
    assert data["file_size"] == len(file_content)

    print(f"  Document ID: {data['document_id'][:8]}...")
    print(f"  Task ID:     {data['task_id'][:8]}...")
    print(f"  Filename:    {data['filename']}")
    print(f"  Size:        {data['file_size']} bytes")
    print(f"  MIME:        {data['mime_type']}")
    print(f"  Message:     {data['message']}")
    print("  ✅ Upload OK\n")
    return data


async def test_upload_empty_file(client):
    """Test upload file rỗng → 400."""
    print("=" * 60)
    print("  TEST: POST /api/v1/documents/upload (empty) → 400")
    print("=" * 60)

    resp = await client.post(
        "/api/v1/documents/upload",
        data={"workspace_id": "workspace-api-tests"},
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert resp.status_code == 400
    print(f"  Status: {resp.status_code}")
    print(f"  Detail: {resp.json()['detail']}")
    print("  ✅ Empty file rejected\n")


async def test_get_ingestion_status(client, task_id: str):
    """Test poll trạng thái ingestion."""
    print("=" * 60)
    print("  TEST: GET /api/v1/documents/status/{task_id}")
    print("=" * 60)

    resp = await client.get(f"/api/v1/documents/status/{task_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["task_id"] == task_id
    assert data["status"] in ("pending", "processing", "completed", "failed")
    print(f"  Task ID:    {data['task_id'][:8]}...")
    print(f"  Status:     {data['status']}")
    print(f"  Chunks:     {data['chunks_processed']}")
    print("  ✅ Status polling OK\n")


async def test_get_ingestion_status_not_found(client):
    """Test 404 khi task không tồn tại."""
    print("=" * 60)
    print("  TEST: GET /api/v1/documents/status/{invalid} → 404")
    print("=" * 60)

    resp = await client.get("/api/v1/documents/status/nonexistent-task-9999")
    assert resp.status_code == 404
    print(f"  Status: {resp.status_code}")
    print("  ✅ 404 on status handled\n")


async def test_list_documents(client, workspace_id: str):
    """Test liệt kê documents."""
    print("=" * 60)
    print("  TEST: GET /api/v1/documents")
    print("=" * 60)

    resp = await client.get(f"/api/v1/documents?workspace_id={workspace_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "documents" in data
    assert "total" in data
    assert data["total"] >= 1
    print(f"  Total: {data['total']}")
    for doc in data["documents"][:3]:
        print(f"    - {doc['filename']} (id={doc['id'][:8]}..., chunks={doc['chunk_count']})")
    print("  ✅ List documents OK\n")


async def test_get_document(client, document_id: str, workspace_id: str):
    """Test lấy chi tiết 1 document."""
    print("=" * 60)
    print("  TEST: GET /api/v1/documents/{id}")
    print("=" * 60)

    resp = await client.get(f"/api/v1/documents/{document_id}?workspace_id={workspace_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == document_id
    print(f"  ID:          {data['id'][:8]}...")
    print(f"  Filename:    {data['filename']}")
    print(f"  Size:        {data['file_size']} bytes")
    print(f"  Chunks:      {data['chunk_count']}")
    print("  ✅ Get document OK\n")


async def test_get_document_not_found(client, workspace_id: str):
    """Test 404 khi document không tồn tại."""
    print("=" * 60)
    print("  TEST: GET /api/v1/documents/{invalid} → 404")
    print("=" * 60)

    resp = await client.get(f"/api/v1/documents/ghost-doc-12345?workspace_id={workspace_id}")
    assert resp.status_code == 404
    print(f"  Status: {resp.status_code}")
    print("  ✅ 404 handled\n")


async def test_delete_document(client, document_id: str, workspace_id: str):
    """Test xóa document + cleanup."""
    print("=" * 60)
    print("  TEST: DELETE /api/v1/documents/{id}")
    print("=" * 60)

    resp = await client.delete(f"/api/v1/documents/{document_id}?workspace_id={workspace_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == document_id
    print(f"  Deleted:  {data['filename']} (id={data['document_id'][:8]}...)")
    print(f"  Message:  {data['message']}")

    # Verify
    verify = await client.get(f"/api/v1/documents/{document_id}?workspace_id={workspace_id}")
    assert verify.status_code == 404
    print("  Verified: document no longer exists")
    print("  ✅ Delete document OK\n")


async def test_chat_ask_stream_redirect(client, session_id: str):
    """Test POST /chat với stream=true → 400 redirect."""
    print("=" * 60)
    print("  TEST: POST /api/v1/chat (stream=true) → 400")
    print("=" * 60)

    resp = await client.post(
        "/api/v1/chat",
        json={
            "session_id": session_id,
            "query": "test stream redirect",
            "stream": True,
        },
    )
    assert resp.status_code == 400
    print(f"  Status: {resp.status_code}")
    print(f"  Detail: {resp.json()['detail']}")
    print("  ✅ Stream redirect handled\n")


async def test_chat_ask_invalid_session(client):
    """Test POST /chat với session_id không tồn tại → 404."""
    print("=" * 60)
    print("  TEST: POST /api/v1/chat (bad session) → 404")
    print("=" * 60)

    # Mock RAG pipeline
    with patch("app.services.chat_service.retrieve") as mock_ret, \
         patch("app.services.chat_service.generate", new_callable=AsyncMock) as mock_gen:
        resp = await client.post(
            "/api/v1/chat",
            json={
                "session_id": "nonexistent-session-id",
                "query": "this should fail",
                "stream": False,
            },
        )
    assert resp.status_code == 404
    print(f"  Status: {resp.status_code}")
    print(f"  Detail: {resp.json()['detail']}")
    print("  ✅ Invalid session handled\n")


# ═════════════════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════════════════

async def main():
    import httpx

    # Tạo thư mục data nếu chưa có
    Path("./data").mkdir(parents=True, exist_ok=True)

    # Xóa test DB cũ nếu có
    test_db_path = Path("./data/test_api.db")
    test_db_path.unlink(missing_ok=True)

    app = _create_test_app()

    transport_kwargs = {"app": app}
    # Chủ động quản lý lifespan bằng app.router.lifespan_context để tương thích
    # nhiều phiên bản httpx và đảm bảo startup (create tables) luôn chạy.
    if "lifespan" in inspect.signature(httpx.ASGITransport.__init__).parameters:
        transport_kwargs["lifespan"] = "off"

    transport = httpx.ASGITransport(**transport_kwargs)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:

            print("\n" + "━" * 60)
            print("  🧪 API Controller Tests — Vòng 5")
            print("━" * 60 + "\n")

            # ── Health ───────────────────────────────────────────────────
            await test_health_check(client)

            # ── Chat Sessions CRUD ───────────────────────────────────────
            session_id = await test_create_session(client)
            default_id = await test_create_session_default_title(client)
            await test_list_sessions(client)
            await test_list_sessions_pagination(client)
            await test_get_session_detail(client, session_id)
            await test_get_session_not_found(client)

            # ── Chat Ask (edge cases) ────────────────────────────────────
            await test_chat_ask_stream_redirect(client, session_id)
            await test_chat_ask_invalid_session(client)

            # ── Cleanup sessions ─────────────────────────────────────────
            await test_delete_session(client, session_id)
            await test_delete_session(client, default_id)
            await test_delete_session_not_found(client)

            # ── Documents ────────────────────────────────────────────────
            docs_workspace_id = "workspace-api-tests"
            upload_data = await test_upload_document(client, docs_workspace_id)
            await test_upload_empty_file(client)
            await test_get_ingestion_status(client, upload_data["task_id"])
            await test_get_ingestion_status_not_found(client)
            await test_list_documents(client, docs_workspace_id)
            await test_get_document(client, upload_data["document_id"], docs_workspace_id)
            await test_get_document_not_found(client, docs_workspace_id)
            await test_delete_document(client, upload_data["document_id"], docs_workspace_id)

    # Cleanup test DB file
    test_db_path.unlink(missing_ok=True)

    print("🎉 All 20 API controller tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
