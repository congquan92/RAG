"""
Test Script: Business Services (Document + Chat)
Kiểm tra logic nghiệp vụ: upload, ingestion, session, query.

Usage:
    cd server
    source .venv/bin/activate
    python scripts/test_services.py
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def test_document_service_upload():
    """Test upload file + tạo DB records."""
    from app.core.database import async_session_factory, init_db
    from app.services.document_service import save_uploaded_file

    print("=" * 60)
    print("  TEST: Document Service — Upload")
    print("=" * 60)

    await init_db()

    async with async_session_factory() as db:
        # Tạo file content giả
        content = b"This is a test document content for ingestion testing."

        document, task = await save_uploaded_file(
            db=db,
            filename="test_report.txt",
            file_content=content,
        )
        await db.commit()

        print(f"  Document ID:   {document.id}")
        print(f"  Filename:      {document.filename}")
        print(f"  File path:     {document.file_path}")
        print(f"  File size:     {document.file_size} bytes")
        print(f"  MIME type:     {document.mime_type}")
        print(f"  Task ID:       {task.id}")
        print(f"  Task status:   {task.status}")

        assert document.filename == "test_report.txt"
        assert document.file_size == len(content)
        assert task.status == "pending"
        assert Path(document.file_path).exists(), "Uploaded file should exist on disk"

        print("  ✅ Upload + DB records created!\n")

        # Cleanup file
        Path(document.file_path).unlink(missing_ok=True)
        return document.id, task.id


async def test_document_service_list():
    """Test list documents."""
    from app.core.database import async_session_factory
    from app.services.document_service import list_documents

    print("=" * 60)
    print("  TEST: Document Service — List Documents")
    print("=" * 60)

    async with async_session_factory() as db:
        documents, total = await list_documents(db, skip=0, limit=10)

        print(f"  Total documents: {total}")
        for doc in documents:
            print(f"    - {doc.filename} (id={doc.id[:8]}..., chunks={doc.chunk_count})")

        assert total >= 1, "Should have at least 1 document from previous test"
        print("  ✅ List documents works!\n")


async def test_document_service_status():
    """Test get task status."""
    from app.core.database import async_session_factory
    from app.services.document_service import get_task_status

    print("=" * 60)
    print("  TEST: Document Service — Task Status")
    print("=" * 60)

    async with async_session_factory() as db:
        # Task không tồn tại
        none_task = await get_task_status(db, "nonexistent-id")
        assert none_task is None
        print("  ✅ Non-existent task returns None")
        print("  ✅ Task status query works!\n")


async def test_chat_service_session():
    """Test CRUD session."""
    from app.core.database import async_session_factory
    from app.services.chat_service import (
        create_session, get_session, list_sessions, delete_session
    )

    print("=" * 60)
    print("  TEST: Chat Service — Session CRUD")
    print("=" * 60)

    async with async_session_factory() as db:
        # Create
        session = await create_session(db, title="Test Session")
        await db.commit()
        print(f"  Created:  id={session.id[:8]}..., title='{session.title}'")

        # Get
        fetched = await get_session(db, session.id)
        assert fetched is not None
        assert fetched.title == "Test Session"
        print(f"  Fetched:  title='{fetched.title}'")

        # List
        sessions, total = await list_sessions(db)
        assert total >= 1
        print(f"  Listed:   {total} sessions")

        # Delete
        deleted = await delete_session(db, session.id)
        await db.commit()
        assert deleted is not None
        print(f"  Deleted:  id={deleted.id[:8]}...")

        # Verify delete
        verify = await get_session(db, session.id)
        assert verify is None
        print("  Verified: session no longer exists")

        print("  ✅ Session CRUD works!\n")


async def test_chat_service_ask_no_context():
    """Test ask_question khi chưa có documents (fallback response)."""
    from app.core.database import async_session_factory
    from app.services.chat_service import create_session, ask_question, get_session_messages

    print("=" * 60)
    print("  TEST: Chat Service — Ask (No Context)")
    print("=" * 60)

    async with async_session_factory() as db:
        # Tạo session
        session = await create_session(db, title="New Conversation")
        await db.commit()

        # Mock embeddings và LLM (không cần thật vì sẽ không có context)
        class MockEmbeddings:
            def embed_query(self, text):
                return [0.0] * 384

        # Ask question — sẽ retrieve empty → fallback response
        result = await ask_question(
            db=db,
            session_id=session.id,
            query="Python là gì?",
            embeddings=MockEmbeddings(),
            reranker=None,
            llm=None,  # Không cần LLM khi không có context
        )
        await db.commit()

        print(f"  Session ID:     {result['session_id']}")
        print(f"  Answer:         {result['answer'][:80]}...")
        print(f"  Citations:      {len(result['citations'])}")
        print(f"  User msg ID:    {result['user_message_id'][:8]}...")
        print(f"  Assist msg ID:  {result['assistant_message_id'][:8]}...")

        # Verify messages saved
        messages = await get_session_messages(db, session.id)
        assert messages is not None
        assert len(messages) == 2, f"Expected 2 messages (user + assistant), got {len(messages)}"
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        print(f"  Messages saved: {len(messages)} (user + assistant)")

        # Verify auto-title
        updated_session = await db.get(session.__class__, session.id)
        assert updated_session.title != "New Conversation", "Title should auto-update"
        print(f"  Auto-title:     '{updated_session.title}'")

        print("  ✅ Ask question (no context) works!\n")

        # Cleanup
        await db.delete(updated_session)
        await db.commit()


async def test_document_delete():
    """Test delete document + cleanup."""
    from app.core.database import async_session_factory
    from app.services.document_service import (
        save_uploaded_file, delete_document, get_document
    )

    print("=" * 60)
    print("  TEST: Document Service — Delete")
    print("=" * 60)

    async with async_session_factory() as db:
        # Create
        doc, task = await save_uploaded_file(
            db=db,
            filename="to_delete.txt",
            file_content=b"Delete me",
        )
        await db.commit()
        file_path = Path(doc.file_path)
        doc_id = doc.id
        assert file_path.exists()
        print(f"  Created: {doc.filename} (id={doc_id[:8]}...)")

        # Delete
        deleted = await delete_document(db, doc_id)
        await db.commit()
        assert deleted is not None
        assert not file_path.exists(), "File should be deleted from disk"
        print(f"  Deleted: file removed from disk")

        # Verify
        verify = await get_document(db, doc_id)
        assert verify is None
        print("  Verified: document no longer in DB")

        print("  ✅ Document delete works!\n")


async def test_chat_stream_cancel_cleanup():
    """Test stream cancel: không được để assistant placeholder rỗng trong DB."""
    from app.core.database import async_session_factory
    from app.rag.retriever import RetrievalResult
    from app.services import chat_service

    print("=" * 60)
    print("  TEST: Chat Service — Stream Cancel Cleanup")
    print("=" * 60)

    async with async_session_factory() as db:
        session = await chat_service.create_session(db, title="Cancel Stream Session")
        await db.commit()

        disconnect_state = {"value": False}

        async def is_client_disconnected() -> bool:
            return bool(disconnect_state["value"])

        original_retrieve = chat_service.retrieve
        chat_service.retrieve = lambda **_: RetrievalResult(
            chunks=[],
            raw_semantic=[],
            raw_keyword=[],
            raw_graph=[],
        )

        try:
            seen_event_types: list[str] = []
            async for event_str in chat_service.ask_question_stream(
                db=db,
                session_id=session.id,
                query="Test stop stream",
                embeddings=None,
                reranker=None,
                llm=None,
                is_client_disconnected=is_client_disconnected,
            ):
                event = json.loads(event_str)
                event_type = str(event.get("type", ""))
                seen_event_types.append(event_type)
                if event_type == "message_ids":
                    disconnect_state["value"] = True

            await db.commit()

            messages = await chat_service.get_session_messages(db, session.id)
            assert messages is not None
            assert len(messages) == 1, f"Expected only user message after cancel, got {len(messages)}"
            assert messages[0].role == "user", "Only user message should remain after early cancel"
            assert "message_ids" in seen_event_types, "Expected message_ids event before cancellation"

            print(f"  Events:        {seen_event_types}")
            print("  ✅ Placeholder assistant message cleaned up on cancel!\n")
        finally:
            chat_service.retrieve = original_retrieve
            persisted_session = await db.get(session.__class__, session.id)
            if persisted_session is not None:
                await db.delete(persisted_session)
                await db.commit()


async def main():
    await test_document_service_upload()
    await test_document_service_list()
    await test_document_service_status()
    await test_chat_service_session()
    await test_chat_service_ask_no_context()
    await test_document_delete()
    await test_chat_stream_cancel_cleanup()

    # Final cleanup
    from app.core.database import dispose_db
    await dispose_db()

    print("🎉 All service tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
