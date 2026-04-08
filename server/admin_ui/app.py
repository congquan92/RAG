"""
RAG Admin Dashboard — Streamlit UI cho giám sát & đánh giá.

Chạy: streamlit run admin_ui/app.py --server.port 8501
(Từ thư mục server/)

Tabs:
  1. 🏠 Overview    — System health, thống kê tổng quan
  2. 📄 Documents   — Quản lý documents, xem trạng thái ingestion
  3. 💬 Chat History — Xem lịch sử hội thoại, messages
  4. 🧪 Evaluation   — Chạy Ragas evaluation, xem điểm chất lượng AI
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Ensure server package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═════════════════════════════════════════════════════════════════════════════
# Helpers — chạy async code trong Streamlit (sync event loop)
# ═════════════════════════════════════════════════════════════════════════════

def _run_async(coro):
    """Chạy coroutine trong Streamlit (không có event loop sẵn)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _init_db_once():
    """Init database 1 lần duy nhất cho session."""
    if "db_initialized" not in st.session_state:
        from app.core.database import init_db
        _run_async(init_db())
        st.session_state.db_initialized = True


# ═════════════════════════════════════════════════════════════════════════════
# Page Config
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG Admin Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.8rem;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; }
    .metric-card p { margin: 0.2rem 0 0; opacity: 0.85; font-size: 0.9rem; }
    .score-good { color: #00c853; font-weight: bold; }
    .score-mid { color: #ff9100; font-weight: bold; }
    .score-bad { color: #ff1744; font-weight: bold; }
    div[data-testid="stSidebar"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

_init_db_once()


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("🧠 RAG Admin")
    st.caption("Enterprise RAG Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "📄 Documents", "💬 Chat History", "🧪 Evaluation"],
        index=0,
    )
    st.divider()

    # Settings info
    try:
        from app.core.settings import settings
        st.markdown("**⚙️ Config**")
        st.text(f"LLM:   {settings.llm_provider}")
        st.text(f"Embed: {settings.embedding_provider}")
        st.text(f"DB:    SQLite (aiosqlite)")
    except Exception:
        st.warning("Cannot load settings")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 1: Overview
# ═════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("🏠 System Overview")
    st.markdown("Tổng quan trạng thái hệ thống RAG.")

    # Fetch stats
    async def _get_stats():
        from sqlalchemy import select, func
        from app.core.database import async_session_factory
        from app.models.chat import ChatSession, ChatMessage
        from app.models.document import Document, IngestionTask

        async with async_session_factory() as db:
            # Counts
            sess_count = (await db.execute(
                select(func.count(ChatSession.id))
            )).scalar() or 0

            msg_count = (await db.execute(
                select(func.count(ChatMessage.id))
            )).scalar() or 0

            doc_count = (await db.execute(
                select(func.count(Document.id))
            )).scalar() or 0

            # Task status breakdown
            tasks = (await db.execute(
                select(IngestionTask.status, func.count(IngestionTask.id))
                .group_by(IngestionTask.status)
            )).all()
            task_stats = {row[0]: row[1] for row in tasks}

            # Total chunks
            total_chunks = (await db.execute(
                select(func.sum(Document.chunk_count))
            )).scalar() or 0

            return {
                "sessions": sess_count,
                "messages": msg_count,
                "documents": doc_count,
                "chunks": total_chunks,
                "tasks": task_stats,
            }

    stats = _run_async(_get_stats())

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 Sessions", stats["sessions"])
    with col2:
        st.metric("📝 Messages", stats["messages"])
    with col3:
        st.metric("📄 Documents", stats["documents"])
    with col4:
        st.metric("🧩 Chunks", stats["chunks"])

    st.divider()

    # Ingestion task status
    st.subheader("📊 Ingestion Task Status")
    if stats["tasks"]:
        task_cols = st.columns(len(stats["tasks"]))
        status_emoji = {
            "pending": "⏳",
            "processing": "⚙️",
            "completed": "✅",
            "failed": "❌",
        }
        for i, (status, count) in enumerate(stats["tasks"].items()):
            with task_cols[i]:
                emoji = status_emoji.get(status, "❓")
                st.metric(f"{emoji} {status.capitalize()}", count)
    else:
        st.info("No ingestion tasks yet.")

    # Server config
    st.divider()
    st.subheader("⚙️ Server Configuration")
    try:
        config_data = {
            "LLM Provider": settings.llm_provider,
            "LLM Model": settings.ollama_model if settings.llm_provider == "ollama" else settings.gemini_model,
            "Embedding Provider": settings.embedding_provider,
            "Embedding Model": settings.embedding_model,
            "Embedding Device": settings.embedding_device,
            "Reranker Device": settings.reranker_device,
            "Chunk Size": settings.chunk_size,
            "Top-K Retrieval": settings.retrieval_top_k,
            "Top-K Reranker": settings.reranker_top_k,
        }
        left, right = st.columns(2)
        items = list(config_data.items())
        for k, v in items[:len(items)//2]:
            left.text(f"{k}: {v}")
        for k, v in items[len(items)//2:]:
            right.text(f"{k}: {v}")
    except Exception as exc:
        st.error(f"Cannot load config: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 2: Documents
# ═════════════════════════════════════════════════════════════════════════════

elif page == "📄 Documents":
    st.title("📄 Document Management")

    async def _get_documents():
        from app.core.database import async_session_factory
        from app.services.document_service import list_documents

        async with async_session_factory() as db:
            docs, total = await list_documents(db, skip=0, limit=100)
            return [
                {
                    "ID": doc.id[:12] + "…",
                    "Filename": doc.filename,
                    "Size (KB)": round(doc.file_size / 1024, 1),
                    "MIME Type": doc.mime_type or "—",
                    "Chunks": doc.chunk_count,
                    "Uploaded": doc.created_at.strftime("%Y-%m-%d %H:%M") if doc.created_at else "—",
                }
                for doc in docs
            ], total

    docs_data, total = _run_async(_get_documents())

    st.metric("Total Documents", total)

    if docs_data:
        st.dataframe(docs_data, use_container_width=True, hide_index=True)
    else:
        st.info("No documents uploaded yet. Use POST /api/v1/documents/upload to upload files.")

    # Ingestion tasks detail
    st.divider()
    st.subheader("🔄 Recent Ingestion Tasks")

    async def _get_tasks():
        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models.document import IngestionTask

        async with async_session_factory() as db:
            result = await db.execute(
                select(IngestionTask).order_by(IngestionTask.created_at.desc()).limit(20)
            )
            tasks = list(result.scalars().all())
            return [
                {
                    "Task ID": t.id[:12] + "…",
                    "Doc ID": t.document_id[:12] + "…",
                    "Status": t.status,
                    "Chunks": t.chunks_processed,
                    "Error": (t.error_message[:50] + "…") if t.error_message else "—",
                    "Updated": t.updated_at.strftime("%Y-%m-%d %H:%M") if t.updated_at else "—",
                }
                for t in tasks
            ]

    tasks_data = _run_async(_get_tasks())
    if tasks_data:
        st.dataframe(tasks_data, use_container_width=True, hide_index=True)
    else:
        st.info("No ingestion tasks yet.")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 3: Chat History
# ═════════════════════════════════════════════════════════════════════════════

elif page == "💬 Chat History":
    st.title("💬 Chat History")

    async def _get_sessions():
        from app.core.database import async_session_factory
        from app.services.chat_service import list_sessions

        async with async_session_factory() as db:
            sessions, total = await list_sessions(db, skip=0, limit=100)
            return sessions, total

    sessions_data, total = _run_async(_get_sessions())

    st.metric("Total Sessions", total)

    if not sessions_data:
        st.info("No chat sessions yet. Start a conversation via POST /api/v1/chat/sessions.")
    else:
        # Session selector
        session_options = {
            f"{s['title'][:50]}  ({s['message_count']} msgs)": s["id"]
            for s in sessions_data
        }

        selected_label = st.selectbox(
            "Select a session",
            options=list(session_options.keys()),
        )

        if selected_label:
            selected_id = session_options[selected_label]

            # Fetch messages
            async def _get_messages(sid):
                from app.core.database import async_session_factory
                from app.services.chat_service import get_session_messages

                async with async_session_factory() as db:
                    return await get_session_messages(db, sid)

            messages = _run_async(_get_messages(selected_id))

            if messages:
                st.divider()
                for msg in messages:
                    role_icon = "🧑" if msg.role == "user" else "🤖"
                    with st.chat_message(msg.role):
                        st.markdown(msg.content)

                        # Show citations nếu có
                        if msg.citations and msg.role == "assistant":
                            try:
                                citations = json.loads(msg.citations)
                                if citations:
                                    with st.expander(f"📚 Citations ({len(citations)})"):
                                        for c in citations:
                                            if isinstance(c, dict):
                                                st.markdown(
                                                    f"**{c.get('filename', 'Unknown')}** "
                                                    f"(score: {c.get('relevance_score', 0):.2f})"
                                                )
                                                if c.get("chunk_text"):
                                                    st.caption(c["chunk_text"][:200] + "…")
                            except (json.JSONDecodeError, TypeError):
                                pass
            else:
                st.info("No messages in this session.")


# ═════════════════════════════════════════════════════════════════════════════
# Tab 4: Evaluation
# ═════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Evaluation":
    st.title("🧪 RAG Quality Evaluation")
    st.markdown(
        "Sử dụng **Ragas** + **Gemini** (giám khảo) để chấm điểm chất lượng "
        "câu trả lời của hệ thống RAG.\n\n"
        "Gemini → đánh giá → Ollama's answers."
    )

    st.divider()

    # Quick check data availability
    st.subheader("📋 Data Preview")

    from admin_ui.evaluator import quick_check, evaluate_sessions

    preview = _run_async(quick_check())

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Q&A Pairs Available", preview["total_pairs"])
    with col2:
        st.metric("Sessions with Data", len(preview["sessions"]))
    with col3:
        st.metric("Pairs with Context", preview["has_contexts"])

    if preview["sample_questions"]:
        with st.expander("Sample Questions"):
            for q in preview["sample_questions"]:
                st.text(f"• {q}")

    st.divider()

    # Evaluation controls
    st.subheader("🚀 Run Evaluation")

    eval_limit = st.slider(
        "Max sessions to evaluate",
        min_value=1,
        max_value=50,
        value=10,
        help="Số sessions gần nhất sẽ được đánh giá.",
    )

    # Gemini API key check
    try:
        has_gemini = bool(settings.gemini_api_key)
    except Exception:
        has_gemini = False

    if not has_gemini:
        st.warning(
            "⚠️ GEMINI_API_KEY chưa được cấu hình trong `.env`. "
            "Cần Gemini làm giám khảo (critic LLM) để chạy evaluation."
        )

    if preview["total_pairs"] == 0:
        st.info("Chưa có dữ liệu Q&A. Hỏi vài câu qua API trước khi đánh giá.")

    # Run button
    can_run = has_gemini and preview["total_pairs"] > 0
    if st.button("🧪 Run Ragas Evaluation", disabled=not can_run, type="primary"):
        with st.spinner("Đang chạy Ragas evaluation... (có thể mất vài phút)"):
            eval_result = _run_async(evaluate_sessions(limit=eval_limit))

        if eval_result["error"]:
            st.error(f"❌ {eval_result['error']}")
        else:
            st.success(f"✅ Đánh giá xong {eval_result['total_questions']} câu hỏi!")

            # Score display
            st.subheader("📊 Overall Scores")
            score_cols = st.columns(len(eval_result["scores"]))

            for i, (metric, score) in enumerate(eval_result["scores"].items()):
                with score_cols[i]:
                    # Color coding
                    if score >= 0.7:
                        delta_color = "normal"
                        label = "Good"
                    elif score >= 0.4:
                        delta_color = "off"
                        label = "Needs Improvement"
                    else:
                        delta_color = "inverse"
                        label = "Poor"

                    display_name = metric.replace("_", " ").title()
                    st.metric(
                        display_name,
                        f"{score:.2%}",
                        delta=label,
                        delta_color=delta_color,
                    )

            # Per-question details
            if eval_result["details"]:
                st.divider()
                st.subheader("📝 Per-Question Details")
                st.dataframe(
                    eval_result["details"],
                    use_container_width=True,
                    hide_index=True,
                )

    # Saved results display (from session state)
    if "last_eval" in st.session_state:
        st.divider()
        st.subheader("📜 Last Evaluation Result")
        st.json(st.session_state.last_eval)
