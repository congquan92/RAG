"""
RAG Admin Dashboard - Streamlit UI for monitoring and evaluation.

Run:
    streamlit run admin_ui/app.py --server.port 8501
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# Ensure server package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _run_async(coro):
    """Run coroutine safely from Streamlit rerun context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _init_db_once() -> None:
    """Initialize DB once for current Streamlit session."""
    if "db_initialized" in st.session_state:
        return

    from app.core.database import init_db

    _run_async(init_db())
    st.session_state.db_initialized = True


def _load_settings_snapshot() -> dict[str, Any]:
    """Read runtime config in a UI-safe way."""
    try:
        from app.core.settings import settings

        llm_model = (
            settings.ollama_model
            if settings.llm_provider == "ollama"
            else settings.gemini_model
        )
        return {
            "llm_provider": settings.llm_provider,
            "llm_model": llm_model,
            "embedding_provider": settings.embedding_provider,
            "embedding_model": settings.embedding_model,
            "embedding_device": settings.embedding_device,
            "docling_device": settings.docling_device,
            "reranker_device": settings.reranker_device,
            "chunk_size": settings.chunk_size,
            "retrieval_top_k": settings.retrieval_top_k,
            "reranker_top_k": settings.reranker_top_k,
            "has_gemini_key": bool(settings.gemini_api_key),
        }
    except Exception:
        return {}


def _format_datetime(dt: Optional[datetime]) -> str:
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M")


def _parse_citations(citations_raw: Any) -> list[dict[str, Any]]:
    if not citations_raw:
        return []

    try:
        data = json.loads(citations_raw)
    except (json.JSONDecodeError, TypeError):
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


async def _fetch_overview_stats() -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.core.database import async_session_factory
    from app.models.chat import ChatMessage, ChatSession
    from app.models.document import Document, IngestionTask

    async with async_session_factory() as db:
        session_count = (
            await db.execute(select(func.count(ChatSession.id)))
        ).scalar() or 0
        message_count = (
            await db.execute(select(func.count(ChatMessage.id)))
        ).scalar() or 0
        document_count = (
            await db.execute(select(func.count(Document.id)))
        ).scalar() or 0
        chunk_count = (
            await db.execute(select(func.sum(Document.chunk_count)))
        ).scalar() or 0

        task_rows = (
            await db.execute(
                select(IngestionTask.status, func.count(IngestionTask.id)).group_by(
                    IngestionTask.status
                )
            )
        ).all()

        latest_task_rows = (
            await db.execute(
                select(IngestionTask)
                .order_by(IngestionTask.updated_at.desc())
                .limit(12)
            )
        ).scalars().all()

        return {
            "sessions": session_count,
            "messages": message_count,
            "documents": document_count,
            "chunks": chunk_count,
            "tasks": {row[0]: row[1] for row in task_rows},
            "latest_tasks": [
                {
                    "Task ID": task.id,
                    "Document ID": task.document_id,
                    "Status": task.status,
                    "Chunks": task.chunks_processed,
                    "Updated": _format_datetime(task.updated_at),
                    "Error": task.error_message or "-",
                }
                for task in latest_task_rows
            ],
        }


async def _fetch_documents(limit: int = 200) -> tuple[list[dict[str, Any]], int]:
    from app.core.database import async_session_factory
    from app.services.document_service import list_documents

    async with async_session_factory() as db:
        documents, total = await list_documents(db, skip=0, limit=limit)

    rows = [
        {
            "ID": doc.id,
            "Filename": doc.filename,
            "MIME": doc.mime_type or "-",
            "Size KB": round((doc.file_size or 0) / 1024, 2),
            "Chunks": doc.chunk_count,
            "Uploaded": _format_datetime(doc.created_at),
        }
        for doc in documents
    ]
    return rows, total


async def _fetch_ingestion_tasks(limit: int = 200) -> list[dict[str, Any]]:
    from sqlalchemy import select

    from app.core.database import async_session_factory
    from app.models.document import IngestionTask

    async with async_session_factory() as db:
        tasks = (
            await db.execute(
                select(IngestionTask)
                .order_by(IngestionTask.created_at.desc())
                .limit(limit)
            )
        ).scalars().all()

    return [
        {
            "Task ID": task.id,
            "Document ID": task.document_id,
            "Status": task.status,
            "Chunks": task.chunks_processed,
            "Created": _format_datetime(task.created_at),
            "Updated": _format_datetime(task.updated_at),
            "Error": task.error_message or "-",
        }
        for task in tasks
    ]


async def _fetch_sessions(limit: int = 150) -> tuple[list[dict[str, Any]], int]:
    from app.core.database import async_session_factory
    from app.services.chat_service import list_sessions

    async with async_session_factory() as db:
        return await list_sessions(db, skip=0, limit=limit)


async def _fetch_session_messages(session_id: str):
    from app.core.database import async_session_factory
    from app.services.chat_service import get_session_messages

    async with async_session_factory() as db:
        return await get_session_messages(db, session_id)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            :root {
                --panel-bg: #f3f6fa;
                --card-bg: #ffffff;
                --ink-main: #1d2a39;
                --ink-soft: #5b6776;
                --accent: #00897b;
                --accent-2: #f4511e;
            }
            .stApp {
                background: radial-gradient(circle at 20% 0%, #f8fbff 0%, #eff6f9 45%, #f7f4ed 100%);
                color: var(--ink-main);
            }
            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2.5rem;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f2438 0%, #15314b 100%);
            }
            div[data-testid="stSidebar"] * {
                color: #f4f9ff;
            }
            .admin-header {
                background: linear-gradient(120deg, #ffffff 0%, #e8f6f3 55%, #ffe8df 100%);
                border: 1px solid rgba(0, 137, 123, 0.2);
                border-radius: 16px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
            }
            .admin-subtle {
                color: var(--ink-soft);
                margin-top: 0.2rem;
                margin-bottom: 0;
            }
            .quick-pill {
                display: inline-block;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                border: 1px solid rgba(0, 0, 0, 0.08);
                margin-right: 0.35rem;
                font-size: 0.8rem;
                background: var(--card-bg);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(last_refresh_at: str) -> None:
    st.markdown(
        (
            '<div class="admin-header">'
            "<h2 style='margin:0;'>RAG Admin Workspace</h2>"
            "<p class='admin-subtle'>"
            "Monitor ingestion, inspect chat sessions, and run quality evaluation from one place."
            "</p>"
            f"<p class='admin-subtle'>Last refresh: {last_refresh_at}</p>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_sidebar(settings_snapshot: dict[str, Any]) -> str:
    with st.sidebar:
        st.title("RAG Admin")
        st.caption("Operations and evaluation console")

        page = st.radio(
            "Section",
            ["Overview", "Documents", "Chat History", "Evaluation"],
            index=0,
        )

        if st.button("Refresh now", type="primary", use_container_width=True):
            st.session_state.last_refresh_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()

        st.divider()
        st.markdown("### Runtime")

        if settings_snapshot:
            st.write(f"LLM: {settings_snapshot['llm_provider']}")
            st.write(f"Model: {settings_snapshot['llm_model']}")
            st.write(f"Embeddings: {settings_snapshot['embedding_provider']}")
            st.write(f"Embed Device: {settings_snapshot['embedding_device']}")
            st.write(f"Docling Device: {settings_snapshot['docling_device']}")
            st.write(f"Reranker Device: {settings_snapshot['reranker_device']}")
        else:
            st.warning("Could not load runtime settings.")

    return page


def _render_overview(settings_snapshot: dict[str, Any]) -> None:
    st.subheader("System Overview")
    stats = _run_async(_fetch_overview_stats())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", stats["sessions"])
    c2.metric("Messages", stats["messages"])
    c3.metric("Documents", stats["documents"])
    c4.metric("Chunks", stats["chunks"])

    st.markdown("### Ingestion Status")
    status_order = ["pending", "processing", "completed", "failed"]
    status_counts = stats["tasks"]

    if not status_counts:
        st.info("No ingestion tasks yet.")
    else:
        cols = st.columns(4)
        for idx, status in enumerate(status_order):
            cols[idx].metric(status.capitalize(), status_counts.get(status, 0))

    if stats["latest_tasks"]:
        st.markdown("### Latest Tasks")
        st.dataframe(stats["latest_tasks"], use_container_width=True, hide_index=True)

    if settings_snapshot:
        st.markdown("### Configuration Snapshot")
        st.markdown(
            (
                f"<span class='quick-pill'>TopK Retrieve: {settings_snapshot['retrieval_top_k']}</span>"
                f"<span class='quick-pill'>TopK Rerank: {settings_snapshot['reranker_top_k']}</span>"
                f"<span class='quick-pill'>Chunk Size: {settings_snapshot['chunk_size']}</span>"
            ),
            unsafe_allow_html=True,
        )


def _render_documents() -> None:
    st.subheader("Document Management")
    docs, total = _run_async(_fetch_documents(limit=500))
    tasks = _run_async(_fetch_ingestion_tasks(limit=500))

    col1, col2, col3 = st.columns([2, 1, 1])
    search_term = col1.text_input("Search by filename", placeholder="example: policy, report, q1")
    mime_filter = col2.selectbox(
        "MIME type",
        options=["All"] + sorted({d["MIME"] for d in docs}),
    )
    min_chunks = col3.number_input("Min chunks", min_value=0, value=0, step=1)

    filtered_docs = [
        d
        for d in docs
        if (not search_term or search_term.lower() in d["Filename"].lower())
        and (mime_filter == "All" or d["MIME"] == mime_filter)
        and d["Chunks"] >= min_chunks
    ]

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Documents", total)
    s2.metric("Displayed", len(filtered_docs))
    s3.metric("Recent Tasks", len(tasks))

    if filtered_docs:
        st.dataframe(filtered_docs, use_container_width=True, hide_index=True)
    else:
        st.info("No documents match current filters.")

    st.markdown("### Ingestion Tasks")
    task_statuses = ["All"] + sorted({t["Status"] for t in tasks})
    status_filter = st.selectbox("Task status", task_statuses)

    filtered_tasks = [
        t for t in tasks if status_filter == "All" or t["Status"] == status_filter
    ]

    if filtered_tasks:
        st.dataframe(filtered_tasks, use_container_width=True, hide_index=True)
    else:
        st.info("No tasks available for this status.")


def _render_chat_history() -> None:
    st.subheader("Chat Sessions")
    sessions, total = _run_async(_fetch_sessions(limit=200))

    st.metric("Total Sessions", total)

    if not sessions:
        st.info("No chat sessions yet.")
        return

    search_text = st.text_input("Search sessions", placeholder="Find by title or id")

    filtered_sessions = [
        s
        for s in sessions
        if not search_text
        or search_text.lower() in (s.get("title") or "").lower()
        or search_text.lower() in s.get("id", "").lower()
    ]

    if not filtered_sessions:
        st.info("No sessions match your search.")
        return

    option_map = {
        (
            f"{s.get('title') or 'Untitled'} "
            f"| msgs: {s.get('message_count', 0)} "
            f"| updated: {_format_datetime(s.get('updated_at'))}"
        ): s
        for s in filtered_sessions
    }

    selected_label = st.selectbox("Select a session", options=list(option_map.keys()))
    selected = option_map[selected_label]

    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.write(f"Session ID: {selected.get('id')}")
    info_col2.write(f"Created: {_format_datetime(selected.get('created_at'))}")
    info_col3.write(f"Messages: {selected.get('message_count', 0)}")

    if selected.get("description"):
        st.caption(selected["description"])

    messages = _run_async(_fetch_session_messages(selected["id"]))

    if not messages:
        st.info("No messages in this session.")
        return

    st.markdown("### Message Timeline")
    for idx, msg in enumerate(messages, start=1):
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            st.caption(f"#{idx} | {_format_datetime(msg.created_at)}")

            if msg.role == "assistant":
                citations = _parse_citations(msg.citations)
                if citations:
                    with st.expander(f"Citations ({len(citations)})"):
                        for citation in citations:
                            source = citation.get("filename") or citation.get("document_id") or "Unknown"
                            score = citation.get("relevance_score")
                            score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "-"
                            st.markdown(f"Source: {source} | score: {score_text}")
                            chunk = citation.get("chunk_text")
                            if chunk:
                                st.caption(chunk[:350])
                            st.divider()


def _render_evaluation(settings_snapshot: dict[str, Any]) -> None:
    st.subheader("RAG Quality Evaluation")
    st.write(
        "Run Ragas metrics on recent chat data. "
        "Gemini is used as critic model for evaluation."
    )

    from admin_ui.evaluator import evaluate_sessions, quick_check

    preview = _run_async(quick_check())

    c1, c2, c3 = st.columns(3)
    c1.metric("Q&A Pairs", preview["total_pairs"])
    c2.metric("Sessions", len(preview["sessions"]))
    c3.metric("Pairs with Context", preview["has_contexts"])

    if preview["sample_questions"]:
        with st.expander("Sample Questions"):
            for question in preview["sample_questions"]:
                st.write(f"- {question}")

    st.markdown("### Run Evaluation")
    eval_limit = st.slider(
        "Max recent sessions",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of latest sessions used for evaluation.",
    )

    has_gemini = bool(settings_snapshot.get("has_gemini_key"))
    if not has_gemini:
        st.warning("GEMINI_API_KEY is not configured. Evaluation cannot run.")

    can_run = has_gemini and preview["total_pairs"] > 0
    if st.button("Run Ragas Evaluation", disabled=not can_run, type="primary"):
        with st.spinner("Running evaluation..."):
            result = _run_async(evaluate_sessions(limit=eval_limit))

        st.session_state.last_eval = result

        if result["error"]:
            st.error(result["error"])
        else:
            st.success(f"Evaluated {result['total_questions']} Q&A pairs.")
            _render_eval_result(result)

    if st.session_state.get("last_eval"):
        st.markdown("### Last Evaluation")
        _render_eval_result(st.session_state["last_eval"])
        st.download_button(
            label="Download last result as JSON",
            data=json.dumps(
                st.session_state["last_eval"],
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            file_name="rag_eval_result.json",
            mime="application/json",
        )


def _render_eval_result(result: dict[str, Any]) -> None:
    if result.get("error"):
        st.error(result["error"])
        return

    scores = result.get("scores") or {}
    if scores:
        st.markdown("#### Overall Scores")
        cols = st.columns(len(scores))
        for idx, (metric, score) in enumerate(scores.items()):
            label = metric.replace("_", " ").title()
            if score >= 0.7:
                state = "Strong"
            elif score >= 0.4:
                state = "Medium"
            else:
                state = "Weak"
            cols[idx].metric(label, f"{score:.2%}", delta=state)

    details = result.get("details") or []
    if details:
        st.markdown("#### Per Question")
        st.dataframe(details, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(
        page_title="RAG Admin Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_styles()
    _init_db_once()

    if "last_refresh_at" not in st.session_state:
        st.session_state.last_refresh_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    settings_snapshot = _load_settings_snapshot()
    page = _render_sidebar(settings_snapshot)
    _render_header(st.session_state.last_refresh_at)

    if page == "Overview":
        _render_overview(settings_snapshot)
    elif page == "Documents":
        _render_documents()
    elif page == "Chat History":
        _render_chat_history()
    else:
        _render_evaluation(settings_snapshot)


if __name__ == "__main__":
    main()
