"""
RAG Generator — Ghép context + query, gọi LLM sinh câu trả lời.

Hỗ trợ 2 chế độ:
  - Sync: generate() → trả về ChatAnswerResponse đầy đủ
  - Streaming: generate_stream() → async generator yield từng token (SSE)

Workflow:
  1. Nhận retrieved chunks từ retriever
  2. Build context string từ chunks
  3. Dùng prompts.py format system/human messages
  4. Gọi LLM qua factory (Ollama hoặc Gemini)
  5. Parse response + trích citations
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.settings import settings
from app.rag.prompts import NO_CONTEXT_RESPONSE, QA_HUMAN_TEMPLATE, QA_SYSTEM_PROMPT
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Context builder — ghép chunks thành context string
# ═════════════════════════════════════════════════════════════════════════════

def build_context(chunks: list[RetrievedChunk]) -> str:
    """
    Ghép các retrieved chunks thành context string cho LLM.
    Mỗi chunk kèm metadata nguồn để LLM có thể trích dẫn.
    """
    if not chunks:
        return ""

    context_parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        source_label = chunk.filename or chunk.document_id or f"Source {i}"
        context_parts.append(
            f"--- Document [{source_label}] (relevance: {chunk.score:.2f}) ---\n"
            f"{chunk.text}\n"
        )

    return "\n".join(context_parts)


def extract_citations(
    chunks: list[RetrievedChunk],
) -> list[dict[str, Any]]:
    """
    Trích xuất danh sách citations từ retrieved chunks.
    Trả về format phù hợp với CitationItem schema.
    """
    citations: list[dict[str, Any]] = []
    seen_docs: set[str] = set()

    for chunk in chunks:
        # Deduplicate theo document_id
        doc_key = chunk.document_id or chunk.filename
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)

        citations.append({
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "chunk_text": chunk.text[:300],  # Preview 300 ký tự đầu
            "relevance_score": round(chunk.score, 4),
        })

    return citations


# ═════════════════════════════════════════════════════════════════════════════
# Sync Generation — trả về full response
# ═════════════════════════════════════════════════════════════════════════════

async def generate(
    query: str,
    chunks: list[RetrievedChunk],
    llm: Any,
) -> dict[str, Any]:
    """
    Generate câu trả lời từ RAG pipeline (sync mode).

    Args:
        query: Câu hỏi user
        chunks: Retrieved + reranked chunks
        llm: LangChain BaseChatModel instance

    Returns:
        dict với keys: answer, citations, context_used
    """
    # Nếu không có context, trả fallback
    if not chunks:
        logger.info("No context chunks provided, returning fallback response")
        return {
            "answer": NO_CONTEXT_RESPONSE,
            "citations": [],
            "context_used": False,
        }

    # Build context + messages
    context = build_context(chunks)

    system_msg = SystemMessage(content=QA_SYSTEM_PROMPT.format(context=context))
    human_msg = HumanMessage(content=QA_HUMAN_TEMPLATE.format(question=query))

    logger.info(
        "Generating answer: query_length=%d, context_chunks=%d, llm=%s",
        len(query), len(chunks), settings.llm_provider,
    )

    # Gọi LLM
    try:
        response = await llm.ainvoke([system_msg, human_msg])
        answer = response.content

        # Trích citations
        citations = extract_citations(chunks)

        logger.info("Generated answer: %d chars, %d citations", len(answer), len(citations))

        return {
            "answer": answer,
            "citations": citations,
            "context_used": True,
        }

    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        return {
            "answer": f"Đã xảy ra lỗi khi xử lý câu hỏi: {exc}",
            "citations": [],
            "context_used": False,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Streaming Generation — yield từng token qua SSE
# ═════════════════════════════════════════════════════════════════════════════

async def generate_stream(
    query: str,
    chunks: list[RetrievedChunk],
    llm: Any,
) -> AsyncGenerator[str, None]:
    """
    Generate câu trả lời streaming (Server-Sent Events).

    Yields:
        JSON strings, mỗi event chứa:
          - type="token", data=<token_text>
          - type="citations", data=<citations_list>
          - type="done"
          - type="error", data=<error_message>
    """
    # Nếu không có context → yield fallback
    if not chunks:
        yield json.dumps({"type": "token", "data": NO_CONTEXT_RESPONSE})
        yield json.dumps({"type": "citations", "data": []})
        yield json.dumps({"type": "done"})
        return

    # Build context + messages
    context = build_context(chunks)
    system_msg = SystemMessage(content=QA_SYSTEM_PROMPT.format(context=context))
    human_msg = HumanMessage(content=QA_HUMAN_TEMPLATE.format(question=query))

    logger.info(
        "Streaming answer: query_length=%d, context_chunks=%d",
        len(query), len(chunks),
    )

    try:
        # Stream tokens từ LLM
        full_answer = ""
        async for token_chunk in llm.astream([system_msg, human_msg]):
            token_text = token_chunk.content
            if token_text:
                full_answer += token_text
                yield json.dumps({"type": "token", "data": token_text})

        # Cuối cùng gửi citations
        citations = extract_citations(chunks)
        yield json.dumps({"type": "citations", "data": citations})
        yield json.dumps({"type": "done"})

        logger.info(
            "Stream complete: %d chars, %d citations",
            len(full_answer), len(citations),
        )

    except Exception as exc:
        logger.error("Streaming generation failed: %s", exc)
        yield json.dumps({"type": "error", "data": str(exc)})
