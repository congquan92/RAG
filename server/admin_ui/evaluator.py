"""
RAG Evaluator — Đánh giá chất lượng câu trả lời RAG bằng Ragas.

Sử dụng Gemini làm "giám khảo" (critic LLM) chấm điểm Ollama.
Lôi lịch sử Q&A từ SQLite → build Ragas dataset → evaluate.

Metrics đánh giá:
  - Faithfulness: Câu trả lời có dựa trên context đã retrieve không?
  - Answer Relevancy: Câu trả lời có trả lời đúng câu hỏi không?
  - Context Precision: Context retrieved có liên quan đến câu hỏi?
  - Context Recall: Context có bao phủ đủ info để trả lời?

Usage:
    from admin_ui.evaluator import evaluate_sessions
    results = await evaluate_sessions(session_ids=[...])
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure server package importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data Collection — Lôi lịch sử từ SQLite
# ═════════════════════════════════════════════════════════════════════════════

async def _fetch_qa_pairs(
    session_ids: Optional[list[str]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Lấy các cặp Q&A (user → assistant) từ DB.

    Returns:
        List[dict] với keys: question, answer, contexts (từ citations)
    """
    from sqlalchemy import select
    from app.core.database import async_session_factory
    from app.models.chat import ChatMessage, ChatSession

    qa_pairs: list[dict[str, Any]] = []

    async with async_session_factory() as db:
        # Lấy sessions
        if session_ids:
            stmt = select(ChatSession).where(ChatSession.id.in_(session_ids))
        else:
            stmt = (
                select(ChatSession)
                .order_by(ChatSession.updated_at.desc())
                .limit(limit)
            )
        result = await db.execute(stmt)
        sessions = list(result.scalars().all())

        for session in sessions:
            # Lấy messages trong session
            msg_stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session.id)
                .order_by(ChatMessage.created_at)
            )
            msg_result = await db.execute(msg_stmt)
            messages = list(msg_result.scalars().all())

            # Ghép từng cặp user → assistant
            i = 0
            while i < len(messages) - 1:
                user_msg = messages[i]
                asst_msg = messages[i + 1]

                if user_msg.role == "user" and asst_msg.role == "assistant":
                    # Parse citations ra contexts
                    contexts = []
                    if asst_msg.citations:
                        try:
                            citation_data = json.loads(asst_msg.citations)
                            for c in citation_data:
                                if isinstance(c, dict) and c.get("chunk_text"):
                                    contexts.append(c["chunk_text"])
                        except (json.JSONDecodeError, TypeError):
                            pass

                    qa_pairs.append({
                        "question": user_msg.content,
                        "answer": asst_msg.content,
                        "contexts": contexts if contexts else ["No context retrieved."],
                        "session_id": session.id,
                        "session_title": session.title,
                    })
                    i += 2
                else:
                    i += 1

    logger.info("Fetched %d Q&A pairs from %d sessions", len(qa_pairs), len(sessions))
    return qa_pairs


# ═════════════════════════════════════════════════════════════════════════════
# Ragas Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def _build_ragas_dataset(qa_pairs: list[dict]) -> Any:
    """
    Chuyển Q&A pairs thành Ragas Dataset.

    Ragas cần format:
        question, answer, contexts, ground_truth (optional)
    """
    from datasets import Dataset

    ragas_data = {
        "question": [qa["question"] for qa in qa_pairs],
        "answer": [qa["answer"] for qa in qa_pairs],
        "contexts": [qa["contexts"] for qa in qa_pairs],
    }

    return Dataset.from_dict(ragas_data)


def _get_critic_llm():
    """
    Tạo critic LLM (Gemini) để Ragas dùng làm giám khảo.

    Gemini đánh giá chất lượng câu trả lời của Ollama — đây là "cross-eval":
    LLM online (Gemini) chấm điểm LLM offline (Ollama).
    """
    from app.core.settings import settings

    if not settings.gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY is required for evaluation. "
            "Set it in your .env file. "
            "Gemini is used as the critic/judge LLM to evaluate RAG quality."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,  # Deterministic cho đánh giá
    )


def _get_critic_embeddings():
    """
    Tạo critic embeddings cho Ragas (dùng để tính Answer Relevancy).
    Dùng Gemini embeddings nếu có key, fallback về SentenceTransformers.
    """
    from app.core.settings import settings

    if settings.gemini_api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=settings.gemini_api_key,
        )

    # Fallback: SentenceTransformers local
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": settings.embedding_device},
    )


async def evaluate_sessions(
    session_ids: Optional[list[str]] = None,
    limit: int = 50,
) -> dict[str, Any]:
    """
    Đánh giá chất lượng RAG từ lịch sử chat.

    Args:
        session_ids: List session IDs cụ thể, hoặc None lấy gần nhất
        limit: Số sessions tối đa nếu không chỉ định IDs

    Returns:
        dict chứa:
          - scores: dict metrics → float (0-1)
          - details: per-question scores
          - total_questions: số Q&A đã đánh giá
          - error: None hoặc error message
    """
    result = {
        "scores": {},
        "details": [],
        "total_questions": 0,
        "error": None,
    }

    try:
        # 1. Lấy Q&A data
        qa_pairs = await _fetch_qa_pairs(session_ids, limit)

        if not qa_pairs:
            result["error"] = "No Q&A pairs found in database."
            return result

        result["total_questions"] = len(qa_pairs)

        # 2. Build Ragas dataset
        dataset = _build_ragas_dataset(qa_pairs)

        # 3. Setup critic LLM + embeddings
        critic_llm = _get_critic_llm()
        critic_embeddings = _get_critic_embeddings()

        # 4. Chạy Ragas evaluate
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        metrics = [faithfulness, answer_relevancy, context_precision]

        eval_result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=critic_llm,
            embeddings=critic_embeddings,
        )

        # 5. Tổng hợp kết quả
        result["scores"] = {
            metric: round(float(eval_result[metric]), 4)
            for metric in eval_result
            if isinstance(eval_result[metric], (int, float))
        }

        # Per-question details
        if hasattr(eval_result, "to_pandas"):
            df = eval_result.to_pandas()
            for idx, row in df.iterrows():
                detail = {
                    "question": qa_pairs[idx]["question"][:100],
                    "session_id": qa_pairs[idx]["session_id"],
                }
                for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
                    if metric_name in row:
                        detail[metric_name] = round(float(row[metric_name]), 4)
                result["details"].append(detail)

        logger.info("Evaluation complete: %s", result["scores"])

    except ImportError as exc:
        result["error"] = (
            f"Missing dependency: {exc}. "
            "Install: pip install ragas datasets langchain-google-genai"
        )
        logger.error("Evaluation import error: %s", exc)

    except ValueError as exc:
        result["error"] = str(exc)
        logger.error("Evaluation config error: %s", exc)

    except Exception as exc:
        result["error"] = f"Evaluation failed: {exc}"
        logger.error("Evaluation error: %s", exc, exc_info=True)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Quick Test (standalone)
# ═════════════════════════════════════════════════════════════════════════════

async def quick_check() -> dict[str, Any]:
    """
    Quick check xem data có sẵn sàng cho eval không (không chạy Ragas).
    Dùng trong Streamlit để hiển thị preview trước khi evaluate.
    """
    qa_pairs = await _fetch_qa_pairs(limit=10)

    return {
        "total_pairs": len(qa_pairs),
        "sessions": list({qa["session_id"] for qa in qa_pairs}),
        "sample_questions": [qa["question"][:80] for qa in qa_pairs[:5]],
        "has_contexts": sum(
            1 for qa in qa_pairs
            if qa["contexts"] != ["No context retrieved."]
        ),
    }


if __name__ == "__main__":
    import asyncio

    async def _main():
        from app.core.database import init_db, dispose_db
        await init_db()
        info = await quick_check()
        print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        await dispose_db()

    asyncio.run(_main())
