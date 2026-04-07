"""
RAG System Prompts — Ép LLM trả lời đúng format, có trích dẫn, không bịa đặt.

Chứa các prompt template dùng cho:
  - QA with citations (trả lời dựa trên context)
  - Fallback khi không đủ thông tin
  - Condensed question (rút gọn follow-up thành standalone question)
"""

from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
# System Prompt chính — Trả lời có trích dẫn, chống hallucination
# ═════════════════════════════════════════════════════════════════════════════

QA_SYSTEM_PROMPT = """You are an expert research assistant for a document knowledge base.
Your job is to answer questions STRICTLY based on the provided context documents.

## ABSOLUTE RULES — VIOLATIONS ARE UNACCEPTABLE:

1. **ONLY use information from the CONTEXT below.** If the context does not contain enough information to answer, say so clearly. NEVER make up facts, data, or quotes.
2. **Always cite your sources.** After each claim, reference the source document in [Source: filename] format.
3. **If the answer requires multiple sources**, combine them logically and cite each one separately.
4. **If the context is empty or irrelevant to the question**, respond with: "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu tài liệu. Vui lòng tải lên tài liệu phù hợp hoặc đặt lại câu hỏi."
5. **Respond in the SAME LANGUAGE as the user's question.** If the user asks in Vietnamese, answer in Vietnamese. If in English, answer in English.

## RESPONSE FORMAT:

Provide a clear, well-structured answer:
- Use bullet points or numbered lists for multi-part answers
- Bold key terms for readability
- Keep explanations concise but thorough
- End with source citations

## CONTEXT DOCUMENTS:
{context}
"""

# ═════════════════════════════════════════════════════════════════════════════
# Human message template — Gửi kèm query
# ═════════════════════════════════════════════════════════════════════════════

QA_HUMAN_TEMPLATE = """Based on the context documents provided in the system message, please answer the following question:

**Question:** {question}

Remember: ONLY use information from the provided context. Cite your sources with [Source: filename] format."""

# ═════════════════════════════════════════════════════════════════════════════
# Condensed Question — Rút gọn follow-up thành standalone question
# Dùng khi user hỏi follow-up mà cần context từ lịch sử chat
# ═════════════════════════════════════════════════════════════════════════════

CONDENSE_QUESTION_PROMPT = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all necessary context.

## Chat History:
{chat_history}

## Follow-up Question:
{question}

## Standalone Question:"""

# ═════════════════════════════════════════════════════════════════════════════
# Prompt khi không tìm thấy context phù hợp
# ═════════════════════════════════════════════════════════════════════════════

NO_CONTEXT_RESPONSE = (
    "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu tài liệu. "
    "Vui lòng tải lên tài liệu phù hợp hoặc đặt lại câu hỏi."
)

# ═════════════════════════════════════════════════════════════════════════════
# Prompt đánh giá chất lượng (dùng cho Ragas evaluation ở Admin UI)
# ═════════════════════════════════════════════════════════════════════════════

EVAL_FAITHFULNESS_PROMPT = """Given the following context and answer, determine if the answer is faithful to the context.
Rate on a scale of 0-1 where:
- 0 = completely hallucinated, no basis in context
- 1 = every claim is directly supported by context

Context: {context}
Answer: {answer}
Score (0-1):"""

EVAL_RELEVANCY_PROMPT = """Given the following question and answer, determine if the answer is relevant to the question.
Rate on a scale of 0-1 where:
- 0 = completely irrelevant
- 1 = directly and fully addresses the question

Question: {question}
Answer: {answer}
Score (0-1):"""
