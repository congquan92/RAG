from app.core.config import settings

from typing import Optional
import logging

logger = logging.getLogger(__name__)

class RewriterService:
    """
    Đánh giá và yêu cầu LLM cải thiện lại câu hỏi
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NEXUSRAG_RERANKER_MODEL
        self._model = None

    def rewrite(
        self,
        question: str,
    ) -> str:
        
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        provider = get_llm_provider()
        
        system_prompt = (
            "You are an Information Retrieval expert. Your task is to optimize user queries for Vector Search accuracy.\n\n"
            "STRICT ADHERENCE TO THESE RULES IS MANDATORY:\n"
            "1. DO NOT REWRITE if the query is already a clear, complete, and natural question (e.g., 'X là gì?', 'What is X?', 'Làm thế nào để X?'). If the query makes sense, RETURN THE ORIGINAL EXACTLY.\n"
            "2. NEVER invert word order or use unnatural grammar. The output must sound like a natural human question in the original language.\n"
            "3. ONLY rewrite if the query is: a) Only keywords, b) Full of typos, c) Too short to have a specific meaning.\n"
            "4. MAINTAIN the original language. If the user asks in Vietnamese, the output must be Vietnamese. If English, stay in English.\n"
            "5. Do not change the core intent. Do not add unrelated information.\n"
            "6. OUTPUT ONLY the query. No preamble, no quotes, no labels like 'Optimized Query:'.\n\n"
            "EXAMPLES:\n"
            "- Input: 'Trái đất hình gì?' -> Output: 'Trái đất hình gì?' (Clear - Keep as-is)\n"
            "- Input: 'how to cook' -> Output: 'How to cook a basic meal for beginners?' (Too short - Expand)\n"
            "- Input: 'quy trình làm pánh' -> Output: 'Quy trình làm bánh như thế nào?' (Typo & Vague - Rewrite)\n"
            "- Input: 'AI là gì?' -> Output: 'AI là gì?' (Clear - Keep as-is)"
        )

        messages = [
            LLMMessage(role="user", content=question)
        ]

        try:
            result = provider.complete(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=200
            )

            if hasattr(result, "content"):
                return result.content.strip()
            return result.strip()

        except Exception as e:
            logger.warning(f"Query Rewriting failed: {e}")
            return question


# Singleton instance
_default_service: Optional[RewriterService] = None


def get_rewriter_service() -> RewriterService:
    global _default_service
    if _default_service is None:
        _default_service = RewriterService()
    return _default_service