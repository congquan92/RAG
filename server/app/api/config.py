"""
Config status endpoint — cung cấp thông tin LLM/embedding provider đang active cho frontend.
"""
from fastapi import APIRouter

from app.core.config import settings
from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/status")
async def get_config_status():
    """Trả về provider đang active và tên model để hiển thị trên UI."""
    llm_provider = settings.LLM_PROVIDER.lower()

    if llm_provider == "ollama":
        llm_model = settings.OLLAMA_MODEL
    else:
        llm_model = settings.LLM_MODEL_FAST

    kg_provider = settings.KG_EMBEDDING_PROVIDER.lower()
    kg_model = settings.KG_EMBEDDING_MODEL

    return {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "kg_embedding_provider": kg_provider,
        "kg_embedding_model": kg_model,
        "kg_embedding_dimension": settings.KG_EMBEDDING_DIMENSION,
        "nexusrag_embedding_model": settings.NEXUSRAG_EMBEDDING_MODEL,
        "nexusrag_reranker_model": settings.NEXUSRAG_RERANKER_MODEL,
    }


@router.get("/chat-default-prompt")
async def get_chat_default_prompt():
    """Trả về default system prompt của backend khi workspace prompt đang rỗng."""
    return {
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
    }
