"""
LLM & Embeddings Factory — Hot-Swap provider selection via .env.

Provides two factory functions:
  - get_llm()        → returns a LangChain-compatible Chat model
  - get_embeddings() → returns a LangChain-compatible Embeddings model

Business logic calls these factories instead of constructing models directly.
Switching between Ollama (offline) and Gemini (cloud) requires only changing
LLM_PROVIDER / EMBEDDING_PROVIDER in .env — zero code changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from app.core.settings import Settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# LLM Factory
# ═════════════════════════════════════════════════════════════════════════════

def get_llm_with_overrides(
    settings: Settings,
    provider_override: Literal["ollama", "gemini"] | None = None,
    model_override: str | None = None,
    gemini_api_key_override: str | None = None,
) -> Any:
    """
    Khởi tạo Chat LLM với runtime overrides (không làm thay đổi settings global).

    Dùng cho các trường hợp per-request như GraphRAG Gemini yêu cầu API key
    từ người dùng trên UI.
    """
    provider = provider_override or settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        model_name = (model_override or settings.ollama_model).strip()
        logger.info(
            "Initializing Ollama LLM: model=%s, base_url=%s",
            model_name,
            settings.ollama_base_url,
        )
        return ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = (model_override or settings.gemini_model).strip()
        api_key = (gemini_api_key_override or settings.gemini_api_key).strip()
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM provider is set to 'gemini'. "
                "Set it in your .env file or pass it from UI."
            )

        logger.info("Initializing Gemini LLM: model=%s", model_name)
        # Keep retries low to avoid hammering Gemini during transient overload.
        try:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                max_retries=1,
            )
        except TypeError:
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
            )

    raise ValueError(
        f"Unsupported LLM provider='{provider}'. "
        "Allowed values: 'ollama', 'gemini'."
    )

def get_llm(settings: Settings) -> Any:
    """
    Khởi tạo Chat LLM dựa trên LLM_PROVIDER trong settings.

    Returns:
        BaseChatModel — LangChain chat model, sẵn sàng gọi .invoke() / .astream()
    Raises:
        ValueError — nếu provider không hợp lệ hoặc thiếu config
    """
    return get_llm_with_overrides(settings)


# ═════════════════════════════════════════════════════════════════════════════
# Embeddings Factory
# ═════════════════════════════════════════════════════════════════════════════

def get_embeddings(settings: Settings) -> Any:
    """
    Khởi tạo Embedding model dựa trên EMBEDDING_PROVIDER trong settings.

    Returns:
        Embeddings — LangChain embeddings, sẵn sàng gọi .embed_documents()
    Raises:
        ValueError — nếu provider không hợp lệ hoặc thiếu config
    """
    provider = settings.embedding_provider

    if provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        logger.info(
            "Initializing SentenceTransformers: model=%s, device=%s",
            settings.embedding_model,
            settings.embedding_device,
        )
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        logger.info(
            "Initializing Ollama Embeddings: model=%s, base_url=%s",
            settings.embedding_model,
            settings.ollama_base_url,
        )
        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when EMBEDDING_PROVIDER='gemini'. "
                "Set it in your .env file."
            )
        logger.info(
            "Initializing Gemini Embeddings: model=%s",
            settings.embedding_model,
        )
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{provider}'. "
        "Allowed values: 'sentence-transformers', 'ollama', 'gemini'."
    )
