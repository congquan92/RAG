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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from app.core.settings import Settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# LLM Factory
# ═════════════════════════════════════════════════════════════════════════════

def get_llm(settings: Settings) -> Any:
    """
    Khởi tạo Chat LLM dựa trên LLM_PROVIDER trong settings.

    Returns:
        BaseChatModel — LangChain chat model, sẵn sàng gọi .invoke() / .astream()
    Raises:
        ValueError — nếu provider không hợp lệ hoặc thiếu config
    """
    provider = settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info(
            "Initializing Ollama LLM: model=%s, base_url=%s",
            settings.ollama_model,
            settings.ollama_base_url,
        )
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER='gemini'. "
                "Set it in your .env file."
            )
        logger.info("Initializing Gemini LLM: model=%s", settings.gemini_model)
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
        )

    # Unreachable nếu Settings validator hoạt động, nhưng phòng thủ sâu
    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Allowed values: 'ollama', 'gemini'."
    )


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
