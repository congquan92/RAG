"""RAG utility endpoints for client capabilities and source feedback."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_settings
from app.core.settings import Settings
from app.schemas.rag_schema import (
    LLMCapabilitiesResponse,
    SourceRatingRequest,
    SourceRatingResponse,
)
from app.services import chat_service

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get(
    "/capabilities",
    response_model=LLMCapabilitiesResponse,
    summary="Report current LLM capability flags",
)
async def get_capabilities(settings: Settings = Depends(get_settings)):
    provider = settings.llm_provider
    model = settings.ollama_model if provider == "ollama" else settings.gemini_model

    # Conservative defaults so frontend toggles only features we can guarantee.
    supports_thinking = provider == "gemini"
    supports_vision = provider == "gemini"

    return LLMCapabilitiesResponse(
        provider=provider,
        model=model,
        supports_thinking=supports_thinking,
        supports_vision=supports_vision,
        thinking_default=False,
    )


@router.post(
    "/chat/{assistant_message_id}/rate",
    response_model=SourceRatingResponse,
    summary="Save user relevance feedback for one retrieved source",
)
async def rate_source(
    assistant_message_id: str,
    body: SourceRatingRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        rating = await chat_service.upsert_source_rating(
            db=db,
            assistant_message_id=assistant_message_id,
            source_index=body.source_index,
            rating=body.rating,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SourceRatingResponse(
        id=rating.id,
        assistant_message_id=rating.assistant_message_id,
        source_index=rating.source_index,
        rating=rating.rating,
        updated_at=rating.updated_at,
    )
