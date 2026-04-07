"""Schemas for RAG utility endpoints (capabilities and source feedback)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LLMCapabilitiesResponse(BaseModel):
    provider: str
    model: str
    supports_thinking: bool
    supports_vision: bool
    thinking_default: bool


class SourceRatingRequest(BaseModel):
    source_index: str = Field(..., min_length=1, max_length=64)
    rating: Literal["relevant", "partial", "not_relevant"]


class SourceRatingResponse(BaseModel):
    id: str
    assistant_message_id: str
    source_index: str
    rating: str
    updated_at: datetime
