"""Các extractor chuyên dụng cho workflow KG chạy offline."""

from app.services.extractor.specialized_kg_extractor import (
    SpecializedKGExtractor,
    get_specialized_kg_extractor,
)

__all__ = ["SpecializedKGExtractor", "get_specialized_kg_extractor"]
