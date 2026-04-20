"""
Service Reranker
================
Cross-encoder reranker để cải thiện độ chính xác retrieval.

Model mặc định: BAAI/bge-reranker-v2-m3 (multilingual, 100+ languages).
Có thể cấu hình qua NEXUSRAG_RERANKER_MODEL trong settings.

Cách dùng:
    reranker = get_reranker_service()
    ranked = reranker.rerank("user question", ["chunk1", "chunk2", ...], top_k=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Một item đã rerank gồm index gốc và relevance score."""
    index: int          # Vị trí gốc trong danh sách input
    score: float        # Relevance score từ cross-encoder (cao hơn = liên quan hơn)
    text: str           # Nội dung chunk


class RerankerService:
    """
    Service cross-encoder reranker.
    Chấm điểm cặp (query, document) đồng thời qua transformer,
    cho relevance score chính xác hơn nhiều so với bi-encoder cosine similarity.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.NEXUSRAG_RERANKER_MODEL
        self._model = None

    @property
    def model(self):
        """Tải cross-encoder model theo cơ chế lazy."""
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Reranker model loaded: {self.model_name}")
        return self._model

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[RerankResult]:
        """
        Rerank document theo mức độ liên quan với query.

        Args:
            query: Search query của người dùng
            documents: Danh sách văn bản document cần rerank
            top_k: Số lượng kết quả tối đa trả về (None = tất cả)
            min_score: Ngưỡng relevance score tối thiểu (None = không lọc)

        Returns:
            Danh sách RerankResult đã sắp xếp theo score giảm dần,
            đã lọc theo top_k và min_score.
        """
        if not documents:
            return []

        # Tạo cặp (query, document) cho cross-encoder
        pairs = [(query, doc) for doc in documents]

        # Chấm điểm toàn bộ cặp trong một batch
        scores = self.model.predict(pairs, batch_size=32).tolist()

        # Dựng kết quả kèm index gốc
        results = [
            RerankResult(index=i, score=s, text=doc)
            for i, (s, doc) in enumerate(zip(scores, documents))
        ]

        # Sắp xếp score giảm dần (liên quan nhất lên trước)
        results.sort(key=lambda r: r.score, reverse=True)

        # Áp dụng bộ lọc min_score
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Áp dụng giới hạn top_k
        if top_k is not None:
            results = results[:top_k]

        return results


# Singleton instance
_default_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Lấy hoặc tạo reranker service mặc định."""
    global _default_service
    if _default_service is None:
        _default_service = RerankerService()
    return _default_service
