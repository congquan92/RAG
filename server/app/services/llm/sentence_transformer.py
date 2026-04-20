"""
Sentence-Transformers Embedding Provider
========================================
Triển khai EmbeddingProvider bằng sentence-transformers cho KG embedding
local hoàn toàn, open-source, không cần API ngoài.

Cách dùng::

    KG_EMBEDDING_PROVIDER=sentence_transformers
    KG_EMBEDDING_MODEL=BAAI/bge-m3
"""
from __future__ import annotations

import logging

import numpy as np

from app.services.llm.base import EmbeddingProvider

logger = logging.getLogger(__name__)

# Bảng tra dimension để có thể báo dimension trước khi load model.
_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "intfloat/multilingual-e5-large-instruct": 1024,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "jinaai/jina-embeddings-v2-base-en": 768,
}


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider dùng bất kỳ model sentence-transformers nào."""

    _BATCH_SIZE = 64

    def __init__(self, model: str = "BAAI/bge-m3"):
        self._model_name = model
        self._model = None
        self._dimension: int | None = _KNOWN_DIMS.get(model)

    # -- lazy load để tránh import ngay lúc startup --

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers KG embedding model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "KG embedding model loaded: %s (dim=%d)",
                self._model_name,
                self._dimension,
            )
        return self._model

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), self._BATCH_SIZE):
            batch = texts[i : i + self._BATCH_SIZE]
            emb = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=self._BATCH_SIZE,
            )
            all_embeddings.append(emb)
        return np.vstack(all_embeddings).astype(np.float32)

    def get_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Bắt buộc load model để phát hiện dimension
        return self.model.get_sentence_embedding_dimension()
