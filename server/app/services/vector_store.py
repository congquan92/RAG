"""
Service Vector Store.
Xử lý thao tác ChromaDB để lưu và truy xuất document embeddings.
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional, TYPE_CHECKING
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ChromaDB client toàn cục
_chroma_client: Optional[chromadb.HttpClient] = None


def get_chroma_client() -> chromadb.HttpClient:
    """Lấy hoặc tạo singleton ChromaDB client."""
    global _chroma_client

    if _chroma_client is None:
        logger.info(f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
        _chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            )
        )
        # Kiểm tra kết nối
        _chroma_client.heartbeat()
        logger.info("Connected to ChromaDB successfully")

    return _chroma_client


class VectorStore:
    """
    Service vector store để quản lý document embeddings trong ChromaDB.
    Mỗi knowledge base có collection riêng để tách namespace.
    """

    COLLECTION_PREFIX = "kb_"

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.collection_name = f"{self.COLLECTION_PREFIX}{workspace_id}"
        self._collection = None

    @property
    def collection(self) -> chromadb.Collection:
        """Lấy hoặc tạo collection."""
        if self._collection is None:
            client = get_chroma_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _recreate_collection(self) -> None:
        """Xóa và tạo lại collection (reset reference đã cache)."""
        client = get_chroma_client()
        try:
            client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection {self.collection_name} for dimension migration")
        except Exception:
            pass
        self._collection = None
        # Bắt buộc tạo lại
        _ = self.collection

    def add_documents(
        self,
        ids: Sequence[str],
        embeddings: Sequence[list[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict] | None = None
    ) -> None:
        """
        Thêm document và embeddings vào collection.
        Tự xử lý dimension mismatch: nếu collection được tạo với embedding
        dimension khác, collection sẽ bị xóa và tạo lại.
        """
        if not ids:
            return

        try:
            self.collection.add(
                ids=list(ids),
                embeddings=list(embeddings),
                documents=list(documents),
                metadatas=list(metadatas) if metadatas else None
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg:
                # Dimension mismatch: collection được tạo bằng embedding model cũ
                logger.warning(
                    f"Dimension mismatch in {self.collection_name}: {e}. "
                    f"Recreating collection for new embedding model."
                )
                self._recreate_collection()
                # Thử lại với collection mới
                self.collection.add(
                    ids=list(ids),
                    embeddings=list(embeddings),
                    documents=list(documents),
                    metadatas=list(metadatas) if metadatas else None
                )
            else:
                raise

        logger.info(f"Added {len(ids)} documents to collection {self.collection_name}")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None
    ) -> dict:
        """Truy vấn collection để tìm document tương tự."""
        if include is None:
            include = ["documents", "metadatas", "distances"]

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=include
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg:
                # Query embedding dimension mới trên collection cũ
                logger.warning(
                    f"Dimension mismatch on query in {self.collection_name}: {e}. "
                    f"Collection needs reindexing."
                )
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}
            raise

        # Làm phẳng kết quả cho truy vấn đơn
        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else []
        }

    def delete_by_document_id(self, document_id: int) -> None:
        """Xóa toàn bộ chunk thuộc về một document cụ thể."""
        self.collection.delete(
            where={"document_id": document_id}
        )
        logger.info(f"Deleted chunks for document {document_id} from collection {self.collection_name}")

    def delete_collection(self) -> None:
        """Xóa toàn bộ collection của knowledge base này."""
        client = get_chroma_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {self.collection_name}: {e}")

    def count(self) -> int:
        """Trả về số lượng document trong collection."""
        return self.collection.count()

    def get_by_ids(self, ids: Sequence[str]) -> dict:
        """Lấy document theo danh sách ID."""
        return self.collection.get(
            ids=list(ids),
            include=["documents", "metadatas"]
        )
    
    def get_with_condition(self, where: dict | None = None) -> dict:
        # Nếu where là một dict trống {}, nên chuyển về None
        filter_query = where if where else None
        return self.collection.get(
            where=filter_query,
            include=["documents", "metadatas"]
        )


def get_vector_store(workspace_id: int) -> VectorStore:
    """Factory function để tạo VectorStore cho một knowledge base."""
    return VectorStore(workspace_id)
