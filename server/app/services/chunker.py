"""
Service Text Chunking.
Xử lý chia tài liệu thành chunk nhỏ để embedding và retrieval.
"""
from __future__ import annotations

from typing import NamedTuple, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunk(NamedTuple):
    """Đại diện cho một text chunk kèm metadata."""
    content: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: dict


class DocumentChunker:
    """
    Chia tài liệu thành chunk bằng phương pháp recursive theo ký tự.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None
    ):
        """
        Khởi tạo chunker.

        Args:
            chunk_size: Số ký tự tối đa cho mỗi chunk
            chunk_overlap: Số ký tự chồng lấp giữa các chunk
            separators: Danh sách separator tùy biến để chia chunk (optional)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=self.separators,
        )

    def split_text(
        self,
        text: str,
        source: str = "",
        extra_metadata: dict | None = None
    ) -> list[TextChunk]:
        """
        Chia text thành chunk kèm metadata.

        Args:
            text: Nội dung text cần tách
            source: Định danh nguồn (vd: filename)
            extra_metadata: Metadata bổ sung cho mỗi chunk

        Returns:
            Danh sách TextChunk gồm content và metadata
        """
        if not text.strip():
            return []

        # Dùng splitter của LangChain
        chunks = self._splitter.split_text(text)

        result = []
        current_pos = 0

        for i, chunk_content in enumerate(chunks):
            # Tìm vị trí thực tế trong text gốc
            # Đây là giá trị xấp xỉ do có xử lý overlap
            start_pos = text.find(chunk_content[:50], current_pos)
            if start_pos == -1:
                start_pos = current_pos

            end_pos = start_pos + len(chunk_content)

            metadata = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(extra_metadata or {})
            }

            result.append(TextChunk(
                content=chunk_content,
                chunk_index=i,
                char_start=start_pos,
                char_end=end_pos,
                metadata=metadata
            ))

            # Cập nhật vị trí cho lần tìm tiếp theo (tính cả overlap)
            current_pos = max(start_pos + len(chunk_content) - self.chunk_overlap, current_pos + 1)

        return result

    def estimate_chunk_count(self, text: str) -> int:
        """
        Ước lượng số lượng chunk mà không cần tách thực tế.

        Args:
            text: Nội dung text cần ước lượng

        Returns:
            Số lượng chunk ước lượng
        """
        if not text:
            return 0

        text_length = len(text)
        effective_chunk = self.chunk_size - self.chunk_overlap

        if effective_chunk <= 0:
            return 1

        return max(1, (text_length + effective_chunk - 1) // effective_chunk)


# Chunker mặc định
default_chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)


def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[TextChunk]:
    """
    Hàm tiện ích để chia chunk với cấu hình mặc định hoặc tùy biến.

    Args:
        text: Text cần chia chunk
        source: Định danh nguồn
        chunk_size: Số ký tự tối đa cho mỗi chunk
        chunk_overlap: Số ký tự chồng lấp giữa các chunk

    Returns:
        Danh sách đối tượng TextChunk
    """
    if chunk_size == 500 and chunk_overlap == 50:
        return default_chunker.split_text(text, source)

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.split_text(text, source)
