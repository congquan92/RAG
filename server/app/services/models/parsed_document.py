"""
Data Models cua NexusRAG
========================

Dataclass cho pipeline NexusRAG: document parsing, enriched chunks,
citation và retrieval result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedImage:
    """Một image được trích xuất từ document bởi Docling."""
    image_id: str
    document_id: int
    page_no: int
    file_path: str
    caption: str = ""
    width: int = 0
    height: int = 0
    bbox: Optional[tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    mime_type: str = "image/png"


@dataclass
class ExtractedTable:
    """Một table được trích xuất từ document bởi Docling."""
    table_id: str
    document_id: int
    page_no: int
    content_markdown: str  # table.export_to_markdown(doc)
    caption: str = ""      # mô tả được sinh bởi LLM
    num_rows: int = 0
    num_cols: int = 0


@dataclass
class EnrichedChunk:
    """Một document chunk được enrich bằng metadata cấu trúc."""
    content: str
    chunk_index: int
    source_file: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)  # image_id lân cận
    table_refs: list[str] = field(default_factory=list)  # table_id lân cận
    has_table: bool = False
    has_code: bool = False
    contextualized: str = ""  # heading_path được nối lại cho context


@dataclass
class ParsedDocument:
    """Kết quả parse một document bằng Docling."""
    document_id: int
    original_filename: str
    markdown: str
    page_count: int
    chunks: list[EnrichedChunk] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    tables_count: int = 0


@dataclass
class Citation:
    """Source citation trỏ tới vị trí cụ thể trong một document."""
    source_file: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = field(default_factory=list)

    def format(self) -> str:
        """Định dạng citation thành chuỗi dễ đọc cho người dùng."""
        parts = [self.source_file]
        if self.page_no > 0:
            parts.append(f"p.{self.page_no}")
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        return " | ".join(parts)


@dataclass
class DeepRetrievalResult:
    """Kết quả deep RAG query kèm citation và KG insights."""
    chunks: list[EnrichedChunk]
    citations: list[Citation]
    context: str  # context đã ghép cho LLM
    query: str
    mode: str = "hybrid"
    knowledge_graph_summary: str = ""
    image_refs: list[ExtractedImage] = field(default_factory=list)
    table_refs: list[ExtractedTable] = field(default_factory=list)
