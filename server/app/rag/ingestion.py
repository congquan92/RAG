"""
Ingestion Pipeline — Parse documents và chia chunks.

Dual-Track Processing:
  Track 1: PyMuPDF  — Fast text extraction (CPU-only, tốc độ cao)
  Track 2: docling  — Advanced table/layout extraction (DOCLING_DEVICE)

Chunking: Tiktoken-based splitter đảm bảo mỗi chunk không vượt token limit.

Flow: file → detect MIME → extract text → split chunks → return chunks
Lưu ChromaDB + LightRAG sẽ được gọi ở service layer (Vòng 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.settings import settings

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentChunk:
    """Một chunk text đã trích xuất từ file."""

    text: str
    metadata: dict = field(default_factory=dict)
    # metadata chứa: source_file, page_number, chunk_index, extraction_method


@dataclass
class ExtractionResult:
    """Kết quả trích xuất từ 1 file."""

    full_text: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    page_count: int = 0
    extraction_method: str = "unknown"
    error: Optional[str] = None


# ═════════════════════════════════════════════════════════════════════════════
# MIME type detection
# ═════════════════════════════════════════════════════════════════════════════

def detect_mime_type(file_path: str | Path) -> str:
    """
    Phát hiện MIME type bằng python-magic (dựa vào file content, không dựa extension).
    Fallback sang extension-based nếu magic không khả dụng.
    """
    file_path = Path(file_path)
    try:
        import magic

        mime = magic.from_file(str(file_path), mime=True)
        logger.debug("MIME detected by magic: %s → %s", file_path.name, mime)
        return mime
    except ImportError:
        logger.warning("python-magic not available, falling back to extension-based detection")
    except Exception as exc:
        logger.warning("magic detection failed for %s: %s", file_path.name, exc)

    # Fallback: extension mapping
    ext_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }
    suffix = file_path.suffix.lower()
    return ext_map.get(suffix, "application/octet-stream")


# ═════════════════════════════════════════════════════════════════════════════
# Track 1: PyMuPDF — Fast text extraction (CPU)
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_pymupdf(file_path: str | Path) -> ExtractionResult:
    """
    Trích xuất text từ PDF bằng PyMuPDF.
    Tốc độ rất nhanh, chạy hoàn toàn trên CPU. Phù hợp cho PDF có text thuần.
    """
    file_path = Path(file_path)
    try:
        import pymupdf  # PyMuPDF >= 1.24 dùng import pymupdf

        doc = pymupdf.open(str(file_path))
        pages_text: list[str] = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)

        doc.close()

        full_text = "\n\n".join(pages_text)
        logger.info(
            "PyMuPDF extracted %d pages, %d chars from %s",
            len(pages_text), len(full_text), file_path.name,
        )
        return ExtractionResult(
            full_text=full_text,
            page_count=len(pages_text),
            extraction_method="pymupdf",
        )
    except ImportError:
        logger.error("pymupdf not installed. Install: pip install pymupdf")
        return ExtractionResult(
            full_text="", extraction_method="pymupdf",
            error="pymupdf not installed",
        )
    except Exception as exc:
        logger.error("PyMuPDF extraction failed for %s: %s", file_path.name, exc)
        return ExtractionResult(
            full_text="", extraction_method="pymupdf",
            error=str(exc),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Track 2: docling — Advanced table/layout extraction
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_docling(file_path: str | Path) -> ExtractionResult:
    """
    Trích xuất text từ file phức tạp (bảng biểu, layout) bằng docling.
    Chạy theo DOCLING_DEVICE (cpu/cuda) từ settings.
    Chậm hơn PyMuPDF nhưng chính xác hơn cho tài liệu có tables.
    """
    file_path = Path(file_path)
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(file_path))
        full_text = result.document.export_to_markdown()

        logger.info(
            "Docling extracted %d chars from %s (device=%s)",
            len(full_text), file_path.name, settings.docling_device,
        )
        return ExtractionResult(
            full_text=full_text,
            page_count=0,  # docling không track page count rõ ràng
            extraction_method="docling",
        )
    except ImportError:
        logger.error("docling not installed. Install: pip install docling")
        return ExtractionResult(
            full_text="", extraction_method="docling",
            error="docling not installed",
        )
    except Exception as exc:
        logger.error("Docling extraction failed for %s: %s", file_path.name, exc)
        return ExtractionResult(
            full_text="", extraction_method="docling",
            error=str(exc),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Text extraction: plain text / markdown files
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_plain(file_path: str | Path) -> ExtractionResult:
    """Đọc trực tiếp nội dung file text/markdown."""
    file_path = Path(file_path)
    try:
        text = file_path.read_text(encoding="utf-8")
        logger.info("Plain text extracted: %d chars from %s", len(text), file_path.name)
        return ExtractionResult(
            full_text=text,
            page_count=1,
            extraction_method="plain_text",
        )
    except Exception as exc:
        logger.error("Plain text extraction failed for %s: %s", file_path.name, exc)
        return ExtractionResult(
            full_text="", extraction_method="plain_text",
            error=str(exc),
        )


# ═════════════════════════════════════════════════════════════════════════════
# Smart extractor — chọn track phù hợp dựa trên MIME type
# ═════════════════════════════════════════════════════════════════════════════

def extract_text(file_path: str | Path, use_docling: bool = False) -> ExtractionResult:
    """
    Auto-detect loại file và chọn extraction method phù hợp.

    Args:
        file_path: Đường dẫn tới file cần trích xuất
        use_docling: True để dùng docling (cho file có bảng biểu phức tạp)
                     False để dùng PyMuPDF (nhanh hơn, cho PDF text thuần)
    """
    file_path = Path(file_path)
    mime = detect_mime_type(file_path)

    if mime == "application/pdf":
        if use_docling:
            return extract_text_docling(file_path)
        return extract_text_pymupdf(file_path)

    if mime in ("text/plain", "text/markdown", "text/csv"):
        return extract_text_plain(file_path)

    # File phức tạp (docx, xlsx, pptx) → dùng docling
    if mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/msword",
    ):
        return extract_text_docling(file_path)

    logger.warning("Unsupported MIME type %s for %s, attempting plain text", mime, file_path.name)
    return extract_text_plain(file_path)


# ═════════════════════════════════════════════════════════════════════════════
# Tiktoken-based chunking
# ═════════════════════════════════════════════════════════════════════════════

def _tiktoken_length(text: str, encoding_name: str = "cl100k_base") -> int:
    """Đếm số tokens bằng Tiktoken (cùng tokenizer với GPT-4/embedding models)."""
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def split_into_chunks(
    text: str,
    source_file: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[DocumentChunk]:
    """
    Chia text thành chunks sử dụng RecursiveCharacterTextSplitter + Tiktoken.

    Args:
        text: Full text cần chia
        source_file: Tên file nguồn (lưu vào metadata)
        chunk_size: Số tokens tối đa mỗi chunk (default từ settings)
        chunk_overlap: Số tokens overlap giữa các chunk (default từ settings)

    Returns:
        List[DocumentChunk] — mỗi chunk kèm metadata
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking, returning empty list")
        return []

    _chunk_size = chunk_size or settings.chunk_size
    _chunk_overlap = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=_chunk_size,
        chunk_overlap=_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(text)

    chunks: list[DocumentChunk] = []
    for idx, chunk_text in enumerate(raw_chunks):
        chunks.append(
            DocumentChunk(
                text=chunk_text,
                metadata={
                    "source_file": source_file,
                    "chunk_index": idx,
                    "token_count": _tiktoken_length(chunk_text),
                },
            )
        )

    logger.info(
        "Split into %d chunks (size=%d, overlap=%d) from '%s'",
        len(chunks), _chunk_size, _chunk_overlap, source_file,
    )
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# Full ingestion pipeline (extract + chunk)
# ═════════════════════════════════════════════════════════════════════════════

def ingest_file(
    file_path: str | Path,
    use_docling: bool = False,
) -> ExtractionResult:
    """
    Full ingestion pipeline cho 1 file:
      1. Detect MIME type
      2. Extract text (PyMuPDF hoặc docling)
      3. Split thành chunks (Tiktoken)
      4. Trả về ExtractionResult kèm chunks

    Args:
        file_path: Đường dẫn file
        use_docling: Dùng docling cho PDF (chậm nhưng chính xác hơn cho bảng biểu)

    Returns:
        ExtractionResult — full_text + chunks + metadata
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return ExtractionResult(
            full_text="",
            error=f"File not found: {file_path}",
        )

    # Step 1–2: Extract text
    result = extract_text(file_path, use_docling=use_docling)

    if result.error or not result.full_text.strip():
        logger.warning("No text extracted from %s: %s", file_path.name, result.error)
        return result

    # Step 3: Chunk text
    result.chunks = split_into_chunks(
        text=result.full_text,
        source_file=file_path.name,
    )

    logger.info(
        "Ingestion complete: %s → %d chunks (method=%s)",
        file_path.name, len(result.chunks), result.extraction_method,
    )
    return result
