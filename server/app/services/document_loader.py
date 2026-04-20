"""
Service Document Loader.
Xử lý việc load và trích xuất text từ nhiều định dạng tài liệu.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import logging

logger = logging.getLogger(__name__)


class LoadedDocument(NamedTuple):
    """Đại diện cho tài liệu đã load cùng nội dung và metadata."""
    content: str
    source: str
    file_type: str
    page_count: int = 1


def load_txt_file(file_path: Path) -> LoadedDocument:
    """Tải file văn bản thuần (plain text)."""
    try:
        content = file_path.read_text(encoding="utf-8")
        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="txt",
            page_count=1
        )
    except UnicodeDecodeError:
        # Thử với encoding khác
        content = file_path.read_text(encoding="latin-1")
        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="txt",
            page_count=1
        )


def load_pdf_file(file_path: Path) -> LoadedDocument:
    """Tải file PDF và trích xuất nội dung text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages_text = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

        content = "\n\n".join(pages_text)

        return LoadedDocument(
            content=content,
            source=str(file_path),
            file_type="pdf",
            page_count=len(reader.pages)
        )
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise ValueError(f"Failed to load PDF: {e}")


def load_markdown_file(file_path: Path) -> LoadedDocument:
    """Tải file markdown."""
    content = file_path.read_text(encoding="utf-8")
    return LoadedDocument(
        content=content,
        source=str(file_path),
        file_type="md",
        page_count=1
    )


def load_document(file_path: str | Path) -> LoadedDocument:
    """
    Load tài liệu dựa theo file type.

    Định dạng hỗ trợ: .txt, .pdf, .md

    Args:
        file_path: Đường dẫn tới file tài liệu

    Returns:
        LoadedDocument gồm content và metadata

    Raises:
        ValueError: Nếu file type không hỗ trợ hoặc file không thể đọc
    """
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    loaders = {
        ".txt": load_txt_file,
        ".pdf": load_pdf_file,
        ".md": load_markdown_file,
    }

    loader = loaders.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {list(loaders.keys())}")

    return loader(path)


def get_supported_extensions() -> list[str]:
    """Trả về danh sách file extension được hỗ trợ."""
    return [".txt", ".pdf", ".md"]
