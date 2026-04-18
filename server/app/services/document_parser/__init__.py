"""
Factory function để tạo document parser dựa trên config.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.services.document_parser.base import BaseDocumentParser


def get_document_parser(
    workspace_id: int,
    output_dir: Optional[Path] = None,
) -> BaseDocumentParser:
    """Tạo document parser dựa theo config ``NEXUSRAG_DOCUMENT_PARSER``."""
    from app.core.config import settings

    provider = settings.NEXUSRAG_DOCUMENT_PARSER.lower()

    if provider == "marker":
        from app.services.document_parser.marker_parser import MarkerDocumentParser

        return MarkerDocumentParser(workspace_id, output_dir)

    # Mặc định: docling
    from app.services.document_parser.docling_parser import DoclingDocumentParser

    return DoclingDocumentParser(workspace_id, output_dir)


__all__ = [
    "get_document_parser",
    "BaseDocumentParser",
]
