"""
Deep Document Parser - Tuong Thich Nguoc
========================================

Module này re-export từ package ``document_parser`` mới.
Các import cũ như::

    from app.services.deep_document_parser import DeepDocumentParser

vẫn hoạt động mà không cần thay đổi.
"""
from app.services.document_parser.docling_parser import DoclingDocumentParser as DeepDocumentParser

__all__ = ["DeepDocumentParser"]
