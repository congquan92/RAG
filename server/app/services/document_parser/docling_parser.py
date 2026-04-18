"""
Docling Document Parser
=======================

Đóng gói pipeline parse dùng Docling hiện có (trước đây trong
``deep_document_parser.py``) phía sau interface ``BaseDocumentParser``.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.document_parser.base import BaseDocumentParser
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# File extension được Docling xử lý so với legacy
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html"}
_LEGACY_EXTENSIONS = {".txt", ".md"}


class DoclingDocumentParser(BaseDocumentParser):
    """
    Document parser chạy bằng Docling.

    - Chuyển đổi PDF/DOCX/PPTX/HTML qua Docling DocumentConverter
    - Chia chunk bằng HybridChunker (semantic + structural)
    - Trích xuất images và có thể caption qua LLM Vision
    - Fallback về trích xuất text legacy cho TXT/MD
    """

    parser_name = "docling"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self._converter = None

    @staticmethod
    def supported_extensions() -> set[str]:
        return _DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # Bộ chuyển đổi
    # ------------------------------------------------------------------

    def _get_converter(self):
        """Khởi tạo Docling DocumentConverter theo kiểu lazy với image extraction."""
        if self._converter is not None:
            return self._converter

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION
        pipeline_options.images_scale = settings.NEXUSRAG_DOCLING_IMAGES_SCALE
        pipeline_options.do_formula_enrichment = settings.NEXUSRAG_ENABLE_FORMULA_ENRICHMENT

        self._converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return self._converter

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        """Kiểm tra file format có được Docling hỗ trợ (không phải legacy)."""
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

    # ------------------------------------------------------------------
    # Điểm vào parse chính
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _DOCLING_EXTENSIONS:
            result = self._parse_with_docling(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[docling] Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline của Docling
    # ------------------------------------------------------------------

    def _parse_with_docling(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse bằng Docling để trích xuất cấu trúc phong phú."""
        converter = self._get_converter()

        logger.info(f"Docling converting: {file_path}")
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        # Trích xuất images và tạo URL mapping cho markdown references
        images, pic_url_list = self._extract_images_with_urls(doc, document_id)

        # Trích xuất tables
        tables = self._extract_tables(doc, document_id)
        if settings.NEXUSRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)

        # Xuất sang markdown
        markdown = self._export_markdown(doc)

        # Hậu xử lý: thay placeholder image bằng markdown image thật
        markdown = self._inject_image_references(markdown, pic_url_list)

        # Hậu xử lý: chèn table caption vào markdown
        markdown = self._inject_table_captions(markdown, tables)

        # Lấy page count
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)

        # Chunk bằng HybridChunker
        chunks = self._chunk_document(doc, document_id, original_filename, images, tables)

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=markdown,
            page_count=page_count,
            chunks=chunks,
            images=images,
            tables=tables,
            tables_count=len(tables),
        )

    # ------------------------------------------------------------------
    # Chia chunk (Docling HybridChunker)
    # ------------------------------------------------------------------

    def _chunk_document(
        self,
        doc,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk document bằng HybridChunker của Docling với image/table enrichment."""
        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            max_tokens=settings.NEXUSRAG_CHUNK_MAX_TOKENS,
            merge_peers=True,
        )

        # Tạo bảng tra page->images
        page_images: dict[int, list[ExtractedImage]] = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        # Tạo bảng tra page->tables
        page_tables: dict[int, list[ExtractedTable]] = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        chunks = []
        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for i, chunk in enumerate(chunker.chunk(doc)):
            # Trích xuất số trang
            page_no = 0
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "page"):
                    page_no = chunk.meta.page or 0
                elif hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no"):
                                    page_no = prov.page_no or 0
                                    break
                            if page_no > 0:
                                break

            # Trích xuất heading path
            heading_path = []
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading_path = list(chunk.meta.headings)

            # Nhận diện loại nội dung
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            has_table = False
            has_code = False
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        label = getattr(item, "label", "") or ""
                        if "table" in label.lower():
                            has_table = True
                        if "code" in label.lower():
                            has_code = True

            contextualized = ""
            if heading_path:
                contextualized = " > ".join(heading_path) + ": " + chunk_text[:100]

            # -- Enrichment nhận biết ảnh --
            chunk_image_refs: list[str] = []
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = []
                for img_id in chunk_image_refs:
                    img = img_by_id.get(img_id)
                    if img and img.caption:
                        desc_parts.append(
                            f"[Image on page {img.page_no}]: {img.caption}"
                        )
                if desc_parts:
                    enriched_text = chunk_text + "\n\n" + "\n".join(desc_parts)

            # -- Enrichment nhận biết bảng --
            chunk_table_refs: list[str] = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = []
                for tbl_id in chunk_table_refs:
                    tbl = tbl_by_id.get(tbl_id)
                    if tbl and tbl.caption:
                        tbl_parts.append(
                            f"[Table on page {tbl.page_no} ({tbl.num_rows}x{tbl.num_cols})]: {tbl.caption}"
                        )
                if tbl_parts:
                    enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

            chunks.append(EnrichedChunk(
                content=enriched_text,
                chunk_index=i,
                source_file=original_filename,
                document_id=document_id,
                page_no=page_no,
                heading_path=heading_path,
                image_refs=chunk_image_refs,
                table_refs=chunk_table_refs,
                has_table=has_table,
                has_code=has_code,
                contextualized=contextualized,
            ))

        if images:
            logger.info(
                f"Image-aware chunking: {len(assigned_images)}/{len(images)} images "
                f"assigned to {len(chunks)} chunks"
            )
        if tables:
            logger.info(
                f"Table-aware chunking: {len(assigned_tables)}/{len(tables)} tables "
                f"assigned to {len(chunks)} chunks"
            )

        return chunks

    # ------------------------------------------------------------------
    # Xuất markdown
    # ------------------------------------------------------------------

    def _export_markdown(self, doc) -> str:
        """Xuất document sang markdown với page break marker nếu được hỗ trợ."""
        try:
            return doc.export_to_markdown(
                page_break_placeholder="\n\n---\n\n",
            )
        except TypeError:
            return doc.export_to_markdown()

    # ------------------------------------------------------------------
    # Trich xuat image
    # ------------------------------------------------------------------

    def _extract_images_with_urls(
        self,
        doc,
        document_id: int,
    ) -> tuple[list[ExtractedImage], list[tuple[str, str]]]:
        """Trích xuất image và tạo URL mapping cho markdown placeholder."""
        if not settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION:
            return [], []

        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[ExtractedImage] = []
        pic_to_image_idx: list[int] = []
        picture_count = 0

        if not hasattr(doc, "pictures") or not doc.pictures:
            return [], []

        for pic in doc.pictures:
            if picture_count >= settings.NEXUSRAG_MAX_IMAGES_PER_DOC:
                pic_to_image_idx.append(-1)
                continue

            image_id = str(uuid.uuid4())

            page_no = 0
            if hasattr(pic, "prov") and pic.prov:
                for prov in pic.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                image_path = images_dir / f"{image_id}.png"

                if hasattr(pic, "image") and pic.image:
                    pil_image = pic.image.pil_image
                    if pil_image:
                        pil_image.save(str(image_path), format="PNG")
                        width, height = pil_image.size
                    else:
                        pic_to_image_idx.append(-1)
                        continue
                else:
                    pic_to_image_idx.append(-1)
                    continue

                caption = ""
                if hasattr(pic, "caption_text"):
                    caption = pic.caption_text(doc) if callable(pic.caption_text) else str(pic.caption_text or "")
                elif hasattr(pic, "text"):
                    caption = str(pic.text or "")

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption=caption,
                    width=width,
                    height=height,
                ))
                pic_to_image_idx.append(len(images) - 1)
                picture_count += 1

            except Exception as e:
                logger.warning(f"Failed to extract image from document {document_id}: {e}")
                pic_to_image_idx.append(-1)
                continue

        logger.info(f"Extracted {len(images)} images from document {document_id}")

        if settings.NEXUSRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        pic_url_list: list[tuple[str, str]] = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                pic_url_list.append((img.caption, url))
            else:
                pic_url_list.append(("", ""))

        return images, pic_url_list

    def _inject_image_references(
        self, markdown: str, pic_url_list: list[tuple[str, str]]
    ) -> str:
        """Thay placeholder <!-- image --> bằng markdown ![caption](url)."""
        placeholder_count = len(re.findall(r"<!--\s*image\s*-->", markdown))

        if not pic_url_list:
            if placeholder_count > 0:
                logger.warning(
                    f"Markdown has {placeholder_count} image placeholders but "
                    f"pic_url_list is empty — images will NOT be injected"
                )
            return markdown

        logger.info(
            f"Injecting {len(pic_url_list)} image URLs into "
            f"{placeholder_count} placeholders"
        )

        injected = 0
        pic_iter = iter(pic_url_list)

        def replacer(match):
            nonlocal injected
            try:
                caption, url = next(pic_iter)
                if url:
                    safe_caption = caption.replace("[", "").replace("]", "")
                    safe_caption = " ".join(safe_caption.split())
                    injected += 1
                    return f"\n![{safe_caption}]({url})\n"
                return ""
            except StopIteration:
                return ""

        result = re.sub(r'<!--\s*image\s*-->', replacer, markdown)
        logger.info(f"Injected {injected}/{placeholder_count} image references")
        return result

    # ------------------------------------------------------------------
    # Trich xuat table (Docling-specific)
    # ------------------------------------------------------------------

    def _extract_tables(self, doc, document_id: int) -> list[ExtractedTable]:
        """Trích xuất table từ tài liệu Docling."""
        if not hasattr(doc, "tables") or not doc.tables:
            return []

        tables: list[ExtractedTable] = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())

            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                content_md = table.export_to_markdown(doc)
            except Exception:
                content_md = ""

            if not content_md.strip():
                continue

            num_rows = 0
            num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0

            tables.append(ExtractedTable(
                table_id=table_id,
                document_id=document_id,
                page_no=page_no,
                content_markdown=content_md,
                num_rows=num_rows,
                num_cols=num_cols,
            ))

        logger.info(f"Extracted {len(tables)} tables from document {document_id}")
        return tables

    # ------------------------------------------------------------------
    # Fallback kiểu legacy (TXT/MD)
    # ------------------------------------------------------------------

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD bằng legacy loader."""
        from app.services.document_loader import load_document
        from app.services.chunker import DocumentChunker

        loaded = load_document(str(file_path))
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text_chunks = chunker.split_text(
            text=loaded.content,
            source=original_filename,
            extra_metadata={"document_id": document_id, "file_type": loaded.file_type},
        )

        chunks = [
            EnrichedChunk(
                content=tc.content,
                chunk_index=tc.chunk_index,
                source_file=original_filename,
                document_id=document_id,
                page_no=0,
            )
            for tc in text_chunks
        ]

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=loaded.content,
            page_count=loaded.page_count,
            chunks=chunks,
            images=[],
            tables_count=0,
        )
