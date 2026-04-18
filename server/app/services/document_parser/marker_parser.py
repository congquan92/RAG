"""
Marker Document Parser
======================

Document parser thay thế dùng Marker (marker-pdf) cho chất lượng trích xuất
math/formula cao (LaTeX qua Surya), nhẹ GPU hơn (~2-4GB VRAM),
và hỗ trợ nhiều định dạng (PDF, DOCX, PPTX, XLSX, EPUB, HTML, images).

Cài đặt: ``pip install marker-pdf[full]``
"""
from __future__ import annotations

import json
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

_MARKER_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".epub"}
_LEGACY_EXTENSIONS = {".txt", ".md"}

# Page separator mặc định Marker dùng khi paginate_output=True
_PAGE_SEPARATOR = "-" * 48


class MarkerDocumentParser(BaseDocumentParser):
    """
    Document parser chạy bằng Marker (marker-pdf).

    Tính năng:
    - Trích xuất math/formula tốt hơn (LaTeX qua Surya)
    - Nhẹ GPU hơn (~2-4GB so với Docling ~18-20GB)
    - Tích hợp sẵn image extraction, table -> markdown, code blocks
    - Chế độ LLM-enhanced tùy chọn cho bảng và phương trình tốt hơn
    """

    parser_name = "marker"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self._converter = None
        self._artifact_dict = None

    @staticmethod
    def supported_extensions() -> set[str]:
        return _MARKER_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # Khởi tạo lazy
    # ------------------------------------------------------------------

    def _get_converter(self):
        """Khởi tạo Marker PdfConverter theo kiểu lazy với model artifacts dùng chung."""
        if self._converter is not None:
            return self._converter

        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser

        # Tải model một lần (~2GB, cache qua nhiều lần gọi)
        if self._artifact_dict is None:
            logger.info("Loading Marker ML models...")
            self._artifact_dict = create_model_dict()

        config = {
            "output_format": "markdown",
            "paginate_output": True,
            "disable_image_extraction": not settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION,
        }

        # Chế độ LLM-enhanced (tốt hơn cho bảng, phương trình, chữ viết tay)
        if settings.NEXUSRAG_MARKER_USE_LLM:
            config["use_llm"] = True

        config_parser = ConfigParser(config)

        self._converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=self._artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        # Gắn LLM service nếu được bật
        if settings.NEXUSRAG_MARKER_USE_LLM:
            try:
                self._converter.llm_service = config_parser.get_llm_service()
            except Exception as e:
                logger.warning(f"Failed to init Marker LLM service: {e}")

        return self._converter

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

        if suffix in _MARKER_EXTENSIONS:
            result = self._parse_with_marker(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_MARKER_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[marker] Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline của Marker
    # ------------------------------------------------------------------

    def _parse_with_marker(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse bằng Marker để trích xuất tài liệu có cấu trúc phong phú."""
        from marker.output import text_from_rendered

        converter = self._get_converter()

        logger.info(f"Marker converting: {file_path}")
        rendered = converter(str(file_path))
        text, ext, marker_images = text_from_rendered(rendered)

        # Trích xuất và lưu images
        images = self._save_marker_images(marker_images, document_id)

        # Tạo caption cho images bằng LLM vision
        if settings.NEXUSRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        # Làm sạch marker số trang của Marker như "{0}", "{1}" trong output
        markdown = re.sub(r"\n\{(\d+)\}", "", text)

        # Cập nhật image references trong markdown bằng served URL
        markdown = self._replace_image_refs_in_markdown(markdown, marker_images, images)

        # Trích xuất tables từ markdown
        tables = self._extract_tables_from_markdown(markdown, document_id)

        # Tạo caption cho tables
        if settings.NEXUSRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)

        # Chèn caption của table
        markdown = self._inject_table_captions(markdown, tables)

        # Đếm số trang từ paginated output
        page_count = self._count_pages(markdown)

        # Chia document thành chunk
        chunks = self._chunk_markdown(
            markdown, document_id, original_filename, images, tables
        )

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
    # Xử lý image
    # ------------------------------------------------------------------

    def _save_marker_images(
        self,
        marker_images: dict,
        document_id: int,
    ) -> list[ExtractedImage]:
        """Lưu images Marker trích xuất (PIL) xuống đĩa và tạo danh sách ExtractedImage."""
        if not marker_images or not settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION:
            return []

        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[ExtractedImage] = []
        count = 0

        for filename, pil_image in marker_images.items():
            if count >= settings.NEXUSRAG_MAX_IMAGES_PER_DOC:
                break

            try:
                image_id = str(uuid.uuid4())
                image_path = images_dir / f"{image_id}.png"

                # Chuyển sang RGB nếu cần (RGBA/P có thể lỗi với một số định dạng)
                if pil_image.mode in ("RGBA", "P", "LA"):
                    pil_image = pil_image.convert("RGB")

                pil_image.save(str(image_path), format="PNG")
                width, height = pil_image.size

                # Thử trích xuất số trang từ filename (vd: "page_3_image_1.png")
                page_no = self._extract_page_from_filename(filename)

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption="",
                    width=width,
                    height=height,
                ))
                count += 1

            except Exception as e:
                logger.warning(f"Failed to save Marker image {filename}: {e}")
                continue

        logger.info(f"Saved {len(images)} Marker images from document {document_id}")
        return images

    @staticmethod
    def _extract_page_from_filename(filename: str) -> int:
        """Thử trích xuất số trang từ filename image của Marker."""
        # Filename Marker có dạng: "{doc_name}_page_{N}_image_{M}.png"
        match = re.search(r"page[_-]?(\d+)", filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _replace_image_refs_in_markdown(
        self,
        markdown: str,
        marker_images: dict,
        images: list[ExtractedImage],
    ) -> str:
        """Thay filename image của Marker trong markdown bằng served URL."""
        if not marker_images or not images:
            return markdown

        # Tạo mapping: original filename -> served URL
        # dict images từ Marker và danh sách images của ta cùng thứ tự
        filenames = list(marker_images.keys())
        for i, img in enumerate(images):
            if i < len(filenames):
                original_name = filenames[i]
                served_url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                # Thay trong markdown: ![alt](original_name) -> ![alt](served_url)
                markdown = markdown.replace(f"]({original_name})", f"]({served_url})")

        return markdown

    # ------------------------------------------------------------------
    # Trích xuất table từ markdown
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tables_from_markdown(
        markdown: str, document_id: int
    ) -> list[ExtractedTable]:
        """Trích xuất các block table từ markdown output."""
        tables: list[ExtractedTable] = []
        lines = markdown.split("\n")
        current_page = 1
        i = 0

        while i < len(lines):
            line = lines[i]

            # Theo dõi số trang từ page separator
            if line.strip() == _PAGE_SEPARATOR:
                current_page += 1
                i += 1
                continue

            # Phát hiện điểm bắt đầu table
            if line.strip().startswith("|"):
                table_lines = [line]
                while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    i += 1
                    table_lines.append(lines[i])

                content_md = "\n".join(table_lines)

                # Đếm số hàng/cột
                data_rows = [
                    l for l in table_lines
                    if l.strip().startswith("|") and "---" not in l
                ]
                num_rows = max(0, len(data_rows) - 1)  # loại trừ hàng header
                num_cols = 0
                if data_rows:
                    num_cols = len([
                        c for c in data_rows[0].split("|") if c.strip()
                    ])

                if num_rows > 0 or num_cols > 0:
                    tables.append(ExtractedTable(
                        table_id=str(uuid.uuid4()),
                        document_id=document_id,
                        page_no=current_page,
                        content_markdown=content_md,
                        num_rows=num_rows,
                        num_cols=num_cols,
                    ))

            i += 1

        if tables:
            logger.info(f"Extracted {len(tables)} tables from Marker markdown")
        return tables

    # ------------------------------------------------------------------
    # Đếm trang
    # ------------------------------------------------------------------

    @staticmethod
    def _count_pages(markdown: str) -> int:
        """Đếm số trang từ paginated markdown output."""
        if not markdown:
            return 0
        # Marker dùng 48 dấu gạch ngang làm page separator
        separators = markdown.count(_PAGE_SEPARATOR)
        return separators + 1  # pages = separators + 1

    # ------------------------------------------------------------------
    # Chia chunk
    # ------------------------------------------------------------------

    def _chunk_markdown(
        self,
        markdown: str,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk Marker markdown output into EnrichedChunks.

        Chiến lược: tách theo page separator trước, rồi tách theo heading trong
        từng trang, tôn trọng giới hạn max_tokens. Mỗi chunk giữ page_no và
        heading context.
        """
        pages = markdown.split(_PAGE_SEPARATOR)
        chunks: list[EnrichedChunk] = []
        chunk_index = 0

        for page_idx, page_text in enumerate(pages):
            page_no = page_idx + 1
            page_text = page_text.strip()
            if not page_text:
                continue

            # Bỏ marker số trang của Marker như "{0}", "{1}", ...
            page_text = re.sub(r"^\{(\d+)\}\s*", "", page_text)

            # Tách trang thành section theo heading
            sections = self._split_by_headings(page_text)

            for heading_path, section_text in sections:
                if not section_text.strip():
                    continue

                # Tách section dài thành sub-chunk
                sub_chunks = self._split_text_by_tokens(
                    section_text,
                    max_tokens=settings.NEXUSRAG_CHUNK_MAX_TOKENS,
                )

                for sub_text in sub_chunks:
                    if not sub_text.strip():
                        continue

                    has_table = "|" in sub_text and "---" in sub_text
                    has_code = "```" in sub_text

                    contextualized = ""
                    if heading_path:
                        contextualized = " > ".join(heading_path) + ": " + sub_text[:100]

                    chunks.append(EnrichedChunk(
                        content=sub_text,
                        chunk_index=chunk_index,
                        source_file=original_filename,
                        document_id=document_id,
                        page_no=page_no,
                        heading_path=heading_path,
                        has_table=has_table,
                        has_code=has_code,
                        contextualized=contextualized,
                    ))
                    chunk_index += 1

        # Enrich chunk bằng image/table refs (logic dùng chung từ base)
        chunks = self._enrich_chunks_with_refs(chunks, images, tables)

        return chunks

    @staticmethod
    def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
        """Tách text thành các section theo markdown heading.

        Trả về danh sách tuple (heading_path, section_text).
        """
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [([], text)]

        sections: list[tuple[list[str], str]] = []
        # Theo dõi thứ bậc heading hiện tại
        heading_stack: list[tuple[int, str]] = []

        # Phần text trước heading đầu tiên
        if matches[0].start() > 0:
            pre_text = text[:matches[0].start()].strip()
            if pre_text:
                sections.append(([], pre_text))

        for i, match in enumerate(matches):
            level = len(match.group(1))  # số lượng #
            title = match.group(2).strip()

            # Cập nhật heading stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_path = [h[1] for h in heading_stack]

            # Lấy text section (từ sau heading này tới heading kế tiếp)
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:
                sections.append((heading_path, section_text))

        return sections

    @staticmethod
    def _split_text_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
        """Tách text thành chunk theo giới hạn token xấp xỉ.

        Dùng xấp xỉ đơn giản theo word (1 token ~= 0.75 words).
        """
        # Xấp xỉ: 1 token ~= 4 chars cho English, 2 chars cho CJK
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return [text]

        # Tách theo đoạn văn trước
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > max_chars:
                if current:
                    chunks.append(current.strip())
                # Xử lý đoạn văn dài hơn max_chars
                if len(para) > max_chars:
                    # Tách theo câu
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current = ""
                    for sent in sentences:
                        if len(current) + len(sent) + 1 > max_chars:
                            if current:
                                chunks.append(current.strip())
                            current = sent
                        else:
                            current = current + " " + sent if current else sent
                else:
                    current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    # Fallback kiểu legacy (TXT/MD) - giống Docling
    # ------------------------------------------------------------------

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD bằng bộ legacy loader."""
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
