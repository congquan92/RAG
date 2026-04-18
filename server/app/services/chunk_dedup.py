"""
Pipeline Deduplication truoc ingest
===================================

Lọc nhiễu và loại chunk trùng/gần trùng TRƯỚC khi embedding,
giảm ô nhiễm vector space và cải thiện chất lượng retrieval.

Pipeline 3 giai đoạn:
    1. Noise filter  - loại boilerplate header/footer, legal text, chunk quá nhỏ
    2. Exact dedup   - hash SHA-256 nội dung để loại chunk giống hệt
    3. Near dedup    - shingling n-gram ký tự + Jaccard similarity cho khớp mờ
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import Sequence

from app.core.config import settings
from app.services.models.parsed_document import EnrichedChunk

logger = logging.getLogger(__name__)

# -- Mẫu boilerplate đã compile --
# Mỗi pattern khớp một chunk ĐẦY ĐỦ có tính boilerplate chiếm ưu thế.
# Dùng re.IGNORECASE | re.DOTALL để xử lý được chunk nhiều dòng.

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Dòng copyright / license
    re.compile(
        r"^[\s\S]{0,30}(?:©|copyright|\(c\)|all\s+rights?\s+reserved)"
        r"[\s\S]{0,300}$",
        re.IGNORECASE,
    ),
    # Câu disclaimer kiểu "confidential" / "proprietary"
    re.compile(
        r"^[\s\S]{0,30}(?:confidential|proprietary|internal\s+use\s+only)"
        r"[\s\S]{0,300}$",
        re.IGNORECASE,
    ),
    # Chỉ chứa số trang ("Page 3", "- 12 -", "3 / 10", "Trang 5")
    re.compile(
        r"^\s*(?:page|trang|p\.?)?\s*\d{1,4}\s*(?:[/of|trên]\s*\d{1,4})?\s*$",
        re.IGNORECASE,
    ),
    # Dấu gạch/underscore/equals lặp lại (vạch phân tách thị giác)
    re.compile(r"^\s*[-_=~*]{4,}\s*$"),
    # Heading độc lập kiểu "Table of Contents" / "Mục lục"
    re.compile(
        r"^\s*(?:table\s+of\s+contents?|mục\s+lục|nội\s+dung)\s*$",
        re.IGNORECASE,
    ),
    # Text kiểu draft / watermark
    re.compile(
        r"^\s*(?:draft|bản\s+nháp|watermark|confidential)\s*$",
        re.IGNORECASE,
    ),
    # Pattern header/footer: "Company Name | Page X" hoặc "Report Title - 2024"
    re.compile(
        r"^[A-ZÀ-Ỹa-zà-ỹ\s\-|·•]{3,60}\s*[|·•\-—]\s*(?:page|trang|p\.?)?\s*\d{0,4}\s*$",
        re.IGNORECASE,
    ),
]

# Mảnh legal boilerplate tiếng Việt (khớp một phần - nếu chunk CHỨA các cụm này
# VÀ ngắn thì nhiều khả năng là boilerplate)
_LEGAL_FRAGMENTS_VI = [
    "theo quy định của pháp luật",
    "không được sao chép",
    "bảo mật thông tin",
    "điều khoản sử dụng",
    "chịu trách nhiệm trước pháp luật",
    "bản quyền thuộc về",
]

_LEGAL_FRAGMENTS_EN = [
    "all rights reserved",
    "without prior written consent",
    "this document is confidential",
    "for internal use only",
    "subject to change without notice",
    "disclaimer:",
    "terms and conditions",
    "no part of this publication",
]


def _normalize_text(text: str) -> str:
    """Rút gọn khoảng trắng và chuyển lowercase để so sánh."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _content_hash(text: str) -> str:
    """SHA-256 của text đã normalize."""
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """Sinh shingles n-gram cấp ký tự từ text đã normalize."""
    normed = _normalize_text(text)
    if len(normed) < n:
        return {normed}
    return {normed[i : i + n] for i in range(len(normed) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity giữa hai tập."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# -- Giai đoạn 1: Noise Filter --

def _is_boilerplate(text: str) -> bool:
    """Kiểm tra text có khớp pattern boilerplate đã biết không."""
    stripped = text.strip()

    # Pattern khớp toàn phần
    for pattern in _BOILERPLATE_PATTERNS:
        if pattern.match(stripped):
            return True

    # Chunk ngắn có chứa legal fragment
    normed = stripped.lower()
    if len(stripped) < 300:
        for frag in _LEGAL_FRAGMENTS_VI + _LEGAL_FRAGMENTS_EN:
            if frag in normed:
                return True

    return False


def _meaningful_char_count(text: str) -> int:
    """Đếm ký tự không phải khoảng trắng và dấu câu."""
    return len(re.sub(r"[\s\-_=~*|#>•·\"\'`(){}\[\]]+", "", text))


def filter_noise(chunks: list[EnrichedChunk]) -> list[EnrichedChunk]:
    """
    Stage 1: Loại chunk chủ yếu là nhiễu.

        Loại bỏ:
            - Chunk ngắn hơn DEDUP_MIN_CHUNK_LENGTH ký tự có nghĩa
            - Boilerplate header/footer/legal disclaimer/copyright notice
            - Chunk chỉ có khoảng trắng hoặc chỉ có ký tự format

    Giữ lại chunk có image_refs hoặc table_refs bất kể độ dài text,
    vì caption enrich của chúng vẫn mang giá trị semantic.
    """
    min_len = settings.NEXUSRAG_DEDUP_MIN_CHUNK_LENGTH
    kept: list[EnrichedChunk] = []
    removed = 0

    for chunk in chunks:
        # Luôn giữ chunk có ảnh/bảng đính kèm
        if chunk.image_refs or chunk.table_refs:
            kept.append(chunk)
            continue

        text = chunk.content.strip()

        # Rỗng / chỉ khoảng trắng
        if not text:
            removed += 1
            continue

        # Quá ngắn (sau khi loại ký tự format)
        if _meaningful_char_count(text) < min_len:
            removed += 1
            continue

        # Khớp boilerplate
        if _is_boilerplate(text):
            removed += 1
            continue

        kept.append(chunk)

    if removed:
        logger.info(f"Noise filter: removed {removed}/{len(chunks)} boilerplate/short chunks")

    return kept


# -- Giai đoạn 2: Exact Dedup --

def dedup_exact(chunks: list[EnrichedChunk]) -> list[EnrichedChunk]:
    """
    Stage 2: Loại chunk có nội dung normalize giống hệt nhau.

    Dùng SHA-256 trên text lowercase và rút gọn khoảng trắng. Giữ lần xuất hiện đầu tiên.
    """
    seen_hashes: set[str] = set()
    kept: list[EnrichedChunk] = []
    removed = 0

    for chunk in chunks:
        h = _content_hash(chunk.content)
        if h in seen_hashes:
            removed += 1
            continue
        seen_hashes.add(h)
        kept.append(chunk)

    if removed:
        logger.info(f"Exact dedup: removed {removed}/{len(chunks)} identical chunks")

    return kept


# -- Giai đoạn 3: Near-duplicate Detection --

def dedup_near(
    chunks: list[EnrichedChunk],
    threshold: float | None = None,
) -> list[EnrichedChunk]:
    """
    Stage 3: Loại chunk gần trùng bằng Jaccard similarity
    trên shingles n-gram ký tự.

    Với mỗi cặp, chunk ĐẾN SAU (theo chunk_index) sẽ bị loại khi
    similarity >= threshold. Độ phức tạp O(n^2) nhưng n thường < 200 chunk
    mỗi document nên vẫn đủ nhanh.
    """
    if threshold is None:
        threshold = settings.NEXUSRAG_DEDUP_NEAR_THRESHOLD

    if threshold >= 1.0:
        return chunks  # da tat

    # Tiền tính shingles
    shingles = [_char_ngrams(c.content) for c in chunks]

    drop_indices: set[int] = set()

    for i in range(len(chunks)):
        if i in drop_indices:
            continue
        for j in range(i + 1, len(chunks)):
            if j in drop_indices:
                continue
            sim = _jaccard_similarity(shingles[i], shingles[j])
            if sim >= threshold:
                drop_indices.add(j)

    kept = [c for idx, c in enumerate(chunks) if idx not in drop_indices]
    removed = len(drop_indices)

    if removed:
        logger.info(
            f"Near dedup (threshold={threshold:.2f}): "
            f"removed {removed}/{len(chunks)} near-duplicate chunks"
        )

    return kept


# -- API công khai --

def deduplicate_chunks(
    chunks: list[EnrichedChunk],
) -> tuple[list[EnrichedChunk], dict[str, int]]:
    """
    Chạy toàn bộ pipeline deduplication 3 giai đoạn.

    Returns:
        (filtered_chunks, stats) trong đó stats = {
            "input": tổng số chunk đầu vào,
            "noise_removed": số lượng bị loại bởi noise filter,
            "exact_removed": số lượng bị loại bởi exact dedup,
            "near_removed": số lượng bị loại bởi near dedup,
            "output": tổng số chunk đầu ra,
        }
    """
    if not settings.NEXUSRAG_DEDUP_ENABLED:
        return chunks, {"input": len(chunks), "output": len(chunks),
                        "noise_removed": 0, "exact_removed": 0, "near_removed": 0}

    total_input = len(chunks)

    # Giai đoạn 1: Noise filter
    after_noise = filter_noise(chunks)
    noise_removed = total_input - len(after_noise)

    # Giai đoạn 2: Exact dedup
    after_exact = dedup_exact(after_noise)
    exact_removed = len(after_noise) - len(after_exact)

    # Giai đoạn 3: Near dedup
    after_near = dedup_near(after_exact)
    near_removed = len(after_exact) - len(after_near)

    # Đánh lại chunk_index liên tục
    for i, chunk in enumerate(after_near):
        chunk.chunk_index = i

    stats = {
        "input": total_input,
        "noise_removed": noise_removed,
        "exact_removed": exact_removed,
        "near_removed": near_removed,
        "output": len(after_near),
    }

    total_removed = total_input - len(after_near)
    if total_removed:
        logger.info(
            f"Dedup pipeline: {total_input} → {len(after_near)} chunks "
            f"(-{total_removed}: noise={noise_removed}, exact={exact_removed}, "
            f"near={near_removed})"
        )

    return after_near, stats
