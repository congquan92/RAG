"""
Test Script: Ingestion Pipeline
Kiểm tra file parsing (PyMuPDF) và chunking (Tiktoken).

Usage:
    cd server
    source .venv/bin/activate
    python scripts/test_ingestion.py

Tự tạo file test tạm, không cần chuẩn bị gì trước.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_mime_detection():
    """Test MIME type detection."""
    from app.rag.ingestion import detect_mime_type

    print("=" * 60)
    print("  TEST: MIME Type Detection")
    print("=" * 60)

    # Tạo file test tạm
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("Hello world")
        txt_path = f.name

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write("# Heading\nSome content")
        md_path = f.name

    txt_mime = detect_mime_type(txt_path)
    md_mime = detect_mime_type(md_path)

    print(f"  .txt → {txt_mime}")
    print(f"  .md  → {md_mime}")

    assert "text" in txt_mime, f"Expected text/* for .txt, got {txt_mime}"
    print("  ✅ MIME detection works!\n")

    # Cleanup
    Path(txt_path).unlink(missing_ok=True)
    Path(md_path).unlink(missing_ok=True)


def test_plain_text_extraction():
    """Test trích xuất text từ file plain text."""
    from app.rag.ingestion import extract_text_plain

    print("=" * 60)
    print("  TEST: Plain Text Extraction")
    print("=" * 60)

    # Tạo file test
    sample_text = "Đây là đoạn văn bản test.\n" * 50
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
        f.write(sample_text)
        test_path = f.name

    result = extract_text_plain(test_path)

    print(f"  Extracted: {len(result.full_text)} chars")
    print(f"  Method:    {result.extraction_method}")
    print(f"  Error:     {result.error}")

    assert result.full_text.strip(), "Should extract non-empty text"
    assert result.error is None, f"Should not have error: {result.error}"
    print("  ✅ Plain text extraction works!\n")

    Path(test_path).unlink(missing_ok=True)


def test_chunking():
    """Test Tiktoken-based chunking."""
    from app.rag.ingestion import split_into_chunks

    print("=" * 60)
    print("  TEST: Tiktoken Chunking")
    print("=" * 60)

    # Tạo text đủ dài để chia thành nhiều chunks
    paragraphs = []
    for i in range(30):
        paragraphs.append(
            f"Đoạn văn số {i+1}. Đây là nội dung chi tiết về chủ đề số {i+1}. "
            f"Nó chứa nhiều thông tin hữu ích cho việc kiểm tra hệ thống chunking. "
            f"Mỗi đoạn có khoảng 50-60 từ để đảm bảo chunking hoạt động đúng."
        )
    long_text = "\n\n".join(paragraphs)

    chunks = split_into_chunks(
        text=long_text,
        source_file="test_document.txt",
        chunk_size=100,  # Nhỏ để dễ test
        chunk_overlap=20,
    )

    print(f"  Input:  {len(long_text)} chars")
    print(f"  Chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i}: {len(chunk.text)} chars, "
              f"tokens={chunk.metadata.get('token_count', '?')}, "
              f"source={chunk.metadata.get('source_file', '?')}")

    assert len(chunks) > 1, f"Should create multiple chunks, got {len(chunks)}"
    assert all(c.metadata.get("source_file") == "test_document.txt" for c in chunks)
    assert all(c.metadata.get("chunk_index") == i for i, c in enumerate(chunks))
    print("  ✅ Chunking works!\n")


def test_full_ingestion_pipeline():
    """Test full pipeline: extract → chunk."""
    from app.rag.ingestion import ingest_file

    print("=" * 60)
    print("  TEST: Full Ingestion Pipeline")
    print("=" * 60)

    # Tạo file test
    content = "\n\n".join([
        f"# Chapter {i+1}\n\nThis is the content of chapter {i+1}. "
        f"It contains important information about topic {i+1} that should be "
        f"extracted and chunked properly by the ingestion pipeline."
        for i in range(20)
    ])

    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
        f.write(content)
        test_path = f.name

    result = ingest_file(test_path)

    print(f"  File:     {Path(test_path).name}")
    print(f"  Method:   {result.extraction_method}")
    print(f"  Text:     {len(result.full_text)} chars")
    print(f"  Chunks:   {len(result.chunks)}")
    print(f"  Error:    {result.error}")

    assert result.full_text, "Should have extracted text"
    assert len(result.chunks) > 0, "Should have created chunks"
    assert result.error is None, f"Should not have error: {result.error}"

    print("\n  First 3 chunks preview:")
    for i, chunk in enumerate(result.chunks[:3]):
        preview = chunk.text[:80].replace("\n", " ")
        print(f"    [{i}] {preview}...")

    print("\n  ✅ Full ingestion pipeline works!\n")

    Path(test_path).unlink(missing_ok=True)


def test_nonexistent_file():
    """Test handling file không tồn tại."""
    from app.rag.ingestion import ingest_file

    print("=" * 60)
    print("  TEST: Non-existent File Handling")
    print("=" * 60)

    result = ingest_file("/nonexistent/fake_file.pdf")

    assert result.error is not None, "Should return error for missing file"
    assert not result.full_text, "Should have empty text for missing file"
    print(f"  ✅ Error handled: {result.error}")
    print("  ✅ Non-existent file handled gracefully!\n")


if __name__ == "__main__":
    test_mime_detection()
    test_plain_text_extraction()
    test_chunking()
    test_full_ingestion_pipeline()
    test_nonexistent_file()
    print("🎉 All ingestion tests passed!")
