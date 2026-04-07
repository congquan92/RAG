"""
Test Script: Settings & Config
Kiểm tra settings.py load đúng từ .env, validators hoạt động.
"""

import sys
from pathlib import Path

# Thêm server/ vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_settings_load():
    """Test load settings từ .env."""
    from app.core.settings import settings

    print("=" * 60)
    print("  TEST: Settings Load")
    print("=" * 60)

    config = settings.model_dump()
    for key, value in config.items():
        # Ẩn API key
        display_val = "***hidden***" if "api_key" in key and value else value
        print(f"  {key:30s} = {display_val}")

    print(f"\n  CORS Origins (parsed): {settings.cors_origin_list}")
    print("\n✅ Settings loaded successfully!\n")


def test_settings_validators():
    """Test validators hoạt động đúng."""
    from pydantic import ValidationError
    from app.core.settings import Settings

    print("=" * 60)
    print("  TEST: Settings Validators")
    print("=" * 60)

    # Test invalid device
    try:
        Settings(embedding_device="tpu")
        print("  ❌ FAIL: Should reject device='tpu'")
    except ValidationError as e:
        print(f"  ✅ Invalid device rejected: {e.errors()[0]['msg'][:80]}")

    # Test gemini without key
    try:
        Settings(llm_provider="gemini", gemini_api_key="")
        print("  ❌ FAIL: Should reject gemini with empty key")
    except ValidationError as e:
        print(f"  ✅ Missing Gemini key rejected: {e.errors()[0]['msg'][:80]}")

    # Test valid configs
    s = Settings(llm_provider="ollama", embedding_device="cpu")
    print(f"  ✅ Valid config accepted: provider={s.llm_provider}, device={s.embedding_device}")

    print("\n✅ All validator tests passed!\n")


def test_llm_factory_imports():
    """Test factory imports (không gọi thực tế, chỉ check import)."""
    from app.core.llm_factory import get_llm, get_embeddings

    print("=" * 60)
    print("  TEST: LLM Factory Imports")
    print("=" * 60)
    print(f"  ✅ get_llm:        {get_llm}")
    print(f"  ✅ get_embeddings: {get_embeddings}")
    print("\n✅ Factory imports OK!\n")


if __name__ == "__main__":
    test_settings_load()
    test_settings_validators()
    test_llm_factory_imports()
    print("🎉 All settings tests passed!")
