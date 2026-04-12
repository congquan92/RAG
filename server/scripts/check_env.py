from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import dotenv_values


REQUIRED_ENV_KEYS = [
    "LLM_PROVIDER",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "GEMINI_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "DOCLING_DEVICE",
    "RERANKER_DEVICE",
    "CORS_ORIGINS",
    "API_HOST",
    "API_PORT",
    "API_RELOAD",
    "DATABASE_URL",
    "CHROMA_PERSIST_DIR",
    "CHROMA_COLLECTION_NAME",
    "LIGHTRAG_WORKING_DIR",
    "UPLOAD_DIR",
    "MAX_UPLOAD_SIZE_MB",
    "ALLOWED_UPLOAD_MIME_TYPES",
    "RETRIEVAL_TOP_K",
    "RERANKER_TOP_K",
    "RERANKER_MODEL",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
]


def _is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def validate_env(env_path: Path) -> tuple[list[str], list[str]]:
    values = dotenv_values(env_path)
    missing: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED_ENV_KEYS:
        if _is_blank(values.get(key)):
            missing.append(key)

    llm_provider = str(values.get("LLM_PROVIDER") or "").strip().lower()
    embedding_provider = str(values.get("EMBEDDING_PROVIDER") or "").strip().lower()

    if llm_provider not in {"", "ollama", "gemini"}:
        warnings.append("LLM_PROVIDER should be one of: ollama, gemini")

    if embedding_provider not in {"", "sentence-transformers", "ollama", "gemini"}:
        warnings.append(
            "EMBEDDING_PROVIDER should be one of: sentence-transformers, ollama, gemini"
        )

    if llm_provider == "gemini" or embedding_provider == "gemini":
        if _is_blank(values.get("GEMINI_API_KEY")):
            missing.append("GEMINI_API_KEY")

    return sorted(set(missing)), warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check required .env variables")
    parser.add_argument("--env-file", default=".env", help="Path to env file (default: .env)")
    parser.add_argument("--quiet", action="store_true", help="Print only errors")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.exists():
        print(f"[ENV CHECK] File not found: {env_path}")
        print("[ENV CHECK] Create it from .env.example before starting server.")
        return 1

    missing, warnings = validate_env(env_path)

    if missing:
        print("[ENV CHECK] Missing required variables:")
        for key in missing:
            print(f"  - {key}")
        print("[ENV CHECK] Please update .env and retry.")
        return 1

    if not args.quiet:
        print(f"[ENV CHECK] OK: {env_path} has all required variables.")
        if warnings:
            print("[ENV CHECK] Warnings:")
            for msg in warnings:
                print(f"  - {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())