from __future__ import annotations
from typing import Literal
from pydantic import ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _format_settings_error(exc: ValidationError) -> str:
    missing_keys: list[str] = []
    other_errors: list[str] = []

    for err in exc.errors():
        err_type = str(err.get("type", ""))
        loc = err.get("loc", ())
        field = loc[0] if isinstance(loc, tuple) and loc else None

        if err_type == "missing" and isinstance(field, str):
            missing_keys.append(field.upper())
            continue

        msg = str(err.get("msg", "Invalid value"))
        if isinstance(field, str):
            other_errors.append(f"{field.upper()}: {msg}")
        else:
            other_errors.append(msg)

    lines = ["Invalid environment configuration in server/.env."]

    if missing_keys:
        lines.append("Missing required variables:")
        for key in sorted(set(missing_keys)):
            lines.append(f"  - {key}")

    if other_errors:
        lines.append("Invalid values:")
        for item in other_errors:
            lines.append(f"  - {item}")

    lines.append("Run: python scripts/check_env.py")
    return "\n".join(lines)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # bỏ qua biến .env không khai báo ở đây
    )

    # ── LLM Provider (Hot-Swap) ──────────────────────────────────────────
    llm_provider: Literal["ollama", "gemini"]

    # Ollama
    ollama_base_url: str
    ollama_model: str

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str

    # ── Embedding Provider (Hot-Swap) ────────────────────────────────────
    embedding_provider: Literal["sentence-transformers", "ollama", "gemini"]
    embedding_model: str

    # ── Hardware Device Control ──────────────────────────────────────────
    embedding_device: str
    docling_device: str
    reranker_device: str

    # ── CORS ─────────────────────────────────────────────────────────────
    cors_origins: str

    # ── API Runtime ───────────────────────────────────────────────────────
    api_host: str
    api_port: int
    api_reload: bool

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str
    chroma_persist_dir: str
    chroma_collection_name: str
    lightrag_working_dir: str

    # ── Runtime Storage & Upload ────────────────────────────────────────
    upload_dir: str
    max_upload_size_mb: int
    allowed_upload_mime_types: str

    # ── Retrieval Tuning ─────────────────────────────────────────────────
    retrieval_top_k: int
    reranker_top_k: int
    reranker_model: str
    chunk_size: int
    chunk_overlap: int

    # ── Computed helpers ─────────────────────────────────────────────────

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS_ORIGINS into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def allowed_upload_mime_type_set(self) -> set[str]:
        """Parse comma-separated ALLOWED_UPLOAD_MIME_TYPES into a set."""
        return {
            mime.strip()
            for mime in self.allowed_upload_mime_types.split(",")
            if mime.strip()
        }

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("gemini_api_key")
    @classmethod
    def _validate_gemini_key(cls, value: str, info) -> str:
        """Gemini API key bắt buộc khi provider là gemini."""
        # Validator chạy trước khi instance hoàn chỉnh nên check qua info.data
        provider = info.data.get("llm_provider", "ollama")
        emb_provider = info.data.get("embedding_provider", "sentence-transformers")
        if (provider == "gemini" or emb_provider == "gemini") and not value:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER or "
                "EMBEDDING_PROVIDER is set to 'gemini'. "
                "Set it in your .env file."
            )
        return value

    @field_validator("embedding_device", "docling_device", "reranker_device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        """Chỉ cho phép device hợp lệ."""
        allowed = {"cpu", "cuda", "mps"}
        if value not in allowed:
            raise ValueError(
                f"Invalid device '{value}'. Must be one of: {', '.join(sorted(allowed))}"
            )
        return value

    @field_validator("max_upload_size_mb")
    @classmethod
    def _validate_upload_size(cls, value: int) -> int:
        """Giới hạn upload phải là số dương."""
        if value <= 0:
            raise ValueError("MAX_UPLOAD_SIZE_MB must be greater than 0.")
        return value

    @field_validator("api_port")
    @classmethod
    def _validate_api_port(cls, value: int) -> int:
        """Port phải nằm trong khoảng hợp lệ."""
        if value < 1 or value > 65535:
            raise ValueError("API_PORT must be between 1 and 65535.")
        return value

    @field_validator("retrieval_top_k", "reranker_top_k", "chunk_size", "chunk_overlap")
    @classmethod
    def _validate_positive_runtime_ints(cls, value: int, info) -> int:
        """Các tham số runtime dạng số phải là số dương."""
        if value <= 0:
            raise ValueError(f"{info.field_name.upper()} must be greater than 0.")
        return value


# ── Singleton instance — import này từ mọi nơi trong app ────────────────
try:
    settings = Settings()
except ValidationError as exc:
    raise RuntimeError(_format_settings_error(exc)) from exc