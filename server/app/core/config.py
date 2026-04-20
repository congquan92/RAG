from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

# Tìm file .env - ưu tiên project root, fallback cho Docker
_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    # Ứng dụng
    APP_NAME: str = "NexusRAG"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Thư mục gốc (backend folder)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Cơ sở dữ liệu
    DATABASE_URL: str = Field(default="postgresql+asyncpg://anhquan:anhquandeptrai@localhost:5433/graprag")

    # LLM Provider: "gemini" | "ollama"
    LLM_PROVIDER: str = Field(default="gemini")

    # Google AI
    GOOGLE_AI_API_KEY: str = Field(default="")

    # Ollama
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="gemma3:12b")
    OLLAMA_ENABLE_THINKING: bool = Field(default=False)

    # LLM (model nhanh cho chat + KG extraction — dùng khi provider=gemini)
    LLM_MODEL_FAST: str = Field(default="gemini-2.5-flash")

    # Mức thinking cho model Gemini 3.x+: "minimal" | "low" | "medium" | "high"
    # Gemini 2.5 dùng thinking_budget_tokens thay thế (auto-detected)
    LLM_THINKING_LEVEL: str = Field(default="medium")

    # Số output tokens tối đa cho phản hồi chat của LLM (bao gồm thinking tokens)
    # Gemini 3.1 Flash-Lite hỗ trợ tối đa 65536
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=8192)

    # LightRAG LLM timeout (giây) dùng cho timeout worker của KG extraction/query
    LLM_TIMEOUT: int = Field(default=180, ge=1)

    # KG extraction :"llm" | "specialized"
    KG_EXTRACTION_METHOD: str = Field(default="specialized")
    NEXUSRAG_KG_GLINER_MODEL: str = Field(default="urchade/gliner_multi-v2.1")
    NEXUSRAG_KG_RELATION_MODEL: str = Field(default="Babelscape/mrebel-large")

    # KG Embedding provider (llm | gemini)
    KG_EMBEDDING_PROVIDER: str = Field(default="gemini")
    KG_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")
    KG_EMBEDDING_DIMENSION: int = Field(default=3072)

    # ChromaDB
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8002)

    # NexusRAG Pipeline
    NEXUSRAG_ENABLED: bool = True
    NEXUSRAG_ENABLE_KG: bool = True
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = True
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = True
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = True
    NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS: int = 8000
    NEXUSRAG_CHUNK_MAX_TOKENS: int = 512
    NEXUSRAG_KG_QUERY_TIMEOUT: float = 30.0
    NEXUSRAG_KG_CHUNK_TOKEN_SIZE: int = 1200
    NEXUSRAG_KG_LANGUAGE: str = "English"
    NEXUSRAG_KG_ENTITY_TYPES: list[str] = [
        "Organization", "Person", "Product", "Location", "Event",
        "Financial_Metric", "Technology", "Date", "Regulation",
    ]
    NEXUSRAG_DEFAULT_QUERY_MODE: str = "hybrid"
    NEXUSRAG_DOCLING_IMAGES_SCALE: float = 2.0
    NEXUSRAG_MAX_IMAGES_PER_DOC: int = 50
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = True

    # Document Parser provider: "docling" (mặc định) hoặc "marker" (nhẹ hơn, tốt hơn cho math)
    NEXUSRAG_DOCUMENT_PARSER: str = "docling"
    NEXUSRAG_MARKER_USE_LLM: bool = False

    # Processing timeout (phút) — document bị stale sẽ tự khôi phục về FAILED
    NEXUSRAG_PROCESSING_TIMEOUT_MINUTES: int = 10

    # Deduplication trước ingestion
    NEXUSRAG_DEDUP_ENABLED: bool = True
    NEXUSRAG_DEDUP_MIN_CHUNK_LENGTH: int = 50       # số ký tự có ý nghĩa tối thiểu
    NEXUSRAG_DEDUP_NEAR_THRESHOLD: float = 0.85     # ngưỡng Jaccard similarity

    # Chất lượng Retrieval của NexusRAG
    NEXUSRAG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    NEXUSRAG_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    NEXUSRAG_VECTOR_PREFETCH: int = 20
    NEXUSRAG_RERANKER_TOP_K: int = Field(default=8, ge=1)
    NEXUSRAG_MIN_RELEVANCE_SCORE: float = 0.15

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5174", "http://localhost:3000"]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
