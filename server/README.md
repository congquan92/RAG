## 1. Tổng quan (Overview)

| Mục            | Mô tả ngắn                                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------------- |
| Tech stack     | FastAPI, SQLAlchemy Async + PostgreSQL, ChromaDB, LightRAG, Ollama/Gemini, sentence-transformers            |
| Vai trò        | Upload tài liệu, parse/index tài liệu, truy vấn RAG (sync + streaming), quản lý lịch sử chat theo workspace |
| Đường dẫn API  | `http://localhost:8000/api/v1`                                                                              |
| Health check   | `http://localhost:8000/health`                                                                              |
| Vector backend | ChromaDB HTTP (`CHROMA_HOST`, `CHROMA_PORT`) + LightRAG graph retrieval                                     |

> Note: Nếu backend chạy trong Docker, cần map endpoint Ollama đúng theo `OLLAMA_HOST` để server gọi model local.

## 2. Cấu trúc chính (Architecture)

### 2.1 Bố cục dự án (server/)

```text
server/
├── app/
│   ├── api/            # FastAPI routers
│   ├── core/           # config, database, deps, exceptions
│   ├── models/         # SQLAlchemy models
│   ├── schemas/        # Pydantic request/response models
│   ├── services/       # RAG/NexusRAG/KG business logic
│   ├── schema.sql      # bootstrap schema SQL
│   └── main.py         # FastAPI app entrypoint
├── scripts/            # utility/evaluation scripts
├── streamlit/          # streamlit assets (tuỳ chọn)
├── requirements.txt
└── README.md
```

| Khối                         | Vai trò                                       |
| ---------------------------- | --------------------------------------------- |
| `app/api`                    | API layer: workspaces, documents, rag, config |
| `app/services`               | Nghiệp vụ parse/index/retrieve/rerank/KG      |
| `app/models` + `app/schemas` | Data model DB và API contract                 |
| `app/core`                   | Config, database session, dependency          |
| `scripts`                    | Lệnh hỗ trợ tải model, dọn cache, đánh giá    |

| Thư mục/File                              | Vai trò                                                        |
| ----------------------------------------- | -------------------------------------------------------------- |
| `app/main.py`                             | App bootstrap, lifespan startup/shutdown, mount routers/static |
| `app/api/router.py`                       | Tổng hợp router con vào `/api/v1`                              |
| `app/api/workspaces.py`                   | Workspace CRUD                                                 |
| `app/api/documents.py`                    | Upload/list/get/delete document, markdown/images               |
| `app/api/rag.py`                          | Query/process/reindex/KG analytics/chat/history                |
| `app/api/chat_agent.py`                   | SSE streaming chat semi-agentic                                |
| `app/api/config.py`                       | Trạng thái provider/model cho frontend                         |
| `app/core/config.py`                      | Cấu hình đọc từ `.env`                                         |
| `app/core/database.py`                    | Async engine/session factory                                   |
| `app/services/nexus_rag_service.py`       | Pipeline parse -> index -> KG ingest                           |
| `app/services/deep_retriever.py`          | Hybrid retrieval + reranking + image/table refs                |
| `app/services/knowledge_graph_service.py` | LightRAG per workspace                                         |
| `app/services/vector_store.py`            | Chroma collection theo `kb_{workspace_id}`                     |
| `scripts/download_models.py`              | Tải trước model local                                          |

## 3. API Endpoints

### 3.1 System

| Method | Path                    | Mục đích                     | Ghi chú nhanh                  |
| ------ | ----------------------- | ---------------------------- | ------------------------------ |
| GET    | `/health`               | Kiểm tra liveness            | Trả `{status: healthy}`        |
| GET    | `/ready`                | Kiểm tra readiness           | Trả `{status: ready}`          |
| GET    | `/api/v1/config/status` | Xem provider/model hiện dùng | Dùng cho frontend status badge |

### 3.2 Workspaces

| Method | Path                                | Mục đích           | Ghi chú nhanh                                |
| ------ | ----------------------------------- | ------------------ | -------------------------------------------- |
| GET    | `/api/v1/workspaces`                | Liệt kê workspaces | Kèm `document_count`, `indexed_count`        |
| POST   | `/api/v1/workspaces`                | Tạo workspace      | Body: `name`, `description?`, `kg_language?` |
| GET    | `/api/v1/workspaces/summary`        | Danh sách rút gọn  | Dùng cho dropdown                            |
| GET    | `/api/v1/workspaces/{workspace_id}` | Chi tiết workspace |                                              |
| PUT    | `/api/v1/workspaces/{workspace_id}` | Cập nhật workspace | Hỗ trợ `system_prompt`, KG settings          |
| DELETE | `/api/v1/workspaces/{workspace_id}` | Xoá workspace      | Cleanup vector/KG/images liên quan           |

### 3.3 Documents

| Method | Path                                         | Mục đích                         | Ghi chú nhanh                                |
| ------ | -------------------------------------------- | -------------------------------- | -------------------------------------------- |
| GET    | `/api/v1/documents/workspace/{workspace_id}` | Liệt kê documents theo workspace |                                              |
| POST   | `/api/v1/documents/upload/{workspace_id}`    | Upload document                  | Multipart `file`, optional `custom_metadata` |
| GET    | `/api/v1/documents/{document_id}`            | Chi tiết document                | Metadata + status/chunk_count                |
| GET    | `/api/v1/documents/{document_id}/markdown`   | Lấy markdown đã parse            | Trả `text/markdown`                          |
| GET    | `/api/v1/documents/{document_id}/images`     | Lấy ảnh extract                  | URL ảnh qua `/static/doc-images/...`         |
| DELETE | `/api/v1/documents/{document_id}`            | Xoá document + cleanup vector    |                                              |

### 3.4 RAG + KG + Chat

| Method | Path                                           | Mục đích                       | Ghi chú nhanh                     |
| ------ | ---------------------------------------------- | ------------------------------ | --------------------------------- |
| POST   | `/api/v1/rag/query/{workspace_id}`             | Query chunks/citations/context | Body: `question`, `top_k`, `mode` |
| POST   | `/api/v1/rag/process/{document_id}`            | Trigger process 1 document     | Chạy background                   |
| POST   | `/api/v1/rag/process-batch`                    | Process nhiều document         | Body: `document_ids`              |
| POST   | `/api/v1/rag/reindex/{document_id}`            | Reindex 1 document             | Reset metadata + index lại        |
| POST   | `/api/v1/rag/reindex-workspace/{workspace_id}` | Reindex toàn workspace         | Chạy nền                          |
| GET    | `/api/v1/rag/stats/{workspace_id}`             | RAG stats                      | Tổng docs/chunks/images           |
| GET    | `/api/v1/rag/chunks/{document_id}`             | Xem chunks theo document       |                                   |
| GET    | `/api/v1/rag/entities/{workspace_id}`          | KG entities                    | Filter `search`, `entity_type`    |
| GET    | `/api/v1/rag/relationships/{workspace_id}`     | KG relationships               |                                   |
| GET    | `/api/v1/rag/graph/{workspace_id}`             | Payload graph cho frontend     |                                   |
| GET    | `/api/v1/rag/analytics/{workspace_id}`         | Analytics tổng hợp             | Stats + KG + breakdown            |
| GET    | `/api/v1/rag/chat/{workspace_id}/history`      | Lịch sử chat                   |                                   |
| DELETE | `/api/v1/rag/chat/{workspace_id}/history`      | Xoá lịch sử chat               |                                   |
| POST   | `/api/v1/rag/chat/{workspace_id}/rate`         | Đánh giá source citation       |                                   |
| POST   | `/api/v1/rag/chat/{workspace_id}`              | Chat non-stream                |                                   |
| POST   | `/api/v1/rag/chat/{workspace_id}/stream`       | Chat streaming SSE             | Events: status/thinking/token/... |
| GET    | `/api/v1/rag/capabilities`                     | Khả năng model                 | thinking + vision                 |
| POST   | `/api/v1/rag/debug-chat/{workspace_id}`        | Debug retrieval/prompt/answer  | Hỗ trợ tune chất lượng            |

## 4. Retrieval Modes (Chế độ truy vấn)

- `hybrid`: Kết hợp vector + KG context, có rerank (mặc định khuyến nghị)
- `vector_only`: Chỉ vector similarity
- `naive`: Chế độ cơ bản của KG backend
- `local`: KG query thiên về neighborhood local
- `global`: KG query thiên về global context

Ghi chú: endpoint chat hiện gọi retrieval theo `hybrid` để ưu tiên chất lượng trả lời.

## 5. Request Fields hay dùng

| Field             | Kiểu           | Ghi chú                                         |
| ----------------- | -------------- | ----------------------------------------------- |
| `workspace_id`    | int (path)     | Định danh knowledge base/workspace              |
| `document_id`     | int (path)     | Định danh document                              |
| `question`        | string         | Dùng cho `/rag/query/{workspace_id}`            |
| `message`         | string         | Dùng cho `/rag/chat/{workspace_id}`             |
| `history`         | list           | Lịch sử hội thoại trước đó cho chat             |
| `document_ids`    | list[int]      | Giới hạn retrieval trên tập document chỉ định   |
| `top_k`           | int            | Số chunk muốn lấy                               |
| `mode`            | string         | Chế độ retrieval (`hybrid`, `vector_only`, ...) |
| `file`            | multipart file | File upload document                            |
| `custom_metadata` | string(JSON)   | Metadata tuỳ chỉnh khi upload                   |
| `enable_thinking` | bool           | Bật/tắt thinking ở endpoint chat                |

## 6. Biến môi trường quan trọng (Environment Variables)

```bash
# Đường dẫn kết nối Database (Cổng 5433 để né SlideAgentPro)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag

# Cấu hình ChromaDB (Nơi lưu trữ Vector tìm kiếm - Cổng 8002)
CHROMA_HOST=localhost
CHROMA_PORT=8002

# LLM Provider: "gemini" | "ollama"
# --- Gemini ---
LLM_PROVIDER=gemini
GOOGLE_AI_API_KEY=your-gemini-api-key
LLM_MODEL_FAST=gemini-2.5-flash

#Thinking level for Gemini 3.x "minimal" | "low" | "medium" | "high"
# Gemini 2.5 uses thinking_budget_tokens automatically (this setting is ignored)
LLM_THINKING_LEVEL=medium

# Max output tokens for chat (includes thinking tokens)
# Gemini 2.5 Flash: up to 8192, Gemini 3.1 Flash-Lite: up to 65536
LLM_MAX_OUTPUT_TOKENS=8192

# --- Ollama (inactive — uncomment use ) ---
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# Multimodal (Recommend)
# OLLAMA_MODEL=qwen3.5:9b
# OLLAMA_MODEL=qwen3.5:4b
# OLLAMA_MODEL=gemma3:12b
# Text only model
# OLLAMA_MODEL=kamekichi128/qwen3-4b-instruct-2507
# Enable thinking mode for Ollama (default: false — reduces latency for thinking models like qwen3.5)
# OLLAMA_ENABLE_THINKING=false

# ===========================================
# KG Embedding (có thể khác LLM provider)
# ===========================================

# --- KG Embedding: Gemini (active) ---
KG_EMBEDDING_PROVIDER=gemini
KG_EMBEDDING_MODEL=gemini-embedding-001
KG_EMBEDDING_DIMENSION=3072

# --- KG Embedding: Ollama (inactive) ---
# KG_EMBEDDING_PROVIDER=ollama
# KG_EMBEDDING_MODEL=bge-m3
# KG_EMBEDDING_DIMENSION=1024

# --- KG Embedding: Sentence-Transformers (inactive — fully local, no API needed) ---
# KG_EMBEDDING_PROVIDER=sentence_transformers
# KG_EMBEDDING_MODEL=BAAI/bge-m3
# KG_EMBEDDING_DIMENSION=1024


# =================================================================
# NexusRAG Pipeline (all optional, defaults shown)
# =================================================================

# Bật/Tắt hệ thống NexusRAG và Đồ thị tri thức (Knowledge Graph)
NEXUSRAG_ENABLED=true
NEXUSRAG_ENABLE_KG=true
# Global default KG language (can be overridden per workspace in UI)( đồ thị cây)
NEXUSRAG_KG_LANGUAGE=English
# Các loại thực thể AI cần bóc tách từ tài liệu
NEXUSRAG_KG_ENTITY_TYPES=["Organization","Person","Product","Location","Event","Financial_Metric","Technology","Date","Regulation"]
#Document Parser: "docling" "docling" (Xịn, nặng GPU) hoặc "marker" (lighter ~2-4GB, better math/LaTeX)
# NEXUSRAG_DOCUMENT_PARSER=docling
# NEXUSRAG_MARKER_USE_LLM=false

# Bật tính năng bóc ảnh từ tài liệu và viết mô tả ảnh bằng AI
NEXUSRAG_ENABLE_IMAGE_EXTRACTION=true
NEXUSRAG_ENABLE_IMAGE_CAPTIONING=true
# Model xử lý tìm kiếm và sắp xếp kết quả (Reranker)
NEXUSRAG_EMBEDDING_MODEL=BAAI/bge-m3
NEXUSRAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
# Cấu hình chia nhỏ tài liệu (Chunking)
NEXUSRAG_CHUNK_MAX_TOKENS=512
NEXUSRAG_VECTOR_PREFETCH=20
NEXUSRAG_RERANKER_TOP_K=8
NEXUSRAG_MIN_RELEVANCE_SCORE=0.15

# Processing timeout — stale documents auto-recover to FAILED after this many minutes
# NEXUSRAG_PROCESSING_TIMEOUT_MINUTES=10

# =================================================================
# CHỐNG TRÙNG LẶP DỮ LIỆU - Deduplication (noise filter + content-hash + near-duplicate)
# =================================================================
# Bật lọc rác và lọc trùng nội dung trước khi nạp vào máy
NEXUSRAG_DEDUP_ENABLED=true
NEXUSRAG_DEDUP_MIN_CHUNK_LENGTH=50# các đoạn văn ngắn dưới 50 chữ (thường là rác)
NEXUSRAG_DEDUP_NEAR_THRESHOLD=0.85# (0.85 = giống trên 85% thì coi là trùng)



# CORS
CORS_ORIGINS=["http://localhost:5174","http://localhost:3000"]
# uvicorn app.main:app --reload --port 8080

```

## 7. Setup nhanh (Quick Start)

```bash
# 1) Tạo và kích hoạt virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Cài dependency
pip install -r requirements.txt

# 3) Tạo file env
cp .env.example .env
# chỉnh sửa .env theo máy của bạn

# 4) Chạy server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI:

- `http://localhost:8000/docs`

ReDoc:

- `http://localhost:8000/redoc`

## 8. Lệnh vận hành hữu ích

```bash
# Tải trước model local (nếu cần)
python scripts/download_models.py

# Dọn cache
python scripts/clean_cache.py

# Đánh giá RAG
python scripts/eval_rag.py

# Đánh giá synthetic theo Ragas
python scripts/eval_ragas_synthetic.py
```

## 9. Lưu ý

1. Upload và process là 2 bước tách riêng: upload xong cần gọi process/reindex để index.
2. Streaming endpoint `/api/v1/rag/chat/{workspace_id}/stream` trả SSE events theo thời gian thực, client cần parse event stream.
3. Nếu dùng Gemini (LLM hoặc KG embedding), bắt buộc set `GOOGLE_AI_API_KEY`.
4. Static ảnh tài liệu được mount qua `/static/doc-images`.
5. Với ingest lớn, hạn chế bật reload để tránh gián đoạn pipeline.
