## 1. Tổng quan (Overview)

| Mục            | Mô tả ngắn                                                                                                   |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| Tech stack     | FastAPI, SQLAlchemy Async + PostgreSQL, ChromaDB, LightRAG, Ollama/Gemini, sentence-transformers          |
| Vai trò        | Upload tài liệu, parse/index tài liệu, truy vấn RAG (sync + streaming), quản lý lịch sử chat theo workspace |
| Đường dẫn API  | `http://localhost:8000/api/v1`                                                                               |
| Health check   | `http://localhost:8000/health`                                                                               |
| Vector backend | ChromaDB HTTP (`CHROMA_HOST`, `CHROMA_PORT`) + LightRAG graph retrieval                                      |

> Note: Nếu backend chạy trong Docker/container, kiểm tra lại `OLLAMA_HOST` để backend gọi đúng endpoint model local.

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
├── data/               # dữ liệu runtime (docling, lightrag)
├── scripts/            # utility/evaluation scripts
├── uploads/            # file upload gốc
├── requirements.txt
└── README.md
```

| Khối                         | Vai trò                                       |
| ---------------------------- | --------------------------------------------- |
| `app/api`                    | API layer: workspaces, documents, rag, config |
| `app/services`               | Nghiệp vụ parse/index/retrieve/rerank/KG      |
| `app/models` + `app/schemas` | Data model DB và API contract                  |
| `app/core`                   | Config, database session, dependency           |
| `scripts`                    | Lệnh hỗ trợ tải model, dọn cache, đánh giá    |

| Thư mục/File                              | Vai trò                                                        |
| ----------------------------------------- | -------------------------------------------------------------- |
| `app/main.py`                             | App bootstrap, lifespan startup/shutdown, mount routers/static |
| `app/api/router.py`                       | Tổng hợp router con vào `/api/v1`                              |
| `app/api/workspaces.py`                   | Workspace CRUD                                                 |
| `app/api/documents.py`                    | Upload/list/get/delete document, markdown/images               |
| `app/api/rag.py`                          | Query/process/reindex/KG analytics/chat/history/debug          |
| `app/api/chat_agent.py`                   | SSE streaming chat semi-agentic                                |
| `app/api/config.py`                       | Trạng thái provider/model + default chat prompt                |
| `app/core/config.py`                      | Cấu hình đọc từ `.env`                                         |
| `app/core/database.py`                    | Async engine/session factory                                   |
| `app/services/nexus_rag_service.py`       | Pipeline parse -> index -> KG ingest                           |
| `app/services/deep_retriever.py`          | Hybrid retrieval + reranking + image/table refs                |
| `app/services/knowledge_graph_service.py` | LightRAG per workspace                                         |
| `app/services/vector_store.py`            | Chroma collection theo `kb_{workspace_id}`                     |
| `scripts/download_models.py`              | Tải trước model local                                          |

## 3. API Endpoints

### 3.1 System + Config

| Method | Path                               | Mục đích                            | Ghi chú nhanh                           |
| ------ | ---------------------------------- | ----------------------------------- | --------------------------------------- |
| GET    | `/health`                          | Kiểm tra liveness                   | Trả `{status: healthy}`                 |
| GET    | `/ready`                           | Kiểm tra readiness                  | Trả `{status: ready}`                   |
| GET    | `/api/v1/config/status`            | Xem provider/model đang active      | Dùng cho frontend status badge          |
| GET    | `/api/v1/config/chat-default-prompt` | Lấy default system prompt backend | Dùng khi workspace chưa set prompt riêng |

### 3.2 Workspaces

| Method | Path                                | Mục đích           | Ghi chú nhanh                                |
| ------ | ----------------------------------- | ------------------ | -------------------------------------------- |
| GET    | `/api/v1/workspaces`                | Liệt kê workspaces | Kèm `document_count`, `indexed_count`         |
| POST   | `/api/v1/workspaces`                | Tạo workspace      | Body: `name`, `description?`, `kg_language?`, `kg_entity_types?`, `chunk_size?`, `chunk_overlap?` |
| GET    | `/api/v1/workspaces/summary`        | Danh sách rút gọn  | Dùng cho dropdown                            |
| GET    | `/api/v1/workspaces/{workspace_id}` | Chi tiết workspace |                                              |
| PUT    | `/api/v1/workspaces/{workspace_id}` | Cập nhật workspace | Hỗ trợ `system_prompt`, KG settings, chunk settings |
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

| Method | Path                                           | Mục đích                        | Ghi chú nhanh                              |
| ------ | ---------------------------------------------- | ------------------------------- | ------------------------------------------ |
| POST   | `/api/v1/rag/query/{workspace_id}`             | Query chunks/citations/context  | Body: `question`, `top_k`, `mode`          |
| POST   | `/api/v1/rag/process/{document_id}`            | Trigger process 1 document      | Chạy nền bằng async task                   |
| POST   | `/api/v1/rag/process-batch`                    | Process nhiều document          | Body: `document_ids`                       |
| POST   | `/api/v1/rag/reindex/{document_id}`            | Reindex 1 document              | Xoá dữ liệu cũ rồi index lại               |
| POST   | `/api/v1/rag/reindex-workspace/{workspace_id}` | Reindex toàn workspace          | Chạy nền, hỗ trợ thay đổi embedding dim    |
| GET    | `/api/v1/rag/stats/{workspace_id}`             | RAG stats                       | Tổng docs/chunks/images                    |
| GET    | `/api/v1/rag/chunks/{document_id}`             | Xem chunks theo document        | Chỉ có dữ liệu khi document đã indexed     |
| GET    | `/api/v1/rag/entities/{workspace_id}`          | KG entities                     | Filter `search`, `entity_type`, `limit`    |
| GET    | `/api/v1/rag/relationships/{workspace_id}`     | KG relationships                | Filter `entity`, `limit`                   |
| GET    | `/api/v1/rag/graph/{workspace_id}`             | Payload graph cho frontend      | Hỗ trợ `center`, `max_depth`, `max_nodes`  |
| GET    | `/api/v1/rag/analytics/{workspace_id}`         | Analytics tổng hợp              | Stats + KG + breakdown theo document       |
| GET    | `/api/v1/rag/chat/{workspace_id}/history`      | Lịch sử chat                    |                                             |
| DELETE | `/api/v1/rag/chat/{workspace_id}/history`      | Xoá lịch sử chat                |                                             |
| DELETE | `/api/v1/rag/workspace/{workspace_id}/vector-store` | Xoá toàn bộ dữ liệu workspace | Xoá documents, upload files, vector store, KG |
| POST   | `/api/v1/rag/chat/{workspace_id}/rate`         | Đánh giá source citation        | rating: `relevant/partial/not_relevant`    |
| POST   | `/api/v1/rag/chat/{workspace_id}`              | Chat non-stream                 | Trả answer + sources + image refs + thinking |
| POST   | `/api/v1/rag/chat/{workspace_id}/stream`       | Chat streaming SSE              | Event: `status`, `thinking`, `token`, ...  |
| GET    | `/api/v1/rag/capabilities`                     | Khả năng model                  | thinking + vision + mặc định thinking      |
| POST   | `/api/v1/rag/debug-chat/{workspace_id}`        | Debug retrieval/prompt/answer   | Hỗ trợ tune chất lượng                     |

## 4. Retrieval Modes (Chế độ truy vấn)

- `hybrid`: Kết hợp vector + KG context, có rerank (mặc định khuyến nghị)
- `vector_only`: Chỉ vector similarity
- `naive`: Chế độ cơ bản của KG backend
- `local`: KG query thiên về neighborhood local
- `global`: KG query thiên về global context

Ghi chú: endpoint chat hiện dùng `hybrid` cho chất lượng trả lời ổn định hơn.

## 5. Request Fields hay dùng

| Field             | Kiểu           | Ghi chú                                              |
| ----------------- | -------------- | ---------------------------------------------------- |
| `workspace_id`    | int (path)     | Định danh knowledge base/workspace                   |
| `document_id`     | int (path)     | Định danh document                                   |
| `question`        | string         | Dùng cho `/rag/query/{workspace_id}`                 |
| `message`         | string         | Dùng cho `/rag/chat/{workspace_id}`                  |
| `history`         | list           | Lịch sử hội thoại trước đó cho chat                  |
| `document_ids`    | list[int]      | Giới hạn retrieval trên tập document chỉ định        |
| `metadata_filter` | dict           | Bộ lọc metadata cho query endpoint                   |
| `top_k`           | int            | Số chunk muốn lấy                                    |
| `mode`            | string         | Chế độ retrieval (`hybrid`, `vector_only`, ...)      |
| `file`            | multipart file | File upload document                                 |
| `custom_metadata` | string(JSON)   | Metadata tuỳ chỉnh khi upload                        |
| `enable_thinking` | bool           | Bật/tắt thinking ở endpoint chat                     |
| `force_search`    | bool           | Ép pre-search trước khi trả lời trong chat           |
| `source_index`    | string         | ID source citation (ví dụ `a3x9`) cho endpoint rate  |
| `rating`          | string         | `relevant` / `partial` / `not_relevant`              |

## 6. Biến môi trường quan trọng (Environment Variables)

```bash
# Database
DATABASE_URL=postgresql+asyncpg://anhquan:anhquandeptrai@localhost:5433/graprag

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8002

# LLM Provider: "gemini" | "ollama"
LLM_PROVIDER=ollama
LLM_TIMEOUT=720

# Ollama (đang dùng)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b
OLLAMA_ENABLE_THINKING=false

# Gemini (tuỳ chọn)
# GOOGLE_AI_API_KEY=your-gemini-api-key
# LLM_MODEL_FAST=gemini-2.5-flash
# LLM_THINKING_LEVEL=medium
# LLM_MAX_OUTPUT_TOKENS=8192

# KG extraction backend: "llm" | "specialized"
KG_EXTRACTION_METHOD=specialized
NEXUSRAG_KG_GLINER_MODEL=urchade/gliner_multi-v2.1
NEXUSRAG_KG_RELATION_MODEL=Babelscape/mrebel-large

# KG Embedding
KG_EMBEDDING_PROVIDER=ollama
KG_EMBEDDING_MODEL=nomic-embed-text:latest
KG_EMBEDDING_DIMENSION=768
# (hỗ trợ thêm `sentence_transformers`)

# NexusRAG pipeline
NEXUSRAG_ENABLED=true
NEXUSRAG_ENABLE_KG=true
NEXUSRAG_KG_LANGUAGE=Vietnamese
NEXUSRAG_KG_ENTITY_TYPES=["Organization","Person","Product","Location","Event","Financial_Metric","Technology","Date","Regulation"]

# Parser: "docling" (mặc định) | "marker"
NEXUSRAG_DOCUMENT_PARSER=docling
# NEXUSRAG_MARKER_USE_LLM=false

NEXUSRAG_ENABLE_FORMULA_ENRICHMENT=true
NEXUSRAG_ENABLE_IMAGE_EXTRACTION=false
NEXUSRAG_ENABLE_IMAGE_CAPTIONING=false
NEXUSRAG_ENABLE_TABLE_CAPTIONING=false

NEXUSRAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
NEXUSRAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
NEXUSRAG_CHUNK_MAX_TOKENS=500
NEXUSRAG_KG_CHUNK_TOKEN_SIZE=500
NEXUSRAG_VECTOR_PREFETCH=20
NEXUSRAG_RERANKER_TOP_K=7
NEXUSRAG_MIN_RELEVANCE_SCORE=0.15

# Timeout tự recover document bị treo
# NEXUSRAG_PROCESSING_TIMEOUT_MINUTES=10

# Deduplication
NEXUSRAG_DEDUP_ENABLED=true
NEXUSRAG_DEDUP_MIN_CHUNK_LENGTH=50
NEXUSRAG_DEDUP_NEAR_THRESHOLD=0.85

# CORS
CORS_ORIGINS=["http://localhost:5174","http://localhost:3000"]
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

# 4) Chạy PostgreSQL + Chroma bằng Docker (khuyến nghị)
docker compose up -d db chroma

# 5) Chạy server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI:

- `http://localhost:8000/docs`

ReDoc:

- `http://localhost:8000/redoc`

## 8. Lệnh vận hành hữu ích

```bash
# Tải trước model local (đặc biệt hữu ích khi dùng specialized KG extractor)
python scripts/download_models.py

# Dọn sạch cache vector + KG
python scripts/clean.py

# Đánh giá RAG
python scripts/eval_rag.py

# Đánh giá synthetic theo Ragas
python scripts/eval_ragas_synthetic.py
```

## 9. Lưu ý

1. Upload và process là 2 bước tách riêng: upload xong cần gọi process/reindex để index.
2. Streaming endpoint `/api/v1/rag/chat/{workspace_id}/stream` trả SSE events thời gian thực; các event gồm `status`, `thinking`, `sources`, `images`, `token`, `token_rollback`, `complete`, `error`.
3. Nếu dùng Gemini (LLM hoặc KG embedding), bắt buộc set `GOOGLE_AI_API_KEY`.
4. Static ảnh tài liệu được mount qua `/static/doc-images` (nguồn từ `server/data/docling`).
5. App có cơ chế auto-recover document bị treo ở trạng thái processing sau timeout cấu hình.
6. Mặc định startup có thể auto tạo bảng (`AUTO_CREATE_TABLES=true`), nhưng production nên quản lý migration chủ động.
