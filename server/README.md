## 1. Tổng quan (Overview)

| Mục            | Mô tả ngắn                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------- |
| Tech stack     | FastAPI, SQLAlchemy Async + SQLite, ChromaDB, LightRAG, LangChain, Ollama/Gemini, Streamlit  |
| Vai trò        | Upload tài liệu, ingestion + chunking, truy vấn RAG (sync + streaming), quản lý lịch sử chat |
| Đường dẫn API  | `http://localhost:8000/api/v1`                                                               |
| Health check   | `http://localhost:8000/health`                                                               |
| Vector backend | ChromaDB local (`CHROMA_PERSIST_DIR`), kết hợp LightRAG graph retrieval                      |

> Note thực tế: Nếu backend chạy trong Docker, cần map được endpoint Ollama (`OLLAMA_BASE_URL`) để server gọi model local.

## 2. Cấu trúc chính (Architecture)

| Thư mục/File                       | Vai trò                                                                      |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                          | App bootstrap, lifespan startup/shutdown, mount routers, run server          |
| `app/api/v1/chat_router.py`        | Chat session CRUD, chat sync và SSE stream                                   |
| `app/api/v1/document_router.py`    | Upload document, poll ingestion status, document CRUD                        |
| `app/api/deps.py`                  | Dependency injection (`get_db`, `get_embeddings`, `get_reranker`, `get_llm`) |
| `app/services/chat_service.py`     | Điều phối chat flow: lưu message -> retrieve -> generate -> citations        |
| `app/services/document_service.py` | Save file, background ingestion, lưu chunk vào ChromaDB                      |
| `app/rag/ingestion.py`             | Parse tài liệu (PyMuPDF/docling), chunking bằng tiktoken                     |
| `app/rag/retriever.py`             | Tri-search (semantic + keyword + graph) và reranking                         |
| `app/rag/generator.py`             | Sinh câu trả lời sync/stream và trích citations                              |
| `app/core/settings.py`             | Cấu hình trung tâm đọc từ `.env` (strict mode)                               |
| `app/core/llm_factory.py`          | Hot-swap LLM/Embeddings theo provider trong `.env`                           |
| `app/core/database.py`             | Async engine/session factory, init/dispose DB                                |
| `scripts/check_env.py`             | Kiểm tra biến môi trường trước khi startup                                   |
| `admin_ui/app.py`                  | Dashboard Streamlit: overview, documents, chat history, evaluation           |

> Pro-tip: Tách API layer và Service layer giúp logic RAG không bị dính chặt vào HTTP handler, rất dễ test và mở rộng.

## 3. API Endpoints

| Method | Path                                 | Mục đích                      | Ghi chú nhanh                               |
| ------ | ------------------------------------ | ----------------------------- | ------------------------------------------- |
| GET    | `/health`                            | Kiểm tra trạng thái hệ thống  | Trả về provider, trạng thái model load      |
| POST   | `/api/v1/chat/sessions`              | Tạo session chat mới          | Body: `title` (optional)                    |
| GET    | `/api/v1/chat/sessions`              | Liệt kê sessions              | Hỗ trợ `skip`, `limit`                      |
| GET    | `/api/v1/chat/sessions/{session_id}` | Chi tiết session + messages   | Kèm lịch sử message và citations            |
| DELETE | `/api/v1/chat/sessions/{session_id}` | Xóa session                   | Cascade xóa messages                        |
| POST   | `/api/v1/chat`                       | Chat non-stream               | Body: `session_id`, `query`, `stream=false` |
| POST   | `/api/v1/chat/stream`                | Chat streaming SSE            | Yield token/citations/done theo event       |
| POST   | `/api/v1/documents/upload`           | Upload và queue ingestion     | Multipart file, trả `task_id` ngay          |
| GET    | `/api/v1/documents/status/{task_id}` | Poll status ingestion         | `pending/processing/completed/failed`       |
| GET    | `/api/v1/documents`                  | Liệt kê documents             | Hỗ trợ `skip`, `limit`                      |
| GET    | `/api/v1/documents/{document_id}`    | Chi tiết document             | Metadata + chunk count                      |
| DELETE | `/api/v1/documents/{document_id}`    | Xóa document + cleanup vector | Xóa file và chunks trong ChromaDB           |

## 4. Retrieval Modes (Chế độ truy vấn)

- `semantic`: Tìm theo vector similarity trong ChromaDB.
- `keyword`: Tìm theo BM25 trên corpus đã index (hiệu quả cho keyword/mã lỗi/tên riêng).
- `graph`: Tìm theo quan hệ tri thức từ LightRAG.
- `tri-search + rerank`: Gom 3 nguồn trên, deduplicate, sau đó rerank bằng FlashRank để lấy top-k cuối.

## 5. Request Fields hay dùng

| Field           | Kiểu           | Ghi chú                                                   |
| --------------- | -------------- | --------------------------------------------------------- |
| `session_id`    | string         | Định danh session chat để lưu lịch sử và truy vấn context |
| `query`         | string         | Câu hỏi user gửi vào RAG                                  |
| `stream`        | bool           | `true` dùng endpoint stream; `false` dùng endpoint sync   |
| `title`         | string         | Tiêu đề session khi tạo chat session                      |
| `file`          | multipart file | File tài liệu upload để ingestion                         |
| `skip`, `limit` | int            | Pagination cho list sessions/documents                    |

## 6. Biến môi trường quan trọng (Environment Variables)

### Provider và model

- `LLM_PROVIDER`: `ollama` hoặc `gemini`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`

### Hardware và runtime

- `EMBEDDING_DEVICE`, `DOCLING_DEVICE`, `RERANKER_DEVICE`
- `API_HOST`, `API_PORT`, `API_RELOAD`
- `CORS_ORIGINS`

### Data và retrieval

- `DATABASE_URL`
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`
- `LIGHTRAG_WORKING_DIR`, `UPLOAD_DIR`
- `MAX_UPLOAD_SIZE_MB`, `ALLOWED_UPLOAD_MIME_TYPES`
- `RETRIEVAL_TOP_K`, `RERANKER_TOP_K`, `RERANKER_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

### Observability

- `PHOENIX_ENABLED`, `PHOENIX_PORT`

> Lưu ý: `app/core/settings.py` đang bật strict mode. Nếu thiếu biến bắt buộc trong `.env`, app sẽ fail startup với danh sách key thiếu rõ ràng.

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

# 4) Kiểm tra env
python scripts/check_env.py

# 5) Chạy server (tự động check .env trước khi boot)
python3 main.py
```

Swagger UI:

- `http://localhost:8000/docs`

ReDoc:

- `http://localhost:8000/redoc`

## 8. Lệnh vận hành hữu ích

```bash
# Chạy admin dashboard
streamlit run admin_ui/app.py --server.port 8501

# Chạy test API controller
python scripts/test_api.py

# Chạy test settings
python scripts/test_settings.py

# Chạy test evaluator
python scripts/test_evaluator.py
```

## 9. Lưu ý vận hành (Operational Notes)

1. Ingestion document chạy background task: upload xong có task id ngay, cần poll status để biết đã completed chưa.
2. SSE endpoint (`/api/v1/chat/stream`) trả event token theo thời gian thực; cần xử lý stream phía client.
3. Nếu dùng `gemini` cho LLM hoặc embeddings, bắt buộc set `GEMINI_API_KEY`.
4. Nếu `flashrank` không available, retriever vẫn chạy nhưng fallback sort theo score ban đầu.
5. ChromaDB đang local-first; scale multi-instance cần tính toán strategy lưu trữ chia sẻ.

---

Backend này được thiết kế theo hướng modular và env-driven: đổi provider/model/device/chunking chủ yếu qua `.env`, hạn chế sửa code business logic.
