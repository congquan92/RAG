## 1. Tong quan (Overview)

| Muc            | Mo ta ngan                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------- |
| Tech stack     | FastAPI, SQLAlchemy Async + SQLite, ChromaDB, LightRAG, LangChain, Ollama/Gemini, Streamlit  |
| Vai tro        | Upload tai lieu, ingestion + chunking, truy van RAG (sync + streaming), quan ly lich su chat |
| Duong dan API  | `http://localhost:8000/api/v1`                                                               |
| Health check   | `http://localhost:8000/health`                                                               |
| Vector backend | ChromaDB local (`CHROMA_PERSIST_DIR`), ket hop LightRAG graph retrieval                      |

> Note thuc te: Neu backend chay trong Docker, can map duoc endpoint Ollama (`OLLAMA_BASE_URL`) de server goi model local.

## 2. Cau truc chinh (Architecture)

| Thu muc/File                       | Vai tro                                                                      |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `main.py`                          | App bootstrap, lifespan startup/shutdown, mount routers, run server          |
| `app/api/v1/chat_router.py`        | Chat session CRUD, chat sync va SSE stream                                   |
| `app/api/v1/document_router.py`    | Upload document, poll ingestion status, document CRUD                        |
| `app/api/deps.py`                  | Dependency injection (`get_db`, `get_embeddings`, `get_reranker`, `get_llm`) |
| `app/services/chat_service.py`     | Dieu phoi chat flow: luu message -> retrieve -> generate -> citations        |
| `app/services/document_service.py` | Save file, background ingestion, luu chunk vao ChromaDB                      |
| `app/rag/ingestion.py`             | Parse tai lieu (PyMuPDF/docling), chunking bang tiktoken                     |
| `app/rag/retriever.py`             | Tri-search (semantic + keyword + graph) va reranking                         |
| `app/rag/generator.py`             | Sinh cau tra loi sync/stream va trich citations                              |
| `app/core/settings.py`             | Cau hinh trung tam doc tu `.env` (strict mode)                               |
| `app/core/llm_factory.py`          | Hot-swap LLM/Embeddings theo provider trong `.env`                           |
| `app/core/database.py`             | Async engine/session factory, init/dispose DB                                |
| `scripts/check_env.py`             | Kiem tra bien moi truong truoc khi startup                                   |
| `admin_ui/app.py`                  | Dashboard Streamlit: overview, documents, chat history, evaluation           |

> Pro-tip: Tach API layer va Service layer giup logic RAG khong bi dinh chat vao HTTP handler, rat de test va mo rong.

## 3. API Endpoints

| Method | Path                                 | Muc dich                      | Ghi chu nhanh                               |
| ------ | ------------------------------------ | ----------------------------- | ------------------------------------------- |
| GET    | `/health`                            | Kiem tra trang thai he thong  | Tra ve provider, trang thai model load      |
| POST   | `/api/v1/chat/sessions`              | Tao session chat moi          | Body: `title` (optional)                    |
| GET    | `/api/v1/chat/sessions`              | Liet ke sessions              | Ho tro `skip`, `limit`                      |
| GET    | `/api/v1/chat/sessions/{session_id}` | Chi tiet session + messages   | Kem lich su message va citations            |
| DELETE | `/api/v1/chat/sessions/{session_id}` | Xoa session                   | Cascade xoa messages                        |
| POST   | `/api/v1/chat`                       | Chat non-stream               | Body: `session_id`, `query`, `stream=false` |
| POST   | `/api/v1/chat/stream`                | Chat streaming SSE            | Yield token/citations/done theo event       |
| POST   | `/api/v1/documents/upload`           | Upload va queue ingestion     | Multipart file, tra `task_id` ngay          |
| GET    | `/api/v1/documents/status/{task_id}` | Poll status ingestion         | `pending/processing/completed/failed`       |
| GET    | `/api/v1/documents`                  | Liet ke documents             | Ho tro `skip`, `limit`                      |
| GET    | `/api/v1/documents/{document_id}`    | Chi tiet document             | Metadata + chunk count                      |
| DELETE | `/api/v1/documents/{document_id}`    | Xoa document + cleanup vector | Xoa file va chunks trong ChromaDB           |

## 4. Retrieval Modes (Che do truy van)

- `semantic`: Tim theo vector similarity trong ChromaDB.
- `keyword`: Tim theo BM25 tren corpus da index (hieu qua cho keyword/ma loi/ten rieng).
- `graph`: Tim theo quan he tri thuc tu LightRAG.
- `tri-search + rerank`: Gom 3 nguon tren, deduplicate, sau do rerank bang FlashRank de lay top-k cuoi.

## 5. Request Fields hay dung

| Field           | Kieu           | Ghi chu                                                   |
| --------------- | -------------- | --------------------------------------------------------- |
| `session_id`    | string         | Dinh danh session chat de luu lich su va truy van context |
| `query`         | string         | Cau hoi user gui vao RAG                                  |
| `stream`        | bool           | `true` dung endpoint stream; `false` dung endpoint sync   |
| `title`         | string         | Tieu de session khi tao chat session                      |
| `file`          | multipart file | File tai lieu upload de ingestion                         |
| `skip`, `limit` | int            | Pagination cho list sessions/documents                    |

## 6. Bien moi truong quan trong (Environment Variables)

### Provider va model

- `LLM_PROVIDER`: `ollama` hoac `gemini`
- `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `GEMINI_API_KEY`, `GEMINI_MODEL`
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`

### Hardware va runtime

- `EMBEDDING_DEVICE`, `DOCLING_DEVICE`, `RERANKER_DEVICE`
- `API_HOST`, `API_PORT`, `API_RELOAD`
- `CORS_ORIGINS`

### Data va retrieval

- `DATABASE_URL`
- `CHROMA_PERSIST_DIR`, `CHROMA_COLLECTION_NAME`
- `LIGHTRAG_WORKING_DIR`, `UPLOAD_DIR`
- `MAX_UPLOAD_SIZE_MB`, `ALLOWED_UPLOAD_MIME_TYPES`
- `RETRIEVAL_TOP_K`, `RERANKER_TOP_K`, `RERANKER_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`

### Observability

- `PHOENIX_ENABLED`, `PHOENIX_PORT`

> Luu y: `app/core/settings.py` dang bat strict mode. Neu thieu bien bat buoc trong `.env`, app se fail startup voi danh sach key thieu ro rang.

## 7. Setup nhanh (Quick Start)

```bash
# 1) Tao va kich hoat virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 2) Cai dependency
pip install -r requirements.txt

# 3) Tao file env
cp .env.example .env
# chinh sua .env theo may cua ban

# 4) Kiem tra env
python scripts/check_env.py

# 5) Chay server (tu dong check .env truoc khi boot)
python3 main.py
```

Swagger UI:

- `http://localhost:8000/docs`

ReDoc:

- `http://localhost:8000/redoc`

## 8. Lenh van hanh huu ich

```bash
# Chay admin dashboard
streamlit run admin_ui/app.py --server.port 8501

# Chay test API controller
python scripts/test_api.py

# Chay test settings
python scripts/test_settings.py

# Chay test evaluator
python scripts/test_evaluator.py
```

## 9. Luu y van hanh (Operational Notes)

1. Ingestion document chay background task: upload xong co task id ngay, can poll status de biet da completed chua.
2. SSE endpoint (`/api/v1/chat/stream`) tra event token theo thoi gian thuc; can xu ly stream phia client.
3. Neu dung `gemini` cho LLM hoac embeddings, bat buoc set `GEMINI_API_KEY`.
4. Neu `flashrank` khong available, retriever van chay nhung fallback sort theo score ban dau.
5. ChromaDB dang local-first; scale multi-instance can tinh toan strategy luu tru chia se.

---

Backend nay duoc thiet ke theo huong modular va env-driven: doi provider/model/device/chunking chu yeu qua `.env`, han che sua code business logic.
