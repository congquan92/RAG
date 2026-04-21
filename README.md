<div align="center">

# RAG + Knowledge Graph

**Nền tảng truy vấn tài liệu thông minh** kết hợp Vector Search + Knowledge Graph + LLM Chat Streaming,
phù hợp cho xây dựng hệ thống tra cứu tri thức nội bộ theo mô hình Client-Server.

<br/>

[![React](https://img.shields.io/badge/React-19-20232A?style=for-the-badge&logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178C6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-7-646CFF?style=for-the-badge&logo=vite)](https://vite.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?style=for-the-badge&logo=postgresql)](https://www.postgresql.org/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B6B?style=for-the-badge)](https://www.trychroma.com/)
[![LightRAG](https://img.shields.io/badge/LightRAG-Knowledge%20Graph-111827?style=for-the-badge)](https://github.com/HKUDS/LightRAG)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-111111?style=for-the-badge)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)

</div>

---

## Giới thiệu dự án
Là hệ thống RAG full-stack giúp bạn xây dựng **knowledge base theo workspace**,
upload tài liệu, phân tích tự động, truy vấn theo ngữ nghĩa và chat với AI có trích dẫn nguồn.

Hệ thống được thiết kế theo kiến trúc **Client-Server tách biệt**:

- **Client (React + Vite):** Quản lý workspace, tài liệu, chat, trực quan hóa Knowledge Graph.
- **Server (FastAPI):** Parse tài liệu, chunking, embedding, indexing vào ChromaDB, KG ingestion và LLM orchestration.

> Dự án hỗ trợ cả **Ollama local** và **Gemini API** cho lớp sinh câu trả lời.

---

## Tính năng nổi bật

### Dành cho người dùng

| Tính năng | Mô tả |
|---|---|
| **Workspace Management** | Tạo, sửa, xóa không gian tri thức; theo dõi số tài liệu và số tài liệu đã index |
| **Upload tài liệu đa định dạng** | Hỗ trợ `pdf`, `txt`, `md`, `docx`, `pptx`, kèm metadata tùy chỉnh |
| **Phân tích tài liệu nền** | Parse + Index từng tài liệu hoặc chạy batch, theo dõi trạng thái `pending/parsing/indexing/indexed/failed` |
| **Chat Streaming (SSE)** | Trả lời theo thời gian thực, có timeline các bước phân tích/tìm kiếm/sinh nội dung |
| **Citation tương tác** | Trích dẫn nguồn theo ID, click để nhảy tới trang/đoạn trong document viewer |
| **Image-aware answer** | Liên kết ảnh trích xuất từ tài liệu, hỗ trợ tham chiếu trực tiếp trong câu trả lời |
| **Knowledge Graph View** | Xem đồ thị thực thể-quan hệ, danh sách entity, tô sáng theo ngữ cảnh chat |
| **Analytics Dashboard** | Thống kê documents/chunks/images/entities/relationships và breakdown theo từng tài liệu |
| **Chat History bền vững** | Lưu lịch sử hội thoại theo workspace, hỗ trợ xóa toàn bộ lịch sử |

### Dành cho kỹ thuật và vận hành

| Tính năng | Mô tả |
|---|---|
| **NexusRAG Pipeline** | `Parse -> Dedup -> Embed -> Vector Index -> KG Ingest` với cơ chế orchestration rõ ràng |
| **Hybrid Retrieval + Rerank** | Kết hợp vector retrieval và KG context (`hybrid`, `vector_only`, `local`, `global`, `naive`) |
| **Workspace-level tuning** | Tùy chỉnh `chunk_size`, `chunk_overlap`, `kg_language`, `kg_entity_types` theo từng workspace |
| **Provider linh hoạt** | Chuyển đổi LLM/embedding provider giữa Ollama và Gemini bằng biến môi trường |
| **Schema bootstrap + recovery** | Tự khởi tạo schema và tự phục hồi document bị treo xử lý quá timeout |
| **Cleanup endpoint** | Xóa toàn bộ dữ liệu vector/KG/upload theo workspace để reset nhanh môi trường |

---

## Kiến trúc hệ thống

```text
RAG/
├── client/                         # Frontend (React 19 + Vite + TypeScript)
│   ├── src/
│   │   ├── pages/                 # KnowledgeBasesPage, WorkspacePage
│   │   ├── components/
│   │   │   ├── layout/            # AppShell, Sidebar, TopBar
│   │   │   └── rag/               # DataPanel, ChatPanel, VisualPanel, KG, Analytics
│   │   ├── hooks/                 # useRAGChatStream, useChatHistory, useWorkspaces
│   │   ├── stores/                # Zustand stores (theme, panel/workspace state)
│   │   └── lib/                   # API client, utility functions
│   └── vite.config.ts             # Dev server port 5174 + proxy /api, /static
│
└── server/                        # Backend (FastAPI + SQLAlchemy Async)
	├── app/
	│   ├── main.py                # FastAPI app + lifespan + CORS + static mount
	│   ├── api/                   # workspaces, documents, rag, config, chat_agent
	│   ├── core/                  # config, database, deps, exceptions
	│   ├── models/                # knowledge_bases, documents, chat_messages
	│   ├── schemas/               # Request/response models
	│   └── services/              # NexusRAG, retriever, parser, KG, vector store, LLM
	├── docker-compose.yml         # PostgreSQL + ChromaDB
	├── requirements.txt
	└── scripts/                   # clean, download_models, eval_rag, eval_ragas_synthetic
```

---

## Luồng xử lý NexusRAG

1. **Upload** tài liệu vào workspace.
2. **Parser** (Docling hoặc Marker) trích xuất markdown, ảnh, bảng, cấu trúc heading.
3. **Dedup + Chunking** để loại nhiễu và chuẩn hóa context retrieval.
4. **Embedding + Indexing** vào ChromaDB collection theo workspace.
5. **KG Ingestion** vào LightRAG để tạo entity/relationship graph.
6. **Query/Chat** dùng hybrid retrieval + reranker, trả lời kèm citation và image refs.

---

## Hướng dẫn cài đặt

### Yêu cầu hệ thống

- **Node.js** >= 18 (khuyến nghị theo `.nvmrc`: 22)
- **pnpm** >= 8
- **Python** >= 3.10
- **Docker** (để chạy PostgreSQL + ChromaDB)
- **Ollama** (nếu chạy LLM local) hoặc API key Gemini

### 1. Clone repository

```bash
git clone <repository-url>
cd RAG
```

### 2. Cài đặt backend

```bash
cd server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 3. Khởi động PostgreSQL + ChromaDB

```bash
cd server
docker compose up -d db chroma
```

Mặc định:

- PostgreSQL: `localhost:5433`
- ChromaDB: `localhost:8002`

### 4. Cấu hình model (tùy chọn Ollama local)

```bash
ollama pull qwen3.5:4b
ollama pull nomic-embed-text:latest
```

### 5. Chạy API server

```bash
cd server
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Chạy frontend

```bash
cd client
pnpm install
# tùy chọn: cp .env.example .env.local
pnpm dev
```

Ứng dụng chạy tại:

- Frontend: `http://localhost:5174`
- API: `http://localhost:8000/api/v1`
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Cấu hình môi trường quan trọng

Ví dụ các biến thường dùng trong `server/.env`:

```bash
# Database + Vector store
DATABASE_URL=postgresql+asyncpg://anhquan:anhquandeptrai@localhost:5433/graprag
CHROMA_HOST=localhost
CHROMA_PORT=8002

# LLM provider: "ollama" | "gemini"
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen3.5:4b

# NexusRAG core
NEXUSRAG_ENABLED=true
NEXUSRAG_ENABLE_KG=true
NEXUSRAG_DOCUMENT_PARSER=docling   # hoặc marker

# KG extraction
KG_EXTRACTION_METHOD=specialized
NEXUSRAG_KG_GLINER_MODEL=urchade/gliner_multi-v2.1
NEXUSRAG_KG_RELATION_MODEL=Babelscape/mrebel-large
```

Frontend có thể đặt `VITE_API_URL` khi backend không cùng origin.

---

## API chính

| Nhóm | Endpoint tiêu biểu | Mục đích |
|---|---|---|
| **System** | `GET /health`, `GET /ready` | Liveness/Readiness check |
| **Workspaces** | `GET/POST/PUT/DELETE /workspaces...` | Quản lý knowledge base/workspace |
| **Documents** | `/documents/upload/{workspace_id}`, `/documents/{id}/markdown` | Upload, xem nội dung parse, xóa tài liệu |
| **Processing** | `/rag/process/{document_id}`, `/rag/process-batch` | Phân tích/index tài liệu |
| **RAG Query** | `POST /rag/query/{workspace_id}` | Truy vấn retrieval theo mode |
| **Chat** | `POST /rag/chat/{workspace_id}/stream` | Chat SSE có citation + thinking |
| **KG/Analytics** | `/rag/graph/{workspace_id}`, `/rag/analytics/{workspace_id}` | Trực quan KG và thống kê tổng hợp |

---

## Scripts vận hành hữu ích

```bash
cd server
source .venv/bin/activate

# Tải sẵn model phụ trợ
python scripts/download_models.py

# Dọn sạch dữ liệu runtime/vector cache
python scripts/clean.py

# Đánh giá pipeline RAG
python scripts/eval_rag.py

# Đánh giá synthetic bằng Ragas
python scripts/eval_ragas_synthetic.py
```

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy để lại một star để ủng hộ!**

</div>