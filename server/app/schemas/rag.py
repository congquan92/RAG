"""
Pydantic schema liên quan đến RAG dùng để validate request/response.
"""
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class RAGQueryRequest(BaseModel):
    """Schema yêu cầu cho endpoint RAG query."""
    question: str = Field(..., min_length=1, max_length=1000, description="Câu hỏi cần truy vấn")
    top_k: int = Field(default=5, ge=1, le=20, description="Số lượng chunk cần truy xuất")
    document_ids: list[int] | None = Field(default=None, description="Lọc theo document ID cụ thể")
    metadata_filter: dict | None = Field(default=None, description="Bộ lọc metadata tùy chọn cho vector search")
    mode: str = Field(
        default="hybrid",
        description="Chế độ search: hybrid (mặc định), vector_only, naive, local, global"
    )


class CitationResponse(BaseModel):
    """Một trích dẫn nguồn."""
    source_file: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = []
    formatted: str = ""


class RetrievedChunkResponse(BaseModel):
    """Schema phản hồi cho một chunk được truy xuất."""
    content: str
    chunk_id: str
    score: float
    metadata: dict
    citation: CitationResponse | None = None

    model_config = {"from_attributes": True}


class DocumentImageResponse(BaseModel):
    """Schema phản hồi cho ảnh tài liệu."""
    image_id: str
    document_id: int
    page_no: int
    caption: str = ""
    width: int = 0
    height: int = 0
    url: str = ""


class RAGQueryResponse(BaseModel):
    """Schema phản hồi cho RAG query."""
    query: str
    chunks: list[RetrievedChunkResponse]
    context: str
    total_chunks: int
    knowledge_graph_summary: str = ""
    citations: list[CitationResponse] = []
    image_refs: list[DocumentImageResponse] = []


class DocumentProcessRequest(BaseModel):
    """Schema yêu cầu cho xử lý tài liệu."""
    document_id: int


class DocumentProcessResponse(BaseModel):
    """Schema phản hồi cho xử lý tài liệu."""
    document_id: int
    status: str
    chunk_count: int
    message: str


class BatchProcessRequest(BaseModel):
    """Schema yêu cầu cho xử lý tài liệu hàng loạt."""
    document_ids: list[int] = Field(..., min_length=1, description="Danh sách document ID cần xử lý")


class ProjectRAGStatsResponse(BaseModel):
    """Schema phản hồi cho thống kê RAG của workspace."""
    workspace_id: int
    total_documents: int
    indexed_documents: int
    total_chunks: int
    image_count: int = 0
    nexusrag_documents: int = 0


# ---------------------------------------------------------------------------
# Schema cho Knowledge Graph
# ---------------------------------------------------------------------------

class KGEntityResponse(BaseModel):
    """Một entity (node) trong Knowledge Graph."""
    name: str
    entity_type: str = "Unknown"
    description: str = ""
    degree: int = 0  # số lượng relationship


class KGRelationshipResponse(BaseModel):
    """Một relationship (edge) trong Knowledge Graph."""
    source: str
    target: str
    description: str = ""
    keywords: str = ""
    weight: float = 1.0


class KGGraphNodeResponse(BaseModel):
    """Node trong payload visualization của graph."""
    id: str
    label: str
    entity_type: str = "Unknown"
    degree: int = 0


class KGGraphEdgeResponse(BaseModel):
    """Edge trong payload visualization của graph."""
    source: str
    target: str
    label: str = ""
    weight: float = 1.0


class KGGraphResponse(BaseModel):
    """Dữ liệu export graph đầy đủ cho frontend visualization."""
    nodes: list[KGGraphNodeResponse] = []
    edges: list[KGGraphEdgeResponse] = []
    is_truncated: bool = False


class KGAnalyticsResponse(BaseModel):
    """Tóm tắt analytics của Knowledge Graph."""
    entity_count: int = 0
    relationship_count: int = 0
    entity_types: dict[str, int] = {}  # type -> count
    top_entities: list[KGEntityResponse] = []  # top N theo degree
    avg_degree: float = 0.0


class DocumentBreakdownItem(BaseModel):
    """Thông tin breakdown theo từng tài liệu cho analytics."""
    document_id: int
    filename: str
    chunk_count: int = 0
    image_count: int = 0
    page_count: int = 0
    file_size: int = 0
    status: str = "pending"


class ProjectAnalyticsResponse(BaseModel):
    """Analytics mở rộng của project."""
    stats: ProjectRAGStatsResponse
    kg_analytics: KGAnalyticsResponse | None = None
    document_breakdown: list[DocumentBreakdownItem] = []


# ---------------------------------------------------------------------------
# Schema cho Chat
# ---------------------------------------------------------------------------

class ChatMessageSchema(BaseModel):
    """Một chat message trong conversation history."""
    role: str = Field(..., description="user hoặc assistant")
    content: str


class ChatRequest(BaseModel):
    """Yêu cầu cho chat endpoint."""
    message: str = Field(..., min_length=1, max_length=5000)
    history: list[ChatMessageSchema] = []
    document_ids: list[int] | None = None
    enable_thinking: bool = False
    force_search: bool = False  # Pre-search trước khi gọi LLM; inject trực tiếp sources vào context


class ChatSourceChunk(BaseModel):
    """Một source chunk được tham chiếu trong câu trả lời chat."""
    index: str  # ID alphanumeric 4 ký tự, ví dụ "a3x9" (trước đây: int)
    chunk_id: str

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v
    content: str
    document_id: int
    page_no: int = 0
    heading_path: list[str] = []
    score: float = 0.0
    source_type: str = "vector"  # "vector" | "kg"


class ChatImageRef(BaseModel):
    """Một ảnh được tham chiếu trong câu trả lời chat."""
    ref_id: str | None = None  # ID alphanumeric 4 ký tự, ví dụ "p4f2"
    image_id: str
    document_id: int
    page_no: int = 0
    caption: str = ""
    url: str = ""
    width: int = 0
    height: int = 0


class ChatResponse(BaseModel):
    """Phản hồi từ chat endpoint."""
    answer: str
    sources: list[ChatSourceChunk] = []
    related_entities: list[str] = []
    kg_summary: str | None = None
    image_refs: list[ChatImageRef] = []
    thinking: str | None = None


class PersistedChatMessage(BaseModel):
    """Chat message đã được lưu trong database."""
    id: int
    message_id: str
    role: str
    content: str
    sources: list[ChatSourceChunk] | None = None
    related_entities: list[str] | None = None
    image_refs: list[ChatImageRef] | None = None
    thinking: str | None = None
    agent_steps: list | None = None
    created_at: str  # định dạng ISO

    model_config = {"from_attributes": True}


class ChatHistoryResponse(BaseModel):
    """Response cho API GET chat history."""
    workspace_id: int
    messages: list[PersistedChatMessage]
    total: int


class RateSourceRequest(BaseModel):
    """Request để đánh giá một source citation."""
    message_id: str = Field(..., description="message_id chứa source")
    source_index: str = Field(..., description="ID source citation, ví dụ 'a3x9'")
    rating: Literal["relevant", "partial", "not_relevant"] = Field(
        ..., description="Mức đánh giá source"
    )


class RateSourceResponse(BaseModel):
    """Response sau khi đánh giá source."""
    success: bool
    message_id: str
    ratings: dict[str, str]


class LLMCapabilitiesResponse(BaseModel):
    """Response cho kiểm tra capabilities của LLM."""
    provider: str
    model: str
    supports_thinking: bool
    supports_vision: bool
    thinking_default: bool = True


# ---------------------------------------------------------------------------
# Schema cho Debug / QA
# ---------------------------------------------------------------------------

class DebugRetrievedSource(BaseModel):
    """Một source đã truy xuất để debug inspection."""
    index: str  # ID alphanumeric 4 ký tự (trước đây: int)
    document_id: int

    @field_validator("index", mode="before")
    @classmethod
    def coerce_index_to_str(cls, v):
        return str(v) if not isinstance(v, str) else v
    page_no: int
    heading_path: list[str] = []
    source_file: str = ""
    content_preview: str = ""  # 500 ký tự đầu tiên
    score: float = 0.0
    source_type: str = "vector"


class DebugChatResponse(BaseModel):
    """Debug response đầy đủ: retrieval + câu trả lời LLM để quality inspection."""
    # Truy vấn
    question: str
    workspace_id: int

    # Truy xuất
    retrieved_sources: list[DebugRetrievedSource] = []
    kg_summary: str = ""
    total_sources: int = 0

    # LLM
    system_prompt: str = ""
    answer: str = ""
    thinking: str | None = None

    # Ảnh
    image_count: int = 0

    # Metadata
    provider: str = ""
    model: str = ""
