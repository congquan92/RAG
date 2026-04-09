"""
RAG Graph Schemas — DTOs cho graph/entity analytics endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KGGraphNodeResponse(BaseModel):
    id: str
    label: str
    entity_type: str
    degree: int


class KGGraphEdgeResponse(BaseModel):
    source: str
    target: str
    label: str
    weight: float


class KGGraphResponse(BaseModel):
    nodes: list[KGGraphNodeResponse] = Field(default_factory=list)
    edges: list[KGGraphEdgeResponse] = Field(default_factory=list)
    is_truncated: bool = False


class KGEntityResponse(BaseModel):
    name: str
    entity_type: str
    description: str
    degree: int


class KGRelationshipResponse(BaseModel):
    source: str
    target: str
    description: str
    keywords: str
    weight: float


class KGAnalyticsResponse(BaseModel):
    entity_count: int
    relationship_count: int
    entity_types: dict[str, int]
    top_entities: list[KGEntityResponse]
    avg_degree: float


class DocumentBreakdownResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    image_count: int
    page_count: int
    file_size: int
    status: str


class RAGStatsResponse(BaseModel):
    workspace_id: str
    total_documents: int
    indexed_documents: int
    total_chunks: int
    image_count: int
    nexusrag_documents: int


class ProjectAnalyticsResponse(BaseModel):
    stats: RAGStatsResponse
    kg_analytics: KGAnalyticsResponse | None = None
    document_breakdown: list[DocumentBreakdownResponse] = Field(default_factory=list)
