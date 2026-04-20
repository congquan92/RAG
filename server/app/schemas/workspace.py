"""
Schema Knowledge Base (Workspace) cho request/response validation.
"""
from pydantic import BaseModel, Field, model_validator
from datetime import datetime


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    kg_language: str | None = None
    kg_entity_types: list[str] | None = None
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def validate_chunk_params(self):
        if self.chunk_size is not None and self.chunk_overlap is not None:
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class WorkspaceUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    system_prompt: str | None = None
    kg_language: str | None = None
    kg_entity_types: list[str] | None = None
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)

    @model_validator(mode="after")
    def validate_chunk_params(self):
        if self.chunk_size is not None and self.chunk_overlap is not None:
            if self.chunk_overlap >= self.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    description: str | None
    system_prompt: str | None = None
    kg_language: str | None = None
    kg_entity_types: list[str] | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    document_count: int = 0
    indexed_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceSummary(BaseModel):
    """Tóm tắt gọn cho dropdown selectors."""
    id: int
    name: str
    document_count: int = 0

    model_config = {"from_attributes": True}
