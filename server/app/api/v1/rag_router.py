"""
RAG Graph Router — endpoints for graph visualization and analytics panels.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.schemas.rag_schema import (
    KGEntityResponse,
    KGGraphResponse,
    KGRelationshipResponse,
    ProjectAnalyticsResponse,
)
from app.services import rag_graph_service

router = APIRouter(prefix="/rag", tags=["RAG Graph"])


@router.get(
    "/graph/{workspace_id}",
    response_model=KGGraphResponse,
    summary="Lấy dữ liệu graph cho KG canvas",
)
async def get_graph(
    workspace_id: str,
    max_nodes: int = Query(default=150, ge=20, le=500),
    max_depth: int = Query(default=3, ge=1, le=5),
    db: AsyncSession = Depends(get_db),
):
    return await rag_graph_service.get_graph_data(
        db=db,
        workspace_id=workspace_id,
        max_nodes=max_nodes,
        _max_depth=max_depth,
    )


@router.get(
    "/entities/{workspace_id}",
    response_model=list[KGEntityResponse],
    summary="Liệt kê entities nổi bật",
)
async def get_entities(
    workspace_id: str,
    limit: int = Query(default=500, ge=10, le=1000),
    db: AsyncSession = Depends(get_db),
):
    return await rag_graph_service.get_entities(
        db=db,
        workspace_id=workspace_id,
        limit=limit,
    )


@router.get(
    "/relationships/{workspace_id}",
    response_model=list[KGRelationshipResponse],
    summary="Liệt kê quan hệ của một entity",
)
async def get_relationships(
    workspace_id: str,
    entity: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await rag_graph_service.get_relationships(
        db=db,
        workspace_id=workspace_id,
        entity=entity,
        limit=limit,
    )


@router.get(
    "/analytics/{workspace_id}",
    response_model=ProjectAnalyticsResponse,
    summary="Thống kê tổng quan tài liệu + knowledge graph",
)
async def get_analytics(
    workspace_id: str,
    db: AsyncSession = Depends(get_db),
):
    return await rag_graph_service.get_project_analytics(
        db=db,
        workspace_id=workspace_id,
    )
