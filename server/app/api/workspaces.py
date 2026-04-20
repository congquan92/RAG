"""
Các CRUD API endpoint cho Knowledge Base (Workspace).
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import logging
from app.core.deps import get_db
from app.core.exceptions import NotFoundError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document, DocumentStatus
from app.schemas.workspace import (
    WorkspaceCreate,
    WorkspaceUpdate,
    WorkspaceResponse,
    WorkspaceSummary,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])
logger = logging.getLogger(__name__)

async def _enrich_response(db: AsyncSession, kb: KnowledgeBase) -> WorkspaceResponse:
    """Tạo WorkspaceResponse với các số lượng đã tính."""
    total = await db.execute(
        select(func.count(Document.id)).where(Document.workspace_id == kb.id)
    )
    indexed = await db.execute(
        select(func.count(Document.id)).where(
            Document.workspace_id == kb.id,
            Document.status == DocumentStatus.INDEXED,
        )
    )
    return WorkspaceResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        system_prompt=kb.system_prompt,
        kg_language=kb.kg_language,
        kg_entity_types=kb.kg_entity_types,
        chunk_size=kb.chunk_size,
        chunk_overlap=kb.chunk_overlap,
        document_count=total.scalar() or 0,
        indexed_count=indexed.scalar() or 0,
        created_at=kb.created_at,
        updated_at=kb.updated_at,
    )


@router.get("", response_model=list[WorkspaceResponse])
async def list_workspaces(db: AsyncSession = Depends(get_db)):
    """Liệt kê toàn bộ knowledge base."""
    result = await db.execute(
        select(KnowledgeBase).order_by(KnowledgeBase.updated_at.desc())
    )
    kbs = result.scalars().all()
    return [await _enrich_response(db, kb) for kb in kbs]


@router.post("", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: WorkspaceCreate,
    db: AsyncSession = Depends(get_db),
):
    """Tạo knowledge base mới."""
    if (
        body.chunk_size is not None
        and body.chunk_overlap is not None
        and body.chunk_overlap >= body.chunk_size
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        kg_language=body.kg_language,
        kg_entity_types=body.kg_entity_types,
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    db.add(kb)
    await db.commit()
    await db.refresh(kb)
    return await _enrich_response(db, kb)


@router.get("/summary", response_model=list[WorkspaceSummary])
async def list_workspace_summaries(db: AsyncSession = Depends(get_db)):
    """Danh sách gọn cho dropdown selectors."""
    result = await db.execute(
        select(KnowledgeBase).order_by(KnowledgeBase.name)
    )
    kbs = result.scalars().all()
    summaries = []
    for kb in kbs:
        cnt = await db.execute(
            select(func.count(Document.id)).where(Document.workspace_id == kb.id)
        )
        summaries.append(WorkspaceSummary(
            id=kb.id, name=kb.name, document_count=cnt.scalar() or 0
        ))
    return summaries


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Lấy knowledge base theo ID."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)
    return await _enrich_response(db, kb)


@router.put("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: int,
    body: WorkspaceUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Cập nhật tên/mô tả của knowledge base."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    effective_chunk_size = body.chunk_size if body.chunk_size is not None else kb.chunk_size
    effective_chunk_overlap = body.chunk_overlap if body.chunk_overlap is not None else kb.chunk_overlap
    if (
        effective_chunk_size is not None
        and effective_chunk_overlap is not None
        and effective_chunk_overlap >= effective_chunk_size
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be smaller than chunk_size",
        )

    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    if body.system_prompt is not None:
        # Chuỗi rỗng → reset về mặc định (None)
        kb.system_prompt = body.system_prompt or None
    if body.kg_language is not None:
        kb.kg_language = body.kg_language or None
    if body.kg_entity_types is not None:
        kb.kg_entity_types = body.kg_entity_types or None
    if body.chunk_size is not None:
        kb.chunk_size = body.chunk_size
    if body.chunk_overlap is not None:
        kb.chunk_overlap = body.chunk_overlap

    await db.commit()
    await db.refresh(kb)
    return await _enrich_response(db, kb)


@router.delete("/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Xóa knowledge base và toàn bộ document của nó."""
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if kb is None:
        raise NotFoundError("KnowledgeBase", workspace_id)

    # Dọn dẹp vector store và dữ liệu KG
    try:
        from app.services.vector_store import get_vector_store
        vs = get_vector_store(workspace_id)
        vs.delete_collection()
    except Exception as e:
        logger.warning(f"Failed to delete vector collection for workspace {workspace_id}: {e}")

    try:
        from app.services.knowledge_graph_service import KnowledgeGraphService
        kg = KnowledgeGraphService(workspace_id)
        await kg.delete_project_data()
    except Exception as e:
        logger.warning(f"Failed to delete KG data for workspace {workspace_id}: {e}")

    # Dọn dẹp file ảnh
    import shutil
    from app.core.config import settings
    images_dir = settings.BASE_DIR / "data" / "docling" / f"kb_{workspace_id}"
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)

    await db.delete(kb)
    await db.commit()
