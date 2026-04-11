"""
RAG Graph Service — build lightweight knowledge graph views from indexed chunks.

Current repository stores indexed chunks in ChromaDB but does not persist a dedicated
entity graph table. This service derives a graph on-the-fly using chunk co-occurrence
so frontend KG panels can render useful graph/entity analytics endpoints.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.settings import settings
from app.services import document_service

logger = logging.getLogger(__name__)

# Keep stopwords compact to reduce noisy entities in EN + VI text.
_STOPWORDS = {
    "the", "and", "for", "from", "with", "this", "that", "have", "has", "are", "was", "were", "into", "onto", "about", "your", "you", "their", "they", "them", "can", "will", "would", "should", "could", "not", "but", "than", "then", "when", "where", "what", "which", "while", "because", "there", "here", "been", "being", "very", "more", "most", "each", "other", "also", "such", "using", "used", "use", "all", "any", "some", "many", "much", "into", "out", "over", "under",
    "va", "la", "cac", "cho", "mot", "nhung", "trong", "khi", "voi", "duoc", "khong", "nay", "kia", "tu", "den", "nhieu", "it", "cua", "tren", "duoi", "sau", "truoc", "nhu", "neu", "roi", "thi", "se", "dang", "da", "do", "vi", "de", "tai", "va", "nhan", "theo", "cung", "giua", "sao", "nhung", "nhung", "neu", "day", "do", "anh", "chi", "em", "ban", "toi", "chung", "ta", "minh", "no", "ho", "khi", "tuy", "moi", "vung", "nam", "thang", "ngay",
}

_CAPITAL_ENTITY_RE = re.compile(r"\b(?:[A-ZA-ZÀ-Ỵ][\w-]{1,})(?:\s+[A-ZA-ZÀ-Ỵ][\w-]{1,}){0,2}\b")
_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9_-]{2,}")


@dataclass
class _GraphBuildResult:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    frequency_by_id: dict[str, int]
    is_truncated: bool


def _normalize_entity_name(name: str) -> str:
    value = re.sub(r"\s+", " ", name).strip(" \t\n\r-_,.;:()[]{}'\"`")
    return value


def _guess_entity_type(label: str) -> str:
    lower = label.lower()
    if any(token in lower for token in ("inc", "corp", "company", "co.", "ltd", "llc", "jsc", "bank", "university", "institute", "bo", "so", "ubnd")):
        return "organization"
    if any(token in lower for token in ("city", "province", "district", "street", "vietnam", "ha noi", "hanoi", "ho chi minh", "da nang", "can tho")):
        return "location"
    if any(token in lower for token in ("conference", "summit", "workshop", "event", "le hoi", "su kien", "chien dich")):
        return "event"
    # 2-3 capitalized words are commonly person names in VN/EN corpora.
    if re.match(r"^[A-ZA-ZÀ-Ỵ][\w-]+(?:\s+[A-ZA-ZÀ-Ỵ][\w-]+){1,2}$", label):
        return "person"
    return "concept"


def _extract_entities(text: str, max_entities: int = 10) -> list[str]:
    entities: list[str] = []
    seen: set[str] = set()

    for match in _CAPITAL_ENTITY_RE.findall(text):
        normalized = _normalize_entity_name(match)
        if len(normalized) < 3:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(normalized)
        if len(entities) >= max_entities:
            return entities

    # Fallback: high-signal keywords if text has little capitalization.
    if len(entities) < 3:
        for token in _WORD_RE.findall(text.lower()):
            token = _normalize_entity_name(token)
            if len(token) < 4 or token in _STOPWORDS:
                continue
            if token.isdigit():
                continue
            if token in seen:
                continue
            seen.add(token)
            entities.append(token)
            if len(entities) >= max_entities:
                break

    return entities


def _load_indexed_chunks(
    allowed_document_ids: set[str] | None = None,
    max_chunks: int = 2500,
) -> list[tuple[str, dict[str, Any]]]:
    """Load indexed chunk texts + metadata from ChromaDB."""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        existing = [c.name for c in client.list_collections()]
        if settings.chroma_collection_name not in existing:
            return []

        collection = client.get_collection(name=settings.chroma_collection_name)
        payload = collection.get(include=["documents", "metadatas"], limit=max_chunks)
        docs = payload.get("documents") or []
        metas = payload.get("metadatas") or []

        rows: list[tuple[str, dict[str, Any]]] = []
        for i, text in enumerate(docs):
            if not text:
                continue
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}

            if allowed_document_ids is not None:
                meta_doc_id = str(meta.get("document_id", "") or "").strip()
                if meta_doc_id not in allowed_document_ids:
                    continue

            rows.append((str(text), meta))
        return rows
    except ImportError:
        logger.warning("chromadb not installed; KG endpoints return empty payloads")
        return []
    except Exception as exc:
        logger.error("Failed to load indexed chunks for graph: %s", exc)
        return []


def _build_graph(
    max_nodes: int,
    allowed_document_ids: set[str] | None = None,
) -> _GraphBuildResult:
    chunk_rows = _load_indexed_chunks(allowed_document_ids=allowed_document_ids)
    if not chunk_rows:
        return _GraphBuildResult(nodes=[], edges=[], frequency_by_id={}, is_truncated=False)

    freq = Counter()
    display_name: dict[str, str] = {}
    entity_type: dict[str, str] = {}
    edge_weights = Counter()

    for text, _meta in chunk_rows:
        entities = _extract_entities(text)
        if not entities:
            continue

        dedup_in_chunk: list[str] = []
        seen_chunk: set[str] = set()
        for ent in entities:
            key = ent.lower()
            if key in seen_chunk:
                continue
            seen_chunk.add(key)
            dedup_in_chunk.append(key)
            display_name.setdefault(key, ent)
            entity_type.setdefault(key, _guess_entity_type(ent))
            freq[key] += 1

        for a, b in combinations(sorted(dedup_in_chunk), 2):
            edge_weights[(a, b)] += 1

    if not freq:
        return _GraphBuildResult(nodes=[], edges=[], frequency_by_id={}, is_truncated=False)

    sorted_entity_ids = [entity_id for entity_id, _ in sorted(freq.items(), key=lambda item: (-item[1], item[0]))]
    kept_ids = set(sorted_entity_ids[:max_nodes])

    degrees = Counter()
    total_edge_count = 0
    for (a, b), weight in edge_weights.items():
        if a in kept_ids and b in kept_ids:
            total_edge_count += 1
            degrees[a] += 1
            degrees[b] += 1

    nodes = [
        {
            "id": entity_id,
            "label": display_name.get(entity_id, entity_id),
            "entity_type": entity_type.get(entity_id, "concept"),
            "degree": int(degrees.get(entity_id, 0)),
        }
        for entity_id in sorted(kept_ids, key=lambda entity_id: (-degrees.get(entity_id, 0), -freq.get(entity_id, 0), display_name.get(entity_id, entity_id).lower()))
    ]

    # Keep edge count proportional to node budget for performance in SVG force layout.
    max_edges = max_nodes * 4
    filtered_edges: list[dict[str, Any]] = []
    for (a, b), weight in sorted(edge_weights.items(), key=lambda item: (-item[1], item[0])):
        if a not in kept_ids or b not in kept_ids:
            continue
        filtered_edges.append({
            "source": a,
            "target": b,
            "label": "liên quan",
            "weight": float(weight),
        })
        if len(filtered_edges) >= max_edges:
            break

    is_truncated = len(freq) > max_nodes or total_edge_count > len(filtered_edges)
    frequency_by_id = {entity_id: int(freq.get(entity_id, 0)) for entity_id in kept_ids}

    return _GraphBuildResult(nodes=nodes, edges=filtered_edges, frequency_by_id=frequency_by_id, is_truncated=is_truncated)


async def get_graph_data(
    db: AsyncSession,
    workspace_id: str,
    max_nodes: int,
    _max_depth: int,
) -> dict[str, Any]:
    """
    Return graph payload for frontend KG canvas.

    Scope graph theo workspace bằng danh sách document thuộc workspace đó.
    """
    workspace_document_ids = await _get_workspace_indexed_document_ids(db, workspace_id)
    graph = _build_graph(max_nodes=max_nodes, allowed_document_ids=workspace_document_ids)
    return {
        "nodes": graph.nodes,
        "edges": graph.edges,
        "is_truncated": graph.is_truncated,
    }


async def get_entities(
    db: AsyncSession,
    workspace_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    workspace_document_ids = await _get_workspace_indexed_document_ids(db, workspace_id)
    graph = _build_graph(
        max_nodes=max(limit, 50),
        allowed_document_ids=workspace_document_ids,
    )

    entities = [
        {
            "name": node["label"],
            "entity_type": node["entity_type"],
            "description": f"Xuất hiện {graph.frequency_by_id.get(node['id'], 0)} lần trong dữ liệu đã index.",
            "degree": node["degree"],
        }
        for node in graph.nodes
    ]
    return entities[:limit]


async def get_relationships(
    db: AsyncSession,
    workspace_id: str,
    entity: str,
    limit: int,
) -> list[dict[str, Any]]:
    workspace_document_ids = await _get_workspace_indexed_document_ids(db, workspace_id)
    graph = _build_graph(max_nodes=300, allowed_document_ids=workspace_document_ids)
    if not graph.nodes:
        return []

    node_by_id = {node["id"]: node for node in graph.nodes}
    target_id = None
    lowered = entity.lower().strip()
    for node in graph.nodes:
        if node["label"].lower() == lowered:
            target_id = node["id"]
            break

    if target_id is None:
        return []

    relationships: list[dict[str, Any]] = []
    for edge in graph.edges:
        if edge["source"] != target_id and edge["target"] != target_id:
            continue

        source_node = node_by_id.get(edge["source"])
        target_node = node_by_id.get(edge["target"])
        if not source_node or not target_node:
            continue

        relationships.append(
            {
                "source": source_node["label"],
                "target": target_node["label"],
                "description": "Đồng xuất hiện trong các đoạn văn đã index",
                "keywords": edge["label"],
                "weight": edge["weight"],
            }
        )

    relationships.sort(key=lambda rel: float(rel["weight"]), reverse=True)
    return relationships[:limit]


async def get_project_analytics(
    db: AsyncSession,
    workspace_id: str,
) -> dict[str, Any]:
    documents, _total = await document_service.list_documents(
        db,
        workspace_id=workspace_id,
        skip=0,
        limit=5000,
    )

    indexed_documents = sum(1 for doc in documents if (doc.chunk_count or 0) > 0)
    total_chunks = sum(int(doc.chunk_count or 0) for doc in documents)

    workspace_document_ids = {
        str(doc.id)
        for doc in documents
        if int(doc.chunk_count or 0) > 0
    }
    graph = _build_graph(max_nodes=150, allowed_document_ids=workspace_document_ids)
    entity_types = Counter(node["entity_type"] for node in graph.nodes)

    kg_analytics: dict[str, Any] | None = None
    if graph.nodes:
        top_entities = [
            {
                "name": node["label"],
                "entity_type": node["entity_type"],
                "description": f"Xuất hiện {graph.frequency_by_id.get(node['id'], 0)} lần trong dữ liệu đã index.",
                "degree": node["degree"],
            }
            for node in graph.nodes[:10]
        ]

        avg_degree = sum(int(node["degree"]) for node in graph.nodes) / max(len(graph.nodes), 1)
        kg_analytics = {
            "entity_count": len(graph.nodes),
            "relationship_count": len(graph.edges),
            "entity_types": dict(entity_types),
            "top_entities": top_entities,
            "avg_degree": round(avg_degree, 2),
        }

    document_breakdown = []
    for doc in documents:
        latest_status = "indexed" if (doc.chunk_count or 0) > 0 else "pending"
        if getattr(doc, "ingestion_tasks", None):
            latest_status = doc.ingestion_tasks[0].status
            if latest_status == "completed" and (doc.chunk_count or 0) > 0:
                latest_status = "indexed"

        document_breakdown.append(
            {
                "document_id": doc.id,
                "filename": doc.filename,
                "chunk_count": int(doc.chunk_count or 0),
                "image_count": 0,
                "page_count": 0,
                "file_size": int(doc.file_size or 0),
                "status": latest_status,
            }
        )

    return {
        "stats": {
            "workspace_id": workspace_id,
            "total_documents": len(documents),
            "indexed_documents": indexed_documents,
            "total_chunks": total_chunks,
            "image_count": 0,
            "nexusrag_documents": indexed_documents,
        },
        "kg_analytics": kg_analytics,
        "document_breakdown": document_breakdown,
    }


async def _get_workspace_indexed_document_ids(
    db: AsyncSession,
    workspace_id: str,
) -> set[str]:
    documents, _total = await document_service.list_documents(
        db,
        workspace_id=workspace_id,
        skip=0,
        limit=5000,
    )
    return {
        str(doc.id)
        for doc in documents
        if int(doc.chunk_count or 0) > 0
    }
