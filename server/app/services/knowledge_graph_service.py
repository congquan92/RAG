"""
Service Knowledge Graph
=======================

Knowledge Graph theo từng workspace dùng LightRAG với LLM + embeddings có thể cấu hình.
Lưu trữ dạng file (NetworkX graph + NanoVectorDB), không cần thêm Docker service.

Cách dùng:
    kg = KnowledgeGraphService(workspace_id=1)
    await kg.ingest("markdown text from document...")
    result = await kg.query("What are the key themes?", mode="hybrid")
    await kg.cleanup()
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from app.core.config import settings
from app.services.llm import get_embedding_provider, get_llm_provider
from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)

_KG_COMPLETION_DELIMITER = "<|COMPLETE|>"

_INPUT_TEXT_PATTERN = re.compile(
    r"<Input Text>\s*```(?:[A-Za-z0-9_-]+)?\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
_ENTITY_TYPES_PATTERN = re.compile(
    r"<Entity_types>\s*(\[.*?\])",
    flags=re.IGNORECASE | re.DOTALL,
)


def _is_kg_extraction_prompt(prompt: str, system_prompt: Optional[str]) -> bool:
    prompt_lower = (prompt or "").lower()
    system_lower = (system_prompt or "").lower()

    if "extract entities and relationships" in prompt_lower:
        return True

    if "based on the last extraction task" in prompt_lower:
        return True

    return (
        "knowledge graph specialist responsible for extracting entities and relationships"
        in system_lower
    )


def _extract_input_text_from_prompt(prompt: str, history_messages: Optional[list]) -> str:
    def _find_input_text(raw_text: str) -> str:
        if not raw_text:
            return ""
        match = _INPUT_TEXT_PATTERN.search(raw_text)
        return match.group(1).strip() if match else ""

    text = _find_input_text(prompt)
    if text:
        return text

    if not history_messages:
        return ""

    for message in reversed(history_messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            flattened: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    flattened.append(str(item.get("text", "")))
                else:
                    flattened.append(str(item))
            content_text = "\n".join(flattened)
        else:
            content_text = str(content)

        text = _find_input_text(content_text)
        if text:
            return text

    return ""


def _extract_entity_types_from_prompt(prompt: str) -> list[str] | None:
    match = _ENTITY_TYPES_PATTERN.search(prompt or "")
    if not match:
        return None

    raw_block = match.group(1).strip()
    try:
        parsed = json.loads(raw_block)
        if isinstance(parsed, list):
            entity_types = [str(item).strip() for item in parsed if str(item).strip()]
            return entity_types or None
    except json.JSONDecodeError:
        pass

    fallback = raw_block.strip("[]")
    entity_types = [
        token.strip().strip('"').strip("'")
        for token in fallback.split(",")
        if token.strip().strip('"').strip("'")
    ]
    return entity_types or None


# ---------------------------------------------------------------------------
# Adapter theo provider cho LightRAG
# ---------------------------------------------------------------------------

async def _kg_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list] = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """Hàm LLM tương thích LightRAG dùng provider đã cấu hình."""
    if (
        settings.KG_EXTRACTION_METHOD == "specialized"
        and _is_kg_extraction_prompt(prompt=prompt, system_prompt=system_prompt)
    ):
        prompt_lower = (prompt or "").lower()

        # Giai đoạn gleaning không có <Input Text>; no-op để tránh output trùng lặp.
        if "based on the last extraction task" in prompt_lower:
            return _KG_COMPLETION_DELIMITER

        input_text = _extract_input_text_from_prompt(
            prompt=prompt,
            history_messages=history_messages,
        )
        if not input_text:
            logger.warning(
                "Specialized KG extraction was requested but no <Input Text> block was found. "
                "Trả về completion delimiter để tương thích."
            )
            return _KG_COMPLETION_DELIMITER

        entity_types = _extract_entity_types_from_prompt(prompt)

        try:
            from app.services.extractor.specialized_kg_extractor import (
                get_specialized_kg_extractor,
            )

            extractor = get_specialized_kg_extractor()
            extracted = await asyncio.to_thread(
                extractor.extract_entities_and_relations_sync,
                input_text,
                entity_types,
            )
            return extracted or _KG_COMPLETION_DELIMITER
        except Exception as exc:
            logger.exception(
                "Specialized KG extraction failed, falling back to LLM provider: %s",
                exc,
            )

    provider = get_llm_provider()

    messages: list[LLMMessage] = []

    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))

    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(LLMMessage(role=role, content=content))

    messages.append(LLMMessage(role="user", content=prompt))

    return await provider.acomplete(messages, temperature=0.0, max_tokens=4096)


async def _kg_embed(texts: list[str]) -> np.ndarray:
    """Hàm embedding tương thích LightRAG dùng provider đã cấu hình."""
    provider = get_embedding_provider()
    return await provider.embed(texts)


# ---------------------------------------------------------------------------
# Service chính
# ---------------------------------------------------------------------------

class KnowledgeGraphService:
    """
    Service Knowledge Graph theo từng workspace chạy trên LightRAG.

    Storage: file-based (NetworkX cho graph, NanoVectorDB cho vectors).
    Mỗi knowledge base có working directory riêng.
    """

    def __init__(
        self,
        workspace_id: int,
        kg_language: str | None = None,
        kg_entity_types: list[str] | None = None,
    ):
        self.workspace_id = workspace_id
        self.working_dir = str(
            settings.BASE_DIR / "data" / "lightrag" / f"kb_{workspace_id}"
        )
        # Override theo workspace (fallback về global settings)
        self.kg_language = kg_language or settings.NEXUSRAG_KG_LANGUAGE
        self.kg_entity_types = kg_entity_types or settings.NEXUSRAG_KG_ENTITY_TYPES
        self._rag = None
        self._initialized = False

    async def _get_rag(self):
        """Khởi tạo LightRAG instance theo kiểu lazy."""
        if self._rag is not None and self._initialized:
            return self._rag

        from lightrag import LightRAG
        from lightrag.utils import wrap_embedding_func_with_attrs
        from lightrag.kg.shared_storage import initialize_pipeline_status

        os.makedirs(self.working_dir, exist_ok=True)

        # Embedding dimension động từ provider đã cấu hình
        emb_provider = get_embedding_provider()
        embedding_dim = emb_provider.get_dimension()

        # Phát hiện dimension mismatch khi chuyển provider
        dim_marker = Path(self.working_dir) / ".embedding_dim"
        if dim_marker.exists():
            prev_dim = int(dim_marker.read_text().strip())
            if prev_dim != embedding_dim:
                logger.warning(
                    f"Embedding dimension changed ({prev_dim} → {embedding_dim}) "
                    f"for workspace {self.workspace_id}. Clearing KG data for rebuild."
                )
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
        dim_marker.write_text(str(embedding_dim))

        @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await _kg_embed(texts)

        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=_kg_llm_complete,
            default_llm_timeout=settings.LLM_TIMEOUT,
            embedding_func=embedding_func,
            chunk_token_size=settings.NEXUSRAG_KG_CHUNK_TOKEN_SIZE,
            enable_llm_cache=True,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
            addon_params={
                "language": self.kg_language,
                "entity_types": self.kg_entity_types,
            },
        )

        await self._rag.initialize_storages()
        await initialize_pipeline_status()
        self._initialized = True

        logger.info(
            f"LightRAG initialized for workspace {self.workspace_id} "
            f"(embedding_dim={embedding_dim})"
        )
        return self._rag

    async def ingest(self, markdown_content: str) -> None:
        """
        Nạp nội dung markdown vào knowledge graph.
        LightRAG tự động trích xuất entities và relationships.
        """
        rag = await self._get_rag()

        if not markdown_content.strip():
            logger.warning(f"Empty content for workspace {self.workspace_id}, skipping KG ingest")
            return

        try:
            await rag.ainsert(markdown_content)
            logger.info(
                f"KG ingested {len(markdown_content)} chars for workspace {self.workspace_id}"
            )

            # Kiểm tra xem entities có thực sự được trích xuất hay không
            try:
                all_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                if not all_nodes:
                    from app.core.config import settings
                    model = (
                        settings.OLLAMA_MODEL
                        if settings.LLM_PROVIDER.lower() == "ollama"
                        else settings.LLM_MODEL_FAST
                    )
                    logger.warning(
                        f"KG extraction produced 0 entities for workspace {self.workspace_id}. "
                        f"Model '{model}' may not support LightRAG's entity extraction format. "
                        f"Consider using a larger model (e.g. qwen3:14b, gemma3:12b) for KG."
                    )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"KG ingest failed for workspace {self.workspace_id}: {e}")
            raise

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> str:
        """
        Truy vấn knowledge graph.

        Args:
            question: Câu hỏi ngôn ngữ tự nhiên
            mode: Query mode - "naive", "local", "global", "hybrid"
            top_k: Số lượng kết quả

        Returns:
            Văn bản phản hồi từ LightRAG với câu trả lời được tăng cường bởi KG
        """
        from lightrag import QueryParam

        rag = await self._get_rag()

        try:
            result = await asyncio.wait_for(
                rag.aquery(
                    question,
                    param=QueryParam(mode=mode, top_k=top_k),
                ),
                timeout=settings.NEXUSRAG_KG_QUERY_TIMEOUT,
            )
            return result or ""
        except asyncio.TimeoutError:
            logger.warning(
                f"KG query timed out after {settings.NEXUSRAG_KG_QUERY_TIMEOUT}s "
                f"for workspace {self.workspace_id}"
            )
            return ""
        except Exception as e:
            logger.error(f"KG query failed for workspace {self.workspace_id}: {e}")
            return ""

    async def cleanup(self) -> None:
        """Finalize storages khi shutdown."""
        if self._rag:
            try:
                await self._rag.finalize_storages()
                logger.info(f"KG storages finalized for workspace {self.workspace_id}")
            except Exception as e:
                logger.warning(f"KG cleanup failed for workspace {self.workspace_id}: {e}")
            self._rag = None
            self._initialized = False

    def delete_project_data(self) -> None:
        """Xóa toàn bộ dữ liệu KG của knowledge base này."""
        path = Path(self.working_dir)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted KG data for workspace {self.workspace_id}")
        self._rag = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Khám phá Knowledge Graph (Giai đoạn 9)
    # ------------------------------------------------------------------

    async def get_entities(
        self,
        search: str | None = None,
        entity_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """
        Liệt kê toàn bộ entities trong knowledge graph.

        Trả về danh sách dict gồm: name, entity_type, description, degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
        except Exception as e:
            logger.error(f"Failed to get KG nodes for workspace {self.workspace_id}: {e}")
            return []

        entities = []
        for node in all_nodes:
            node_id = node.get("id", "")
            etype = node.get("entity_type", "Unknown")
            desc = node.get("description", "")

            # Bộ lọc
            if entity_type and etype.lower() != entity_type.lower():
                continue
            if search and search.lower() not in node_id.lower():
                continue

            # Lấy degree (số lượng relationships)
            try:
                degree = await storage.node_degree(node_id)
            except Exception:
                degree = 0

            entities.append({
                "name": node_id,
                "entity_type": etype,
                "description": desc,
                "degree": degree,
            })

        # Sắp xếp degree giảm dần
        entities.sort(key=lambda e: e["degree"], reverse=True)

        return entities[offset:offset + limit]

    async def get_relationships(
        self,
        entity_name: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        Liệt kê relationships trong knowledge graph.

        Nếu có entity_name, chỉ trả về relationships liên quan entity đó.
        Trả về danh sách dict gồm: source, target, description, keywords, weight.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG edges for workspace {self.workspace_id}: {e}")
            return []

        relationships = []
        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")

            if entity_name:
                if entity_name.lower() not in (src.lower(), tgt.lower()):
                    continue

            relationships.append({
                "source": src,
                "target": tgt,
                "description": edge.get("description", ""),
                "keywords": edge.get("keywords", ""),
                "weight": float(edge.get("weight", 1.0)),
            })

        return relationships[:limit]

    async def get_graph_data(
        self,
        center_entity: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 150,
    ) -> dict:
        """
        Export dữ liệu graph cho frontend visualization.

        Trả về {nodes: [...], edges: [...], is_truncated: bool}.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            label = center_entity if center_entity else "*"
            kg = await storage.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Failed to get KG graph for workspace {self.workspace_id}: {e}")
            return {"nodes": [], "edges": [], "is_truncated": False}

        nodes = []
        for n in kg.nodes:
            props = n.properties if hasattr(n, "properties") else {}
            try:
                degree = await storage.node_degree(n.id)
            except Exception:
                degree = 0
            nodes.append({
                "id": n.id,
                "label": n.id,
                "entity_type": props.get("entity_type", "Unknown"),
                "degree": degree,
            })

        edges = []
        for e in kg.edges:
            props = e.properties if hasattr(e, "properties") else {}
            edges.append({
                "source": e.source,
                "target": e.target,
                "label": props.get("description", "")[:80],
                "weight": float(props.get("weight", 1.0)),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "is_truncated": kg.is_truncated if hasattr(kg, "is_truncated") else False,
        }

    async def get_relevant_context(
        self,
        question: str,
        max_entities: int = 20,
        max_relationships: int = 30,
    ) -> str:
        """
                Dựng context RAG từ dữ liệu KG thô (không generate bằng LLM).

                Thay vì gọi aquery() của LightRAG (dùng LLM để tạo narrative,
                có thể hallucinate), phương thức này sẽ:
                    1. Tách question thành keywords
                    2. Tìm entities có tên khớp keywords
                    3. Lấy relationships nối các entities đó
                    4. Định dạng lại thành văn bản facts có cấu trúc

        Returns:
                        Chuỗi có cấu trúc gồm entities + relationships, hoặc "" nếu không có.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get raw KG data for workspace {self.workspace_id}: {e}")
            return ""

        if not all_nodes:
            return ""

        # -- 1. Trích xuất keywords từ question --
        # Đơn giản nhưng hiệu quả: tách từ, lowercase, lọc từ quá ngắn
        raw_tokens = question.lower().split()
        # Đồng thời xử lý token có dấu gạch ngang/phiên bản như "deepseek-v3.2"
        keywords = set()
        for token in raw_tokens:
            # Loại dấu câu ở rìa token
            cleaned = token.strip(".,?!:;\"'()[]{}").lower()
            if len(cleaned) >= 2:
                keywords.add(cleaned)

        if not keywords:
            return ""

        # -- 2. Tìm entities khớp --
        matched_entity_names: set[str] = set()
        entity_info: dict[str, dict] = {}  # name -> {type, description}

        for node in all_nodes:
            node_id = node.get("id", "")
            node_lower = node_id.lower()

            # Kiểm tra keyword là chuỗi con của entity name HOẶC ngược lại
            matched = False
            for kw in keywords:
                if kw in node_lower or node_lower in kw:
                    matched = True
                    break
                # Kiểm tra thêm keyword nhiều phần (vd: "deepseek" khớp "DEEPSEEK-V3.2")
                for part in node_lower.split("-"):
                    if kw in part or part in kw:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                matched_entity_names.add(node_id)
                entity_info[node_id] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names and len(all_nodes) <= 50:
            # Graph nhỏ: mặc định lấy thêm top entities
            for node in all_nodes[:10]:
                nid = node.get("id", "")
                matched_entity_names.add(nid)
                entity_info[nid] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names:
            return ""

        # Giới hạn số entities
        matched_list = list(matched_entity_names)[:max_entities]

        # -- 3. Tìm relationships liên quan các entities đã khớp --
        relevant_rels: list[dict] = []
        matched_lower = {n.lower() for n in matched_list}

        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src.lower() in matched_lower or tgt.lower() in matched_lower:
                relevant_rels.append({
                    "source": src,
                    "target": tgt,
                    "description": edge.get("description", ""),
                    "keywords": edge.get("keywords", ""),
                })
                # Bổ sung entities liên thông có thể đã bị bỏ sót
                if src not in entity_info:
                    # Tìm thông tin node
                    for n in all_nodes:
                        if n.get("id", "") == src:
                            entity_info[src] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break
                if tgt not in entity_info:
                    for n in all_nodes:
                        if n.get("id", "") == tgt:
                            entity_info[tgt] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break

            if len(relevant_rels) >= max_relationships:
                break

        # -- 4. Định dạng thành text có cấu trúc --
        parts: list[str] = []

        # Phần entities
        if matched_list:
            parts.append("Entities found in documents:")
            for name in matched_list:
                info = entity_info.get(name, {})
                etype = info.get("entity_type", "")
                desc = info.get("description", "")
                # Cắt ngắn description quá dài
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                type_str = f" [{etype}]" if etype and etype != "Unknown" else ""
                if desc:
                    parts.append(f"- {name}{type_str}: {desc}")
                else:
                    parts.append(f"- {name}{type_str}")

        # Phần relationships
        if relevant_rels:
            parts.append("")
            parts.append("Relationships:")
            for rel in relevant_rels:
                desc = rel["description"]
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                if desc:
                    parts.append(f"- {rel['source']} → {rel['target']}: {desc}")
                else:
                    parts.append(f"- {rel['source']} → {rel['target']}")

        result = "\n".join(parts)
        logger.info(
            f"KG raw context: {len(matched_list)} entities, "
            f"{len(relevant_rels)} relationships for workspace {self.workspace_id}"
        )
        return result

    async def get_analytics(self) -> dict:
        """
        Tính toán tóm tắt analytics của KG.

        Trả về: entity_count, relationship_count, entity_types, top_entities, avg_degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG analytics for workspace {self.workspace_id}: {e}")
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "entity_types": {},
                "top_entities": [],
                "avg_degree": 0.0,
            }

        entity_count = len(all_nodes)
        relationship_count = len(all_edges)

        # Đếm số lượng theo entity type
        type_counts: dict[str, int] = {}
        entities_with_degree = []
        for node in all_nodes:
            etype = node.get("entity_type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
            try:
                degree = await storage.node_degree(node.get("id", ""))
            except Exception:
                degree = 0
            entities_with_degree.append({
                "name": node.get("id", ""),
                "entity_type": etype,
                "description": node.get("description", ""),
                "degree": degree,
            })

        # Sắp xếp theo degree để lấy top entities
        entities_with_degree.sort(key=lambda e: e["degree"], reverse=True)
        top_entities = entities_with_degree[:10]

        avg_degree = (
            sum(e["degree"] for e in entities_with_degree) / entity_count
            if entity_count > 0
            else 0.0
        )

        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "entity_types": type_counts,
            "top_entities": top_entities,
            "avg_degree": round(avg_degree, 2),
        }
