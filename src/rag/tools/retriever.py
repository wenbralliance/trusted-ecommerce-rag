import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from langchain_core.tools import tool

from rag.config import (
    ENABLE_RERANK,
    RERANK_DOC_MAX_CHARS,
    RERANK_EVIDENCE_PER_PRODUCT,
    RERANK_API_URL,
    RERANK_TOP_PRODUCTS,
    RETRIEVAL_TOP_K,
    TOP_N,
)
from rag.db.milvus_client import hybrid_search


@dataclass
class RetrievalBundle:
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


def _score(result: dict[str, Any]) -> float:
    value = result.get("score")
    return float(value) if value is not None else float("-inf")


def group_results_by_product(
    results: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        key = (
            result.get("document_id")
            or result.get("title")
            or result.get("chunk_id")
            or str(len(groups))
        )
        groups.setdefault(str(key), []).append(result)

    ordered_groups = []
    for group in groups.values():
        ordered_groups.append(sorted(group, key=_score, reverse=True))
    return ordered_groups


def _group_rerank_text(
    group: list[dict[str, Any]],
    max_chars: int,
) -> str:
    title = str(group[0].get("title") or "未知商品")
    parts = [f"商品：{title}"]
    used = len(parts[0])
    seen_text: set[str] = set()

    for result in group:
        text = str(result.get("text", "")).strip()
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        remaining = max_chars - used
        if remaining <= 0:
            break
        parts.append(text[:remaining])
        used += min(len(text), remaining) + 1
    return "\n".join(parts)[:max_chars]


async def _rerank_product_groups(
    query: str,
    groups: list[list[dict[str, Any]]],
    top_products: int,
    alerts: list[str],
) -> list[list[dict[str, Any]]]:
    if not groups:
        return []

    top_products = max(1, min(top_products, len(groups)))
    api_key = os.getenv("RERANK_API_KEY")
    if not api_key:
        alerts.append("未配置 RERANK_API_KEY，已使用商品级 RRF 顺序。")
        return groups[:top_products]

    payload = {
        "model": os.getenv("RERANK_MODEL", "qwen3.7-text-rerank"),
        "input": {
            "query": query,
            "documents": [
                _group_rerank_text(group, RERANK_DOC_MAX_CHARS)
                for group in groups
            ],
        },
        "parameters": {
            "top_n": top_products,
            "instruct": (
                "Rank unique products by how well their evidence satisfies all "
                "explicit shopping constraints. Penalize products that satisfy "
                "only a generic category or a single requirement."
            ),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                RERANK_API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            items = response.json()["output"]["results"]

        selected: list[list[dict[str, Any]]] = []
        seen_indexes: set[int] = set()
        for item in items:
            index = int(item["index"])
            if not 0 <= index < len(groups) or index in seen_indexes:
                continue
            seen_indexes.add(index)
            selected.append(groups[index])
        if selected:
            return selected[:top_products]
        alerts.append("Reranker 未返回有效结果，已使用商品级 RRF 顺序。")
        return groups[:top_products]
    except Exception as exc:
        alerts.append(f"Reranker 调用失败，已回退到商品级 RRF 顺序：{exc}")
        return groups[:top_products]


async def rank_candidates(
    query: str,
    results: list[dict[str, Any]],
    *,
    use_rerank: bool,
    top_products: int,
    evidence_per_product: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Aggregate chunk hits by product, optionally rerank, then select evidence."""
    groups = group_results_by_product(results)
    alerts: list[str] = []
    if use_rerank:
        groups = await _rerank_product_groups(
            query,
            groups,
            top_products=top_products,
            alerts=alerts,
        )
    else:
        groups = groups[: max(1, top_products)]

    evidence_limit = max(1, evidence_per_product)
    selected: list[dict[str, Any]] = []
    seen_chunks: set[str] = set()
    for group in groups:
        for result in group[:evidence_limit]:
            key = (
                result.get("document_id", ""),
                result.get("chunk_id", ""),
            )
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            selected.append(result)
    return selected, alerts


def _format_evidence(sources: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, source in enumerate(sources, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[证据 {index}]",
                    f"商品/标题: {source.get('title') or '未知文档'}",
                    f"document_id: {source.get('document_id', '')}",
                    f"chunk_id: {source.get('chunk_id', '')}",
                    f"来源类型: {source.get('source_type', 'unknown')}",
                    f"来源域名: {source.get('source_domain', 'unknown')}",
                    f"来源状态: {source.get('source_status', 'missing')}",
                    "内容:",
                    source.get("text", ""),
                ]
            )
        )
    return "\n\n".join(blocks)


async def search_and_format(
    query: str,
    filters: dict[str, Any] | None = None,
    top_k: int = RETRIEVAL_TOP_K,
    use_rerank: bool | None = None,
) -> RetrievalBundle:
    alerts: list[str] = []
    try:
        results = await hybrid_search(query, top_k=top_k, filters=filters)
    except Exception as exc:
        return RetrievalBundle(
            content=f"检索失败：{exc}",
            alerts=[f"混合检索失败：{exc}"],
        )

    if not results:
        return RetrievalBundle(content="未找到相关文档。")

    rerank_enabled = ENABLE_RERANK if use_rerank is None else use_rerank
    if rerank_enabled:
        selected, ranking_alerts = await rank_candidates(
            query,
            results,
            use_rerank=True,
            top_products=RERANK_TOP_PRODUCTS,
            evidence_per_product=RERANK_EVIDENCE_PER_PRODUCT,
        )
        alerts.extend(ranking_alerts)
    else:
        selected = results[:TOP_N]

    sources = []
    for result in selected:
        source = dict(result)
        source.pop("score", None)
        sources.append(source)

    return RetrievalBundle(
        content=_format_evidence(sources),
        sources=sources,
        alerts=alerts,
    )


@tool
async def retrieve_documents(query: str) -> str:
    """根据用户需求检索商品证据。查询应包含产品类别和关键约束。"""
    bundle = await search_and_format(query)
    return bundle.content
