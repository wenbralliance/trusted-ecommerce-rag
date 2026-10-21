import json
import os
from typing import Any

from pymilvus import AnnSearchRequest, RRFRanker

from rag.config import COLLECTION_NAME, EMBEDDING_DIM, embed_client, milvus_client
from rag.db.schema import FILTERABLE_FIELDS, OUTPUT_FIELDS


def build_filter_expression(filters: dict[str, Any] | None) -> str | None:
    """Build a Milvus expression from a small, allowlisted filter mapping."""
    if not filters:
        return None

    unknown_fields = set(filters) - FILTERABLE_FIELDS
    if unknown_fields:
        raise ValueError(f"Unsupported filters: {sorted(unknown_fields)}")

    expressions: list[str] = []
    for field_name, value in filters.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Filter {field_name} must be a non-empty string")
        encoded = json.dumps(value.strip(), ensure_ascii=False)
        expressions.append(f"{field_name} == {encoded}")

    return " and ".join(expressions) or None


def _source_urls(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(url) for url in value if str(url).strip()]
    if isinstance(value, str) and value.strip():
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(decoded, list):
            return [str(url) for url in decoded if str(url).strip()]
    return []


async def hybrid_search(
    query: str,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        raise ValueError("Search query cannot be empty")
    if not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50")

    response = await embed_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=[query],
    )
    query_dense = response.data[0].embedding
    if len(query_dense) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
            f"received {len(query_dense)}"
        )

    filter_expression = build_filter_expression(filters)
    candidate_limit = max(50, top_k * 5)
    dense_req = AnnSearchRequest(
        data=[query_dense],
        anns_field="dense",
        param={"metric_type": "IP"},
        limit=candidate_limit,
        expr=filter_expression,
    )
    sparse_req = AnnSearchRequest(
        data=[query],
        anns_field="sparse",
        param={"metric_type": "BM25"},
        limit=candidate_limit,
        expr=filter_expression,
    )

    results = await milvus_client.hybrid_search(
        collection_name=COLLECTION_NAME,
        reqs=[dense_req, sparse_req],
        ranker=RRFRanker(k=60),
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
    )

    hits: list[dict[str, Any]] = []
    for hit in results[0]:
        source_urls = _source_urls(hit.entity.get("source_urls"))
        score = getattr(hit, "score", None)
        if score is None:
            score = getattr(hit, "distance", None)
        hits.append(
            {
                "document_id": hit.entity.get("document_id", ""),
                "chunk_id": hit.entity.get("chunk_id", ""),
                "title": hit.entity.get("title", ""),
                "text": hit.entity.get("text", ""),
                "source_url": hit.entity.get("source_url", ""),
                "source_urls": source_urls,
                "source_domain": hit.entity.get("source_domain", "unknown"),
                "source_type": hit.entity.get("source_type", "unknown"),
                "source_status": hit.entity.get("source_status", "missing"),
                "task_id": hit.entity.get("task_id", ""),
                "document_type": hit.entity.get("document_type", ""),
                "dataset": hit.entity.get("dataset", ""),
                "language": hit.entity.get("language", ""),
                "score": score,
            }
        )
    return hits
