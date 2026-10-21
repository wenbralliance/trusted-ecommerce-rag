import pytest

from rag.config import _rerank_api_url
from rag.tools.retriever import group_results_by_product, rank_candidates


def _hit(product_id: str, chunk_id: str, title: str, score: float):
    return {
        "document_id": product_id,
        "chunk_id": chunk_id,
        "title": title,
        "text": f"Evidence for {title}",
        "score": score,
    }


def test_group_results_deduplicates_products_and_keeps_best_evidence():
    results = [
        _hit("product-a", "a-low", "Product A", 0.2),
        _hit("product-b", "b-1", "Product B", 0.9),
        _hit("product-a", "a-high", "Product A", 0.8),
    ]

    groups = group_results_by_product(results)

    assert len(groups) == 2
    assert [item["chunk_id"] for item in groups[0]] == ["a-high", "a-low"]
    assert [item["chunk_id"] for item in groups[1]] == ["b-1"]


@pytest.mark.asyncio
async def test_no_rerank_selects_unique_products():
    results = [
        _hit("product-a", "a-1", "Product A", 0.9),
        _hit("product-a", "a-2", "Product A", 0.8),
        _hit("product-b", "b-1", "Product B", 0.7),
        _hit("product-c", "c-1", "Product C", 0.6),
    ]

    selected, alerts = await rank_candidates(
        "query",
        results,
        use_rerank=False,
        top_products=2,
        evidence_per_product=1,
    )

    assert [item["title"] for item in selected] == ["Product A", "Product B"]
    assert alerts == []


@pytest.mark.asyncio
async def test_missing_rerank_key_falls_back_to_unique_products(monkeypatch):
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    results = [
        _hit("product-a", "a-1", "Product A", 0.9),
        _hit("product-b", "b-1", "Product B", 0.8),
        _hit("product-c", "c-1", "Product C", 0.7),
    ]

    selected, alerts = await rank_candidates(
        "query",
        results,
        use_rerank=True,
        top_products=2,
        evidence_per_product=1,
    )

    assert [item["title"] for item in selected] == ["Product A", "Product B"]
    assert alerts == ["未配置 RERANK_API_KEY，已使用商品级 RRF 顺序。"]


def test_rerank_base_url_is_derived_from_compatible_endpoint(monkeypatch):
    monkeypatch.delenv("RERANK_API_URL", raising=False)
    monkeypatch.setenv(
        "RERANK_BASE_URL",
        "https://example.com/compatible-mode/v1",
    )
    assert _rerank_api_url() == (
        "https://example.com/api/v1/services/"
        "rerank/text-rerank/text-rerank"
    )


def test_explicit_rerank_api_url_has_priority(monkeypatch):
    monkeypatch.setenv(
        "RERANK_API_URL",
        "https://example.com/custom/rerank",
    )
    monkeypatch.setenv(
        "RERANK_BASE_URL",
        "https://ignored.example.com/compatible-mode/v1",
    )
    assert _rerank_api_url() == "https://example.com/custom/rerank"
