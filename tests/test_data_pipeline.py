import json
import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from rag.db.milvus_client import build_filter_expression
from rag.main import app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_shoppingcomp import (  # noqa: E402
    canonicalize_url,
    classify_domain,
    extract_and_strip_urls,
)


def test_url_canonicalization_removes_tracking_parameters():
    result = canonicalize_url(
        "HTTPS://www.Example.com/item?id=42&utm_source=test#reviews"
    )
    assert result == ("https://www.example.com/item?id=42", "example.com")
    assert classify_domain("consumer.huawei.com") == "official"
    assert classify_domain("item.jd.com") == "retailer"
    assert classify_domain("www.zhihu.com") == "community"


def test_url_is_removed_from_embedding_text():
    text, urls, domains = extract_and_strip_urls(
        "支持快充。来源：https://example.com/spec?utm_medium=test。"
    )
    assert "http" not in text
    assert "来源：" not in text
    assert urls == ["https://example.com/spec"]
    assert domains == ["example.com"]


def test_filter_expression_is_allowlisted_and_escaped():
    expression = build_filter_expression(
        {"source_type": "official", "source_domain": 'vendor"example.com'}
    )
    assert expression == (
        'source_type == "official" and '
        'source_domain == "vendor\\"example.com"'
    )


def test_prepared_corpus_has_no_urls_in_searchable_text():
    path = PROJECT_ROOT / "data" / "ecommerce" / "shoppingcomp" / "documents.jsonl"
    documents = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(documents) == 1190
    assert not any(
        re.search(r"https?://", document["text"], re.IGNORECASE)
        for document in documents
    )


def test_openapi_exposes_metadata_filter_and_trace_fields():
    client = TestClient(app)
    spec = client.get("/openapi.json").json()
    chat_schema = spec["components"]["schemas"]["ChatRequest"]
    response_schema = spec["components"]["schemas"]["ChatResponse"]
    citation_schema = spec["components"]["schemas"]["CitationItem"]

    assert "filters" in chat_schema["properties"]
    assert "request_id" in response_schema["properties"]
    assert "retrieval_count" in response_schema["properties"]
    assert "source_urls" in citation_schema["properties"]
    assert "evidence" in citation_schema["properties"]
