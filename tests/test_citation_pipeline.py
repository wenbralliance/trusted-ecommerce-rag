import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from rag.main import app
from rag.graph import nodes
from rag.graph.builder import app as agent_graph
from rag.tools.retriever import RetrievalBundle


@pytest.mark.asyncio
async def test_retrieve_node_applies_filters_and_collects_sources():
    bundle = RetrievalBundle(
        content="[证据 1]",
        sources=[
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Example",
                "text": "Evidence",
                "source_url": "https://example.com",
                "source_urls": ["https://example.com"],
                "source_domain": "example.com",
                "source_type": "official",
                "source_status": "provided",
            }
        ],
    )
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "retrieve_documents",
                        "args": {"query": "test query"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ],
        "filters": {"source_type": "official"},
        "retrieved_sources": [],
        "retrieval_alerts": [],
    }

    with patch(
        "rag.graph.nodes.search_and_format",
        new=AsyncMock(return_value=bundle),
    ) as search:
        result = await nodes.retrieve_node(state)

    search.assert_awaited_once_with(
        "test query",
        filters={"source_type": "official"},
        top_k=nodes.RETRIEVAL_TOP_K,
    )
    assert result["retrieved_sources"][0]["chunk_id"] == "chunk-1"
    assert result["messages"][0].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_generate_node_enriches_verified_citation_and_marks_hallucination():
    response = nodes.AnswerWithCitations(
        answer="Answer",
        citations=[
            nodes.Citation(document_id="doc-1", chunk_id="chunk-1"),
            nodes.Citation(document_id="fake", chunk_id="fake-chunk"),
        ],
    )
    state = {
        "messages": [HumanMessage(content="Question")],
        "retry_count": 0,
        "filters": {},
        "retrieved_sources": [
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Example",
                "text": "Supporting evidence",
                "source_url": "https://example.com",
                "source_urls": ["https://example.com"],
                "source_domain": "example.com",
                "source_type": "official",
                "source_status": "provided",
            }
        ],
        "retrieval_alerts": [],
    }

    generator = AsyncMock()
    generator.ainvoke.return_value = response
    with patch("rag.graph.nodes.generator_llm", new=generator):
        result = await nodes.generate_node(state)

    payload = json.loads(result["messages"][0].content)
    assert payload["verified_citations"][0]["source_url"] == "https://example.com"
    assert payload["verified_citations"][0]["evidence"] == "Supporting evidence"
    assert payload["unverified_citations"][0]["chunk_id"] == "fake-chunk"


def test_chat_endpoint_formats_traceable_response_and_request_id():
    graph_result = {
        "messages": [
            AIMessage(
                content=json.dumps(
                    {
                        "answer": "支持。",
                        "verified_citations": [
                            {
                                "document_id": "doc-1",
                                "chunk_id": "chunk-1",
                                "title": "Example product",
                                "source_url": "https://example.com",
                                "source_urls": ["https://example.com"],
                                "source_domain": "example.com",
                                "source_type": "official",
                                "source_status": "provided",
                                "evidence": "4K video support",
                            }
                        ],
                        "unverified_citations": [],
                    },
                    ensure_ascii=False,
                )
            )
        ],
        "retrieved_sources": [
            {"document_id": "doc-1", "chunk_id": "chunk-1"}
        ],
        "retrieval_alerts": [],
    }
    fake_graph = AsyncMock()
    fake_graph.ainvoke.return_value = graph_result

    with patch("rag.api.routes.agent_app", new=fake_graph):
        client = TestClient(app)
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "是否支持4K？",
                "filters": {"source_type": "official"},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"]
    assert response.headers["X-Request-ID"] == payload["request_id"]
    assert payload["verified_citations"][0]["source_url"] == "https://example.com"
    assert payload["retrieval_count"] == 1


@pytest.mark.asyncio
async def test_compiled_graph_runs_without_preinitialized_list_state():
    agent_model = AsyncMock()
    agent_model.ainvoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "retrieve_documents",
                "args": {"query": "4K camera"},
                "id": "call-graph",
                "type": "tool_call",
            }
        ],
    )
    grader_model = AsyncMock()
    grader_model.ainvoke.return_value = nodes.GradeDocuments(binary_score="yes")
    generator = AsyncMock()
    generator.ainvoke.return_value = nodes.AnswerWithCitations(
        answer="支持4K。",
        citations=[nodes.Citation(document_id="doc-1", chunk_id="chunk-1")],
    )
    bundle = RetrievalBundle(
        content="[证据 1]",
        sources=[
            {
                "document_id": "doc-1",
                "chunk_id": "chunk-1",
                "title": "Camera",
                "text": "Supports 4K video.",
                "source_url": "https://example.com",
                "source_urls": ["https://example.com"],
                "source_domain": "example.com",
                "source_type": "official",
                "source_status": "provided",
            }
        ],
    )

    with (
        patch("rag.graph.nodes.llm_with_tools", new=agent_model),
        patch("rag.graph.nodes.grader_llm", new=grader_model),
        patch("rag.graph.nodes.generator_llm", new=generator),
        patch(
            "rag.graph.nodes.search_and_format",
            new=AsyncMock(return_value=bundle),
        ),
    ):
        result = await agent_graph.ainvoke(
            {"messages": [HumanMessage(content="Does it support 4K?")]}
        )

    payload = json.loads(result["messages"][-1].content)
    assert payload["answer"] == "支持4K。"
    assert payload["verified_citations"][0]["chunk_id"] == "chunk-1"
