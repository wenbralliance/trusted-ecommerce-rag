import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Response
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from rag.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationItem,
    HealthResponse,
)
from rag.config import COLLECTION_NAME, milvus_client
from rag.graph.builder import app as agent_app

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="服务与集合健康检查")
async def health_endpoint():
    try:
        exists = await milvus_client.has_collection(COLLECTION_NAME)
        if not exists:
            raise RuntimeError("collection does not exist")
        stats = await milvus_client.get_collection_stats(
            collection_name=COLLECTION_NAME
        )
        row_count = int(stats.get("row_count", 0))
        if row_count <= 0:
            raise RuntimeError("collection is empty")
    except Exception:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=503,
            detail="Milvus collection is not ready",
        ) from None

    return HealthResponse(
        status="ready",
        collection=COLLECTION_NAME,
        row_count=row_count,
    )


def _parse_citations(
    items: Any,
    alerts: list[str],
    label: str,
) -> list[CitationItem]:
    if not isinstance(items, list):
        alerts.append(f"{label}格式无效，已忽略。")
        return []

    citations: list[CitationItem] = []
    for item in items:
        if not isinstance(item, dict):
            alerts.append(f"{label}中存在非对象条目，已忽略。")
            continue
        try:
            citations.append(CitationItem(**item))
        except ValidationError:
            alerts.append(f"{label}中存在字段不完整的条目，已忽略。")
    return citations


@router.post("/chat", response_model=ChatResponse, summary="Agentic RAG 对话接口")
async def chat_endpoint(req: ChatRequest, response: Response):
    request_id = uuid.uuid4().hex
    response.headers["X-Request-ID"] = request_id

    filters = (
        req.filters.model_dump(exclude_none=True) if req.filters else {}
    )
    inputs = {
        "messages": [HumanMessage(content=req.query)],
        "filters": filters,
        "retry_count": 0,
        "retrieved_sources": [],
        "retrieval_alerts": [],
    }

    try:
        result = await agent_app.ainvoke(inputs)
    except Exception:
        logger.exception("Agent execution failed", extra={"request_id": request_id})
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Agent 内部执行失败，请稍后重试。",
                "request_id": request_id,
            },
        ) from None

    alerts = list(
        dict.fromkeys(str(alert) for alert in result.get("retrieval_alerts", []))
    )
    unique_sources = {
        (source.get("document_id", ""), source.get("chunk_id", "")): source
        for source in result.get("retrieved_sources", [])
    }
    retrieval_count = len(unique_sources)
    final_ai_msg = result["messages"][-1]
    content = (
        final_ai_msg.content
        if isinstance(final_ai_msg.content, str)
        else str(final_ai_msg.content)
    )

    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise json.JSONDecodeError("Expected a JSON object", content, 0)
    except json.JSONDecodeError:
        alerts.append("模型输出没有使用标准结构化格式，引用来源无法解析。")
        return ChatResponse(
            request_id=request_id,
            answer=content,
            retrieval_count=retrieval_count,
            is_degraded=bool(alerts),
            alerts=list(dict.fromkeys(alerts)),
        )

    verified = _parse_citations(
        data.get("verified_citations", []),
        alerts,
        "已验证引用",
    )
    unverified = _parse_citations(
        data.get("unverified_citations", []),
        alerts,
        "未验证引用",
    )
    if unverified:
        alerts.append(
            f"有 {len(unverified)} 条模型引用未通过后验校验，请人工复核。"
        )

    answer = str(data.get("answer", "")).strip()
    if not answer:
        alerts.append("模型返回了空答案。")

    return ChatResponse(
        request_id=request_id,
        answer=answer,
        verified_citations=verified,
        unverified_citations=unverified,
        retrieval_count=retrieval_count,
        is_degraded=bool(alerts),
        alerts=list(dict.fromkeys(alerts)),
    )
