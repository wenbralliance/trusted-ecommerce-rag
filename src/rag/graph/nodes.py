import asyncio
import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from rag.config import MAX_RETRIES, RETRIEVAL_TOP_K, llm
from rag.graph.state import AgentState
from rag.prompt.prompt_template import (
    generate_prompt,
    grade_prompt,
    rewrite_prompt,
    system_prompt,
)
from rag.tools.retriever import (
    RetrievalBundle,
    retrieve_documents,
    search_and_format,
)

tools = [retrieve_documents]
llm_with_tools = llm.bind_tools(tools)


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="相关性评分：'yes' 或 'no'")


grader_llm = llm.with_structured_output(GradeDocuments)


class Citation(BaseModel):
    document_id: str = Field(description="检索结果中的 document_id")
    chunk_id: str = Field(description="检索结果中的 chunk_id")


class AnswerWithCitations(BaseModel):
    answer: str = Field(description="针对用户问题的最终回答")
    citations: list[Citation] = Field(description="回答实际参考的检索证据")


generator_llm = llm.with_structured_output(
    AnswerWithCitations,
    method="json_mode",
)


async def agent_node(state: AgentState):
    """决定直接回答，或调用商品证据检索工具。"""
    response = await llm_with_tools.ainvoke(
        [SystemMessage(content=system_prompt), *state["messages"]]
    )
    return {"messages": [response]}


async def retrieve_node(state: AgentState):
    """执行 Agent 发起的检索，并收集可审计的来源元数据。"""
    messages: list[ToolMessage] = []
    sources: list[dict[str, Any]] = []
    alerts: list[str] = []
    last_message = state["messages"][-1]

    tool_calls = list(getattr(last_message, "tool_calls", []))

    async def run_tool_call(tool_call: dict[str, Any]) -> RetrievalBundle:
        query = str(tool_call.get("args", {}).get("query", "")).strip()
        if not query:
            return RetrievalBundle(
                content="检索失败：工具调用缺少 query。",
                alerts=["Agent 生成了不包含查询内容的检索调用。"],
            )
        return await search_and_format(
            query,
            filters=state.get("filters"),
            top_k=RETRIEVAL_TOP_K,
        )

    bundles = await asyncio.gather(
        *(run_tool_call(tool_call) for tool_call in tool_calls)
    )
    for tool_call, bundle in zip(tool_calls, bundles, strict=True):
        messages.append(
            ToolMessage(
                content=bundle.content,
                tool_call_id=tool_call.get("id", "unknown"),
                name="retrieve_documents",
            )
        )
        sources.extend(bundle.sources)
        alerts.extend(bundle.alerts)

    deduplicated_sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str]] = set()
    for source in sources:
        key = (source.get("document_id", ""), source.get("chunk_id", ""))
        if key in seen_sources:
            continue
        seen_sources.add(key)
        deduplicated_sources.append(source)

    return {
        "messages": messages,
        "retrieved_sources": deduplicated_sources,
        "retrieval_alerts": list(dict.fromkeys(alerts)),
    }


async def grade_documents_node(state: AgentState):
    question = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            question = str(message.content)
            break

    recent_tool_messages: list[str] = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(str(message.content))
        elif message.type == "human":
            break

    context = "\n\n".join(reversed(recent_tool_messages))
    if not context or "未找到相关文档" in context or "检索失败" in context:
        return "rewrite"

    result = await grader_llm.ainvoke(
        grade_prompt.format(question=question, context=context)
    )
    return "generate" if result.binary_score.lower() == "yes" else "rewrite"


async def rewrite_node(state: AgentState):
    question = ""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            question = str(message.content)
            break

    response = await llm.ainvoke(rewrite_prompt.format(question=question))
    rewritten_query = str(response.content).strip()
    return {
        "messages": [
            HumanMessage(
                content=f"请使用这个优化后的查询重新检索: {rewritten_query}"
            )
        ],
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _citation_payload(
    citation: Citation,
    source: dict[str, Any] | None,
) -> dict[str, Any]:
    if source is None:
        return {
            "document_id": citation.document_id,
            "chunk_id": citation.chunk_id,
            "title": None,
            "source_url": None,
            "source_urls": [],
            "source_domain": None,
            "source_type": None,
            "source_status": "unverified",
            "evidence": None,
        }
    return {
        "document_id": citation.document_id,
        "chunk_id": citation.chunk_id,
        "title": source.get("title") or None,
        "source_url": source.get("source_url") or None,
        "source_urls": source.get("source_urls", []),
        "source_domain": source.get("source_domain") or None,
        "source_type": source.get("source_type") or None,
        "source_status": source.get("source_status", "unknown"),
        "evidence": source.get("text") or None,
    }


async def generate_node(state: AgentState):
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return {
            "messages": [
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "抱歉，当前知识库中的证据不足以可靠回答这个问题。"
                                "建议补充更具体的商品信息，或联系人工客服。"
                            ),
                            "verified_citations": [],
                            "unverified_citations": [],
                        },
                        ensure_ascii=False,
                    )
                )
            ],
            "retrieval_alerts": [
                f"已达到最大检索重试次数 {MAX_RETRIES}，已停止继续检索。"
            ],
        }

    response = await generator_llm.ainvoke(
        [SystemMessage(content=generate_prompt), *state["messages"]]
    )

    sources_by_id = {
        (source.get("document_id", ""), source.get("chunk_id", "")): source
        for source in state.get("retrieved_sources", [])
    }
    verified: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    alerts: list[str] = []
    seen_citations: set[tuple[str, str]] = set()

    for citation in response.citations:
        key = (citation.document_id.strip(), citation.chunk_id.strip())
        if not all(key) or key in seen_citations:
            continue
        seen_citations.add(key)

        source = sources_by_id.get(key)
        payload = _citation_payload(citation, source)
        if source is None:
            unverified.append(payload)
            continue

        verified.append(payload)
        if payload["source_status"] != "provided":
            alerts.append(
                f"引用 {key[1]} 缺少可用的来源 URL，请人工复核。"
            )

    final_content = json.dumps(
        {
            "answer": response.answer.strip(),
            "verified_citations": verified,
            "unverified_citations": unverified,
        },
        ensure_ascii=False,
    )
    return {
        "messages": [AIMessage(content=final_content)],
        "retrieval_alerts": list(dict.fromkeys(alerts)),
    }
