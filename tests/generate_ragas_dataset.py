"""Generate LLM reference answers and actual RAG predictions for Ragas."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from rag.config import MILVUS_URI, llm
from rag.graph.builder import app as agent_graph


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ecommerce" / "shoppingcomp"
DEFAULT_INPUT = DATA_DIR / "questions_sample.jsonl"
DEFAULT_OUTPUT_DIR = DATA_DIR / "ragas_eval"
REFERENCE_VERSION = "reference-v1"


class ReferenceAnswer(BaseModel):
    reference_answer: str = Field(
        description="仅依据标准商品证据编写的简洁中文参考回答"
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def evenly_sample(
    records: list[dict[str, Any]],
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(records):
        return records
    if limit <= 0:
        raise ValueError("limit must be positive")
    step = len(records) / limit
    return [records[int(index * step)] for index in range(limit)]


def build_reference_context(
    task_id: str,
    documents_by_task: dict[str, list[dict[str, Any]]],
    rubrics_by_task: dict[str, list[dict[str, Any]]],
    max_chars: int,
) -> str:
    parts: list[str] = []
    used = 0

    for scene in rubrics_by_task.get(task_id, []):
        rubric = str(scene.get("rubric", "")).strip()
        if not rubric:
            continue
        block = f"场景要求：{scene.get('scene', '')}\n评分标准：{rubric}"
        remaining = max_chars - used
        if remaining <= 0:
            break
        parts.append(block[:remaining])
        used += len(parts[-1])

    # Ensure every reference product gets at least one evidence passage first.
    by_title: dict[str, list[dict[str, Any]]] = {}
    for document in documents_by_task.get(task_id, []):
        by_title.setdefault(document["title"], []).append(document)

    ordered_documents: list[dict[str, Any]] = []
    for documents in by_title.values():
        ordered_documents.append(documents[0])
    seen = {document["chunk_id"] for document in ordered_documents}
    for document in documents_by_task.get(task_id, []):
        if document["chunk_id"] not in seen:
            ordered_documents.append(document)
            seen.add(document["chunk_id"])

    for document in ordered_documents:
        remaining = max_chars - used
        if remaining <= 0:
            break
        block = f"标准商品：{document['title']}\n证据：{document['text']}"
        parts.append(block[:remaining])
        used += len(parts[-1])
    return "\n\n".join(parts)


async def generate_reference(
    question: dict[str, Any],
    context: str,
) -> str:
    if not context.strip():
        raise ValueError("No gold evidence available for reference generation")
    prompt = f"""你是 RAG 评测数据标注员。请根据下面的标准商品和证据，为问题编写一份简洁、事实准确的参考回答。

要求：
1. 只能使用提供的标准商品与证据，不要补充常识。
2. 明确写出推荐或满足条件的商品名称。
3. 对每件商品概括与用户需求直接相关的关键规格、限制或风险。
4. 证据不足时明确说明，不要编造。
5. 不要提及“评测”“标准答案”或“证据编号”。
6. 只输出 JSON：{{"reference_answer": "参考回答"}}。

用户问题：
{question['question']}

标准商品与证据：
{context}
"""
    generator = llm.with_structured_output(ReferenceAnswer, method="json_mode")
    result = await generator.ainvoke(prompt)
    answer = result.reference_answer.strip()
    if not answer:
        raise ValueError("Reference generator returned an empty answer")
    return answer


async def run_rag(question: dict[str, Any]) -> dict[str, Any]:
    inputs = {
        "messages": [HumanMessage(content=question["question"])],
        "filters": {},
        "retry_count": 0,
        "retrieved_sources": [],
        "retrieval_alerts": [],
    }
    result = await agent_graph.ainvoke(inputs)
    final_message = result["messages"][-1]
    content = final_message.content
    parsed = json.loads(content) if isinstance(content, str) else {}
    return {
        "answer": str(parsed.get("answer", content)).strip(),
        "verified_citations": parsed.get("verified_citations", []),
        "unverified_citations": parsed.get("unverified_citations", []),
        "retrieved_contexts": [
            source.get("text", "")
            for source in result.get("retrieved_sources", [])
            if source.get("text")
        ],
        "retrieved_products": [
            source.get("title", "")
            for source in result.get("retrieved_sources", [])
            if source.get("title")
        ],
        "retrieval_alerts": result.get("retrieval_alerts", []),
        "retrieval_count": len(result.get("retrieved_sources", [])),
    }


async def process_question(
    question: dict[str, Any],
    documents_by_task: dict[str, list[dict[str, Any]]],
    rubrics_by_task: dict[str, list[dict[str, Any]]],
    max_reference_chars: int,
) -> dict[str, Any]:
    task_id = question["question_id"]
    reference_context = build_reference_context(
        task_id,
        documents_by_task,
        rubrics_by_task,
        max_reference_chars,
    )
    reference_answer = await generate_reference(question, reference_context)
    rag_result = await run_rag(question)
    return {
        "question_id": task_id,
        "question": question["question"],
        "reference_answer": reference_answer,
        "reference_context": reference_context,
        "expected_product_names": question["expected_product_names"],
        "expected_document_ids": question["expected_document_ids"],
        "expected_chunk_ids": question["expected_chunk_ids"],
        **rag_result,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_milvus_available() -> None:
    parsed = urlsplit(MILVUS_URI)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 19530
    try:
        with socket.create_connection((host, port), timeout=3):
            return
    except OSError as exc:
        raise RuntimeError(
            f"Milvus is not available at {host}:{port}. Start it and retry."
        ) from exc


async def main_async(args: argparse.Namespace) -> None:
    ensure_milvus_available()
    questions = evenly_sample(read_jsonl(args.input), args.limit)
    documents_by_task: dict[str, list[dict[str, Any]]] = {}
    for document in read_jsonl(DATA_DIR / "documents.jsonl"):
        documents_by_task.setdefault(document["task_id"], []).append(document)

    rubrics_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in read_jsonl(DATA_DIR / "raw" / "ShoppingComp_97_zh.jsonl"):
        rubrics_by_task[str(task["uuid"])] = task.get("scene_list", [])

    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def guarded(question: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            return await process_question(
                question,
                documents_by_task,
                rubrics_by_task,
                args.max_reference_chars,
            )

    records = await asyncio.gather(*(guarded(question) for question in questions))
    records = [record for record in records if record["answer"]]
    if not records:
        raise RuntimeError("No successful RAG samples were generated")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "samples.jsonl", records)
    write_jsonl(
        output_dir / "ragas_dataset.jsonl",
        [
            {
                "user_input": record["question"],
                "response": record["answer"],
                "retrieved_contexts": record["retrieved_contexts"],
                "reference": record["reference_answer"],
                "question_id": record["question_id"],
            }
            for record in records
        ],
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_version": REFERENCE_VERSION,
        "reference_generator_model": os.getenv("LLM_MODEL", "unknown"),
        "reference_is_llm_generated": True,
        "reference_review_status": "unreviewed",
        "sample_count": len(records),
        "source_questions_sha256": sha256_file(args.input),
        "outputs": {
            "samples": "samples.jsonl",
            "ragas_dataset": "ragas_dataset.jsonl",
        },
    }
    (output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(records)} reference answers and RAG responses.")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-reference-chars", type=int, default=24000)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
