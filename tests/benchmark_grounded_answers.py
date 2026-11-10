"""Generate paired baseline and evidence-grounded answers on the same contexts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.config import llm
from rag.graph.grounding import (
    GroundedAnswer,
    generate_grounded_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "ragas_eval"
    / "samples.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "evaluation"
    / "groundedness"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sources_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "document_id": f"evaluation-doc-{index}",
            "chunk_id": f"evaluation-chunk-{index}",
            "title": f"检索证据 {index}",
            "text": context,
            "source_status": "provided",
        }
        for index, context in enumerate(record["retrieved_contexts"], start=1)
    ]


async def ground_record(
    record: dict[str, Any],
    semaphore: asyncio.Semaphore,
    retries: int,
) -> dict[str, Any]:
    async with semaphore:
        model = llm.with_structured_output(GroundedAnswer, method="json_mode")

        async def generate(prompt: str) -> GroundedAnswer:
            last_error: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return await model.ainvoke(prompt)
                except Exception as exc:
                    last_error = exc
                    if attempt < retries:
                        await asyncio.sleep(1.5**attempt)
            assert last_error is not None
            raise last_error

        response, _, grounding = await generate_grounded_answer(
            record["question"],
            sources_from_record(record),
            generate,
        )

    answer = response.answer.strip() or "\n".join(
        claim.text.strip()
        for claim in response.claims
        if claim.text.strip()
    )
    if not answer:
        raise ValueError(
            f"Grounded generator returned an empty answer: "
            f"{record['question_id']}"
        )

    return {
        **record,
        "answer": answer,
        "grounding": grounding.as_dict(),
        "grounded_claims": [
            {
                "text": claim.text,
                "evidence_ids": claim.evidence_ids,
            }
            for claim in response.claims
        ],
        "missing_information": response.missing_information,
        "experiment": {
            "pipeline": "claim_first_grounded_generation_v1",
            "baseline_answer_chars": len(record["answer"]),
            "candidate_answer_chars": len(answer),
        },
    }


async def run(args: argparse.Namespace) -> None:
    records = read_jsonl(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        records = records[: args.limit]
    if not records:
        raise ValueError("No evaluation samples found")

    semaphore = asyncio.Semaphore(max(1, args.workers))
    candidates = await asyncio.gather(
        *(
            ground_record(record, semaphore, args.retries)
            for record in records
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "baseline_samples.jsonl", records)
    write_jsonl(args.output_dir / "candidate_samples.jsonl", candidates)
    for label, rows in (
        ("baseline", records),
        ("candidate", candidates),
    ):
        write_jsonl(
            args.output_dir / f"{label}_dataset.jsonl",
            [
                {
                    "user_input": row["question"],
                    "response": row["answer"],
                    "retrieved_contexts": row["retrieved_contexts"],
                    "reference": row["reference_answer"],
                    "question_id": row["question_id"],
                }
                for row in rows
            ],
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "claim_first_grounded_generation_v1",
        "source_samples": str(args.input),
        "source_samples_sha256": sha256_file(args.input),
        "sample_count": len(records),
        "model": os.getenv("LLM_MODEL", "unknown"),
        "controlled_variable": (
            "Both variants use the exact same question, retrieved contexts, "
            "reference answer and citation set."
        ),
        "outputs": {
            "baseline": "baseline_samples.jsonl",
            "candidate": "candidate_samples.jsonl",
        },
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Generated {len(candidates)} paired grounded answers in "
        f"{args.output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
