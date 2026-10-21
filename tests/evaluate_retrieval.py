"""Compare product-level retrieval with and without reranking."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from rag.config import RERANK_TOP_PRODUCTS, RETRIEVAL_TOP_K
from rag.db.milvus_client import hybrid_search
from rag.tools.retriever import group_results_by_product, rank_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "questions_sample.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "ecommerce" / "shoppingcomp" / "evaluation"
)


def read_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if limit is not None and len(records) >= limit:
            break
    return records


def _score_products(
    expected: set[str],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    titles = [
        str(source.get("title", "")).casefold()
        for source in selected
        if source.get("title")
    ]
    unique_titles = list(dict.fromkeys(titles))
    correct = expected.intersection(unique_titles)
    precision = len(correct) / max(1, len(unique_titles))
    recall = len(correct) / max(1, len(expected))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    first_correct_rank = next(
        (
            rank
            for rank, title in enumerate(unique_titles, start=1)
            if title in expected
        ),
        None,
    )
    return {
        "selected_products": [
            source.get("title", "") for source in selected if source.get("title")
        ],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit": bool(correct),
        "reciprocal_rank": (
            1 / first_correct_rank if first_correct_rank is not None else 0.0
        ),
    }


async def evaluate_question(
    question: dict[str, Any],
    semaphore: asyncio.Semaphore,
    top_k: int,
    top_products: int,
    modes: list[str],
) -> dict[str, Any]:
    async with semaphore:
        candidates = await hybrid_search(question["question"], top_k=top_k)

    expected_products = {
        name.casefold() for name in question["expected_product_names"]
    }
    candidate_products = {
        str(source.get("title", "")).casefold()
        for source in candidates
        if source.get("title")
    }
    candidate_correct = expected_products.intersection(candidate_products)
    candidate_precision = len(candidate_correct) / max(1, len(candidate_products))
    candidate_recall = len(candidate_correct) / max(1, len(expected_products))

    result: dict[str, Any] = {
        "question_id": question["question_id"],
        "candidate": {
            "candidate_product_count": len(candidate_products),
            "precision": candidate_precision,
            "recall": candidate_recall,
            "hit": bool(candidate_correct),
        },
        "modes": {},
    }

    for mode in modes:
        selected, alerts = await rank_candidates(
            question["question"],
            candidates,
            use_rerank=mode == "rerank",
            top_products=top_products,
            evidence_per_product=1,
        )
        result["modes"][mode] = {
            **_score_products(expected_products, selected),
            "alerts": alerts,
        }
    result["group_count"] = len(group_results_by_product(candidates))
    return result


def summarize(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    rows = [result["modes"][mode] for result in results]
    count = max(1, len(rows))
    return {
        "queries": len(rows),
        "hit_rate": sum(row["hit"] for row in rows) / count,
        "mean_precision": sum(row["precision"] for row in rows) / count,
        "mean_recall": sum(row["recall"] for row in rows) / count,
        "mean_f1": sum(row["f1"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "queries_with_rerank_alerts": sum(bool(row["alerts"]) for row in rows),
    }


async def evaluate(
    questions: list[dict[str, Any]],
    top_k: int,
    top_products: int,
    modes: list[str],
    max_workers: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max(1, max_workers))
    results = await asyncio.gather(
        *(
            evaluate_question(
                question,
                semaphore=semaphore,
                top_k=top_k,
                top_products=top_products,
                modes=modes,
            )
            for question in questions
        )
    )

    candidate_count = max(1, len(results))
    report = {
        "settings": {
            "top_k": top_k,
            "top_products": top_products,
            "modes": modes,
            "queries": len(results),
        },
        "candidate_summary": {
            "hit_rate": sum(row["candidate"]["hit"] for row in results)
            / candidate_count,
            "mean_precision": sum(
                row["candidate"]["precision"] for row in results
            )
            / candidate_count,
            "mean_recall": sum(row["candidate"]["recall"] for row in results)
            / candidate_count,
        },
        "mode_summaries": {
            mode: summarize(results, mode) for mode in modes
        },
        "results": results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument(
        "--top-products",
        type=int,
        default=RERANK_TOP_PRODUCTS,
    )
    parser.add_argument(
        "--mode",
        choices=["compare", "rerank", "no-rerank"],
        default="compare",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    questions = read_jsonl(args.input, args.limit)
    if not questions:
        raise ValueError("No evaluation questions found")
    modes = (
        ["no-rerank", "rerank"]
        if args.mode == "compare"
        else [args.mode]
    )

    report = asyncio.run(
        evaluate(
            questions,
            top_k=args.top_k,
            top_products=args.top_products,
            modes=modes,
            max_workers=args.max_workers,
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "retrieval_report.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["mode_summaries"], ensure_ascii=False, indent=2))
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
