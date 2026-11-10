"""Evaluate generated RAG samples with Ragas metrics."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "data" / "ecommerce" / "shoppingcomp" / "ragas_eval"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _score_samples(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = 0
    citation_total = unverified_total = 0
    for sample in samples:
        expected = {name.casefold() for name in sample["expected_product_names"]}
        retrieved = {
            str(name).casefold()
            for name in sample.get("retrieved_products", [])
            if str(name).strip()
        }
        correct = expected.intersection(retrieved)
        true_positive += len(correct)
        false_positive += len(retrieved - expected)
        false_negative += len(expected - correct)

        verified = sample.get("verified_citations", [])
        unverified = sample.get("unverified_citations", [])
        citation_total += len(verified) + len(unverified)
        unverified_total += len(unverified)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "product_precision": precision,
        "product_recall": recall,
        "product_f1": f1,
        "citation_count": citation_total,
        "unverified_citation_count": unverified_total,
        "unverified_citation_rate": (
            unverified_total / citation_total if citation_total else 0.0
        ),
    }


def split_response(text: str, max_chars: int = 140) -> list[str]:
    parts = [
        part.strip()
        for part in re.split(r"(?<=[。！？!?；;])\s*", text)
        if part.strip()
    ]
    segments: list[str] = []
    current = ""
    for part in parts:
        if current and len(current) + len(part) > max_chars:
            segments.append(current)
            current = part
        else:
            current += part
    if current:
        segments.append(current)
    return segments


async def score_faithfulness(
    sample: dict[str, Any],
    llm: Any,
    max_chars: int = 140,
    retries: int = 2,
) -> dict[str, Any]:
    metric = Faithfulness(llm)
    segments = split_response(sample["answer"], max_chars=max_chars)
    values: list[float] = []
    weights: list[int] = []

    async def score_segment(segment: str) -> dict[str, Any]:
        last_error = ""
        for attempt in range(retries + 1):
            try:
                result = await metric.ascore(
                    user_input=sample["question"],
                    response=segment,
                    retrieved_contexts=sample["retrieved_contexts"],
                )
                value = float(result.value)
                return {
                    "segment": segment,
                    "score": value,
                    "error": None,
                    "attempts": attempt + 1,
                }
            except Exception as exc:
                last_error = str(exc)
                if attempt < retries:
                    await asyncio.sleep(1.5**attempt)
        return {
            "segment": segment,
            "score": None,
            "error": last_error,
            "attempts": retries + 1,
        }

    segment_results = list(
        await asyncio.gather(*(score_segment(segment) for segment in segments))
    )
    for result in segment_results:
        if result["score"] is not None:
            values.append(float(result["score"]))
            weights.append(len(result["segment"]))

    score = (
        sum(value * weight for value, weight in zip(values, weights))
        / sum(weights)
        if weights
        else None
    )
    return {
        "faithfulness": score,
        "segments": segment_results,
        "scored_segments": len(values),
        "total_segments": len(segments),
    }


async def score_answer_relevancy(
    sample: dict[str, Any],
    llm: Any,
    embeddings: Any,
) -> dict[str, Any]:
    try:
        metric = AnswerRelevancy(llm=llm, embeddings=embeddings)
        result = await metric.ascore(
            user_input=sample["question"],
            response=sample["answer"],
        )
        return {"answer_relevancy": float(result.value), "error": None}
    except Exception as exc:
        return {"answer_relevancy": None, "error": str(exc)}


async def run_evaluation(
    samples: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    metric_names: list[str],
    evaluator: Any,
    embeddings: Any,
    workers: int,
    segment_chars: int,
    retries: int,
) -> dict[str, Any]:
    del dataset_rows  # Samples contain every required field and audit metadata.
    semaphore = asyncio.Semaphore(max(1, workers))

    async def one(sample: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            row: dict[str, Any] = {"question_id": sample["question_id"]}
            if "faithfulness" in metric_names:
                row["faithfulness_result"] = await score_faithfulness(
                    sample,
                    evaluator,
                    max_chars=segment_chars,
                    retries=retries,
                )
            if "answer_relevancy" in metric_names:
                row["answer_relevancy_result"] = await score_answer_relevancy(
                    sample,
                    evaluator,
                    embeddings,
                )
            return row

    rows = await asyncio.gather(*(one(sample) for sample in samples))
    return {
        "faithfulness": [
            {
                "question_id": row["question_id"],
                **row["faithfulness_result"],
            }
            for row in rows
            if "faithfulness_result" in row
        ],
        "answer_relevancy": [
            {
                "question_id": row["question_id"],
                **row["answer_relevancy_result"],
            }
            for row in rows
            if "answer_relevancy_result" in row
        ],
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = 20261110,
    iterations: int = 5000,
) -> dict[str, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(iterations)
    )
    lower_index = max(0, int(iterations * 0.025) - 1)
    upper_index = min(iterations - 1, int(iterations * 0.975))
    return {
        "lower": samples[lower_index],
        "upper": samples[upper_index],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--samples",
        type=Path,
        default=None,
        help="样本 JSONL；默认读取 input-dir/samples.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Ragas 数据集 JSONL；仅用于校验样本规模",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="输出文件前缀，例如 improved_",
    )
    parser.add_argument(
        "--segment-chars",
        type=int,
        default=140,
        help="忠实度评分前每个片段的最大字符数",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="单个片段评分失败后的重试次数",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"

    metric_names = [
        name.strip() for name in args.metrics.split(",") if name.strip()
    ]
    unknown = set(metric_names) - {"faithfulness", "answer_relevancy"}
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    samples_path = args.samples or args.input_dir / "samples.jsonl"
    dataset_path = args.dataset or args.input_dir / "ragas_dataset.jsonl"
    dataset_rows = read_jsonl(dataset_path)
    samples = read_jsonl(samples_path)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        dataset_rows = dataset_rows[: args.limit]
        samples = samples[: args.limit]
    if not dataset_rows:
        raise ValueError("Ragas dataset is empty")
    if not samples:
        raise ValueError("Ragas samples are empty")
    if args.segment_chars < 40:
        raise ValueError("segment-chars must be at least 40")
    if args.retries < 0:
        raise ValueError("retries cannot be negative")

    evaluator = llm_factory(
        os.getenv("RAGAS_EVALUATOR_MODEL", "deepseek-flash"),
        client=AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        ),
        max_tokens=8192,
        temperature=0,
    )
    embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        ),
        model=os.getenv("EMBEDDING_MODEL", ""),
    )

    score_data = asyncio.run(
        run_evaluation(
            samples,
            dataset_rows,
            metric_names,
            evaluator,
            embeddings,
            args.workers,
            args.segment_chars,
            args.retries,
        )
    )
    output_dir = args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{args.output_prefix}ragas_scores.json").write_text(
        json.dumps(score_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_summary: dict[str, Any] = {}
    if score_data["faithfulness"]:
        scored = [
            row["faithfulness"]
            for row in score_data["faithfulness"]
            if row["faithfulness"] is not None
        ]
        metric_summary["faithfulness"] = (
            sum(scored) / len(scored) if scored else None
        )
        metric_summary["faithfulness_ci95"] = bootstrap_mean_ci(scored)
        metric_summary["faithfulness_scored_samples"] = len(scored)
        metric_summary["faithfulness_sample_coverage"] = (
            len(scored) / len(score_data["faithfulness"])
            if score_data["faithfulness"]
            else None
        )
        metric_summary["faithfulness_scored_segments"] = sum(
            row["scored_segments"] for row in score_data["faithfulness"]
        )
        metric_summary["faithfulness_total_segments"] = sum(
            row["total_segments"] for row in score_data["faithfulness"]
        )
        metric_summary["faithfulness_segment_errors"] = sum(
            row["total_segments"] - row["scored_segments"]
            for row in score_data["faithfulness"]
        )
        metric_summary["faithfulness_segment_coverage"] = (
            metric_summary["faithfulness_scored_segments"]
            / metric_summary["faithfulness_total_segments"]
            if metric_summary["faithfulness_total_segments"]
            else None
        )
    if score_data["answer_relevancy"]:
        scored = [
            row["answer_relevancy"]
            for row in score_data["answer_relevancy"]
            if row["answer_relevancy"] is not None
        ]
        metric_summary["answer_relevancy"] = (
            sum(scored) / len(scored) if scored else None
        )
        metric_summary["answer_relevancy_ci95"] = bootstrap_mean_ci(scored)
        metric_summary["answer_relevancy_scored_samples"] = len(scored)
        metric_summary["answer_relevancy_sample_coverage"] = (
            len(scored) / len(score_data["answer_relevancy"])
            if score_data["answer_relevancy"]
            else None
        )
    if "faithfulness" in metric_summary:
        metric_summary["unsupported_claim_rate_proxy"] = (
            1 - metric_summary["faithfulness"]
            if metric_summary["faithfulness"] is not None
            else None
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(samples_path),
        "sample_count": len(samples),
        "metrics": metric_names,
        "evaluator_model": os.getenv("RAGAS_EVALUATOR_MODEL", "deepseek-flash"),
        "metric_mode": {
            "faithfulness": (
                "sentence-segmented, character-weighted aggregation; "
                f"max {args.segment_chars} chars per segment"
            ),
            "answer_relevancy": "whole response",
        },
        "ragas_summary": metric_summary,
        "custom_summary": _score_samples(samples),
        "limitations": [
            "reference answers are generated by the project LLM and have not been human-reviewed",
            "Ragas uses LLM-as-a-judge and results depend on the evaluator model and prompts",
            "unsupported_claim_rate_proxy is not a universal hallucination rate",
            "faithfulness is aggregated from sentence-level Ragas scores to avoid long-output truncation",
        ],
    }
    (output_dir / f"{args.output_prefix}ragas_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
