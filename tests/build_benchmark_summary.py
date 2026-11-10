"""Build one compact machine-readable RAG benchmark summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "evaluation"
)
LIVE_RAGAS_REPORT = (
    EVALUATION_DIR / "live_ragas_20261110" / "live_ragas_report.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=EVALUATION_DIR / "retrieval_comparison_report.json",
    )
    parser.add_argument(
        "--answer-comparison",
        type=Path,
        default=EVALUATION_DIR / "groundedness" / "comparison_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVALUATION_DIR / "benchmark_summary.json",
    )
    args = parser.parse_args()

    retrieval = read_json(args.retrieval)
    answer = read_json(args.answer_comparison)
    retrieval_pairs = retrieval["paired_comparison"]["metrics"]
    answer_pairs = answer["comparisons"]
    live_report = (
        read_json(LIVE_RAGAS_REPORT) if LIVE_RAGAS_REPORT.exists() else None
    )
    summary = {
        "dataset": "ShoppingComp zh",
        "retrieval": {
            "queries": retrieval["settings"]["queries"],
            "top_k": retrieval["settings"]["top_k"],
            "top_products": retrieval["settings"]["top_products"],
            "recall": retrieval_pairs["recall"],
            "ndcg_at_5": retrieval_pairs["ndcg_at_5"],
            "average_precision": retrieval_pairs.get("average_precision"),
            "candidate_search_latency_ms": retrieval[
                "candidate_summary"
            ].get("search_latency_ms"),
            "no_rerank_latency_ms": retrieval["mode_summaries"][
                "no-rerank"
            ].get("ranking_latency_ms"),
            "rerank_latency_ms": retrieval["mode_summaries"][
                "rerank"
            ].get("ranking_latency_ms"),
        },
        "answer_quality": {
            "paired_samples": answer_pairs["faithfulness"]["paired_samples"],
            "faithfulness": answer_pairs["faithfulness"],
            "answer_relevancy": answer_pairs.get("answer_relevancy"),
        },
        "live_end_to_end": (
            {
                "sample_count": live_report["sample_count"],
                **live_report["ragas_summary"],
                "custom_summary": live_report["custom_summary"],
            }
            if live_report
            else None
        ),
        "limitations": [
            "Answer-quality experiment contains 10 LLM-generated samples.",
            "Reference answers are not human-reviewed.",
            "RAGAS uses LLM-as-a-judge.",
            "Answer relevancy decreased and requires further optimization.",
            "Retrieval latency is a single local run and is not a load test.",
            "Controlled generation metrics and live end-to-end metrics use different retrieval contexts.",
        ],
        "reports": {
            "retrieval": str(args.retrieval.relative_to(PROJECT_ROOT)),
            "answer_comparison": str(
                args.answer_comparison.relative_to(PROJECT_ROOT)
            ),
            "live_ragas": (
                str(LIVE_RAGAS_REPORT.relative_to(PROJECT_ROOT))
                if live_report
                else None
            ),
        },
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
