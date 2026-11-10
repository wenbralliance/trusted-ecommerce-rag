"""Add ranking metrics and paired confidence intervals to a saved retrieval run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_retrieval import (
    _score_products,
    paired_mode_comparison,
    summarize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "evaluation"
    / "retrieval_report.json"
)
DEFAULT_QUESTIONS = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "questions_sample.jsonl"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "ecommerce"
    / "shoppingcomp"
    / "evaluation"
    / "retrieval_comparison_report.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    questions = {
        row["question_id"]: row for row in read_jsonl(args.questions)
    }
    enriched_results = []
    for result in report["results"]:
        question = questions[result["question_id"]]
        expected = {
            name.casefold()
            for name in question["expected_product_names"]
        }
        modes = {}
        for mode, mode_result in result["modes"].items():
            scores = _score_products(
                expected,
                [
                    {"title": title}
                    for title in mode_result["selected_products"]
                ],
            )
            modes[mode] = {**mode_result, **scores}
        enriched_results.append({**result, "modes": modes})

    modes = list(report["settings"]["modes"])
    summary = {
        "source_report": str(args.report),
        "settings": report["settings"],
        "candidate_summary": report["candidate_summary"],
        "mode_summaries": {
            mode: summarize(enriched_results, mode) for mode in modes
        },
        "paired_comparison": paired_mode_comparison(
            enriched_results,
            "no-rerank",
            "rerank",
        ),
        "method": (
            "Product-level binary relevance; nDCG@5 and AP@5 recomputed from "
            "saved ranked product titles. Paired percentile bootstrap uses "
            "10000 resamples with seed 20261110."
        ),
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
