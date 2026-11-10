"""Compute paired metric deltas and confidence intervals for two RAGAS runs."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_delta_ci(
    deltas: list[float],
    *,
    seed: int = 20261110,
    iterations: int = 10000,
) -> dict[str, float]:
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(iterations)
    )
    lower_index = max(0, int(iterations * 0.025) - 1)
    upper_index = min(iterations - 1, int(iterations * 0.975))
    return {
        "lower": means[lower_index],
        "upper": means[upper_index],
    }


def sign_test_p_value(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = 2 * sum(
        math.comb(trials, index) for index in range(tail + 1)
    ) / (2**trials)
    return min(1.0, probability)


def paired_metric(
    metric: str,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    baseline = {
        row["question_id"]: row[metric]
        for row in baseline_rows
        if row.get(metric) is not None
    }
    candidate = {
        row["question_id"]: row[metric]
        for row in candidate_rows
        if row.get(metric) is not None
    }
    question_ids = sorted(set(baseline).intersection(candidate))
    if not question_ids:
        return None

    before = [float(baseline[question_id]) for question_id in question_ids]
    after = [float(candidate[question_id]) for question_id in question_ids]
    deltas = [
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(before, after)
    ]
    wins = sum(delta > 1e-12 for delta in deltas)
    losses = sum(delta < -1e-12 for delta in deltas)
    ties = len(deltas) - wins - losses
    baseline_mean = sum(before) / len(before)
    candidate_mean = sum(after) / len(after)
    delta = candidate_mean - baseline_mean
    ci = bootstrap_delta_ci(deltas)
    return {
        "paired_samples": len(deltas),
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "delta": delta,
        "delta_percentage_points": delta * 100,
        "relative_change_percent": (
            delta / baseline_mean * 100 if baseline_mean else None
        ),
        "delta_ci95": ci,
        "delta_ci95_percentage_points": {
            "lower": ci["lower"] * 100,
            "upper": ci["upper"] * 100,
        },
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p_value": sign_test_p_value(wins, losses),
        "per_question": [
            {
                "question_id": question_id,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta": delta_value,
            }
            for question_id, baseline_value, candidate_value, delta_value in zip(
                question_ids,
                before,
                after,
                deltas,
                strict=True,
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-faithfulness-delta", type=float, default=None)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    args = parser.parse_args()

    baseline = read_json(args.baseline)
    candidate = read_json(args.candidate)
    comparisons = {}
    for metric in ("faithfulness", "answer_relevancy"):
        result = paired_metric(
            metric,
            baseline.get(metric, []),
            candidate.get(metric, []),
        )
        if result is not None:
            comparisons[metric] = result

    report = {
        "baseline_file": str(args.baseline),
        "candidate_file": str(args.candidate),
        "confidence_method": (
            "paired percentile bootstrap, 10000 resamples, seed 20261110"
        ),
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))

    failures: list[str] = []
    faithfulness = comparisons.get("faithfulness")
    if faithfulness:
        if (
            args.min_faithfulness_delta is not None
            and faithfulness["delta"] < args.min_faithfulness_delta
        ):
            failures.append(
                "faithfulness delta "
                f"{faithfulness['delta']:.4f} is below "
                f"{args.min_faithfulness_delta:.4f}"
            )
        if (
            args.min_faithfulness is not None
            and faithfulness["candidate_mean"] < args.min_faithfulness
        ):
            failures.append(
                "faithfulness "
                f"{faithfulness['candidate_mean']:.4f} is below "
                f"{args.min_faithfulness:.4f}"
            )
    elif args.min_faithfulness is not None or args.min_faithfulness_delta is not None:
        failures.append("faithfulness comparison is unavailable")

    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
