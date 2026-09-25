"""Build paired aggregate metrics from two ignored held-out completion files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_model import summarize  # noqa: E402

from cndriveai.dataset import heldout_records  # noqa: E402
from cndriveai.scoring import score_answer  # noqa: E402

METRICS = (
    "technical_correctness",
    "evidence_grounding",
    "hallucination_control",
    "uncertainty_discipline",
    "code_correctness",
    "command_safety",
    "usefulness",
    "output_completeness",
)


def load_run(path: Path, tasks: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("id")
        if task_id not in tasks or task_id in rows:
            raise ValueError(f"unknown or duplicate task id on line {line_no}")
        rows[task_id] = row
    if rows.keys() != tasks.keys():
        missing = sorted(tasks.keys() - rows.keys())
        extra = sorted(rows.keys() - tasks.keys())
        raise ValueError(f"run must cover the exact held-out set; missing={missing}, extra={extra}")
    return rows


def rescore_run(
    rows: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    model_name: str,
    max_new_tokens: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    items = []
    for task_id, row in rows.items():
        truncated = bool(row["scores"]["truncated_output"])
        scores = score_answer(
            tasks[task_id],
            str(row["answer"]),
            max_new_tokens if truncated else 0,
            max_new_tokens,
            not truncated,
            float(row["scores"]["latency_seconds"]),
        )
        items.append({"id": task_id, "category": tasks[task_id]["category"], "scores": scores})
    return summarize(items, model_name), {item["id"]: item for item in items}


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def _paired_summary(
    base: dict[str, dict[str, Any]], adapter: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids = [
        task_id
        for task_id in base.keys() & adapter.keys()
        if not base[task_id]["scores"]["truncated_output"]
        and not adapter[task_id]["scores"]["truncated_output"]
    ]

    def metrics_for(task_ids: list[str]) -> dict[str, Any]:
        result = {}
        for metric in METRICS:
            left = [base[item]["scores"][metric] for item in task_ids]
            right = [adapter[item]["scores"][metric] for item in task_ids]
            pairs = [
                (float(a), float(b))
                for a, b in zip(left, right, strict=True)
                if isinstance(a, (int, float)) and isinstance(b, (int, float))
            ]
            result[metric] = {
                "base_mean": _mean([a for a, _ in pairs]),
                "adapter_mean": _mean([b for _, b in pairs]),
                "delta": _mean([b - a for a, b in pairs]),
                "paired_count": len(pairs),
            }
        return result

    changed: dict[str, list[str]] = {"improved": [], "regressed": [], "unchanged": []}
    for task_id in ids:
        base_incorrect = base[task_id]["scores"]["incorrect_output"]
        adapter_incorrect = adapter[task_id]["scores"]["incorrect_output"]
        if base_incorrect and not adapter_incorrect:
            changed["improved"].append(task_id)
        elif adapter_incorrect and not base_incorrect:
            changed["regressed"].append(task_id)
        elif base_incorrect == adapter_incorrect:
            changed["unchanged"].append(task_id)

    categories = sorted({base[item]["category"] for item in ids})
    by_category = {}
    for category in categories:
        category_ids = [item for item in ids if base[item]["category"] == category]
        by_category[category] = {
            "paired_count": len(category_ids),
            "dimensions": metrics_for(category_ids),
        }

    latency_deltas = [
        adapter[item]["scores"]["latency_seconds"] - base[item]["scores"]["latency_seconds"]
        for item in ids
    ]
    return {
        "paired_complete_task_count": len(ids),
        "paired_dimensions": metrics_for(ids),
        "incorrect_task_changes": {
            key: {"count": len(value), "task_ids": sorted(value)} for key, value in changed.items()
        },
        "mean_paired_latency_delta_seconds": _mean(latency_deltas),
        "categories": by_category,
    }


def build_comparison(
    base_rows: dict[str, dict[str, Any]],
    adapter_rows: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    max_new_tokens: int = 192,
) -> dict[str, Any]:
    if base_rows.keys() != adapter_rows.keys():
        raise ValueError("base and adapter runs must contain identical task IDs")
    base_metrics, rescored_base = rescore_run(base_rows, tasks, "base", max_new_tokens)
    adapter_metrics, rescored_adapter = rescore_run(adapter_rows, tasks, "adapter", max_new_tokens)
    return {
        "suite": "synthetic-heldout-v1",
        "max_new_tokens": max_new_tokens,
        "base": base_metrics,
        "adapter": adapter_metrics,
        "paired_complete_comparison": _paired_summary(rescored_base, rescored_adapter),
        "limitations": [
            "The held-out set is synthetic and template-based, not an independent industry "
            "benchmark.",
            "Scores are deterministic lexical/syntax proxies; manual review is required.",
            "Python code is syntax-scored only and is not executed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    tasks = {item["id"]: item for item in heldout_records()}
    base_rows = load_run(args.base, tasks)
    adapter_rows = load_run(args.adapter, tasks)
    result = build_comparison(base_rows, adapter_rows, tasks, args.max_new_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["paired_complete_comparison"], indent=2))


if __name__ == "__main__":
    main()
