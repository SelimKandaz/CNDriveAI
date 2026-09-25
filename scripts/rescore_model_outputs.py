"""Recompute aggregate metrics from ignored raw completions using the current rubric."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_model import summarize  # noqa: E402

from cndriveai.dataset import heldout_records  # noqa: E402
from cndriveai.scoring import score_answer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--model-name", choices=("base", "adapter"), required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()

    tasks = {item["id"]: item for item in heldout_records()}
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("id")
        if task_id not in tasks or task_id in seen:
            raise ValueError(f"unknown or duplicate task id on line {line_no}")
        seen.add(task_id)
        truncated = bool(row["scores"]["truncated_output"])
        scores = score_answer(
            tasks[task_id],
            str(row["answer"]),
            args.max_new_tokens if truncated else 0,
            args.max_new_tokens,
            not truncated,
            float(row["scores"]["latency_seconds"]),
        )
        items.append({"id": task_id, "category": tasks[task_id]["category"], "scores": scores})

    summary = summarize(items, args.model_name)
    summary["configuration"] = {
        "base_model": "Qwen/Qwen3.5-9B",
        "base_snapshot_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "quantization": "Quanto int4 weights",
        "compute_dtype": "bfloat16",
        "generation": "greedy, do_sample=false",
        "max_new_tokens": args.max_new_tokens,
        "adapter_loaded": args.model_name == "adapter",
        "prompt_contract": (
            "same CNDriveAI system message used for training plus held-out user prompt"
        ),
        "metric_source": "raw local completions rescored with current deterministic rubric",
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
