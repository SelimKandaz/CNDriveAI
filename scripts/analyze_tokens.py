from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def _template(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> list[int]:
    options = {"tokenize": True, "add_generation_prompt": generation}
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **options)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **options)
    return _input_ids(encoded)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def analyze(data: Path, tokenizer: Any, limits: list[int]) -> dict[str, Any]:
    lengths: list[int] = []
    response_lengths: list[int] = []
    mismatches = 0
    truncation: dict[str, dict[str, int | float]] = {
        str(limit): {
            "examples": 0,
            "any_assistant_tokens_removed": 0,
            "assistant_tokens_removed": 0,
            "assistant_tokens_total": 0,
        }
        for limit in limits
    }
    for _line_no, line in enumerate(data.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record["messages"]
        full = _template(tokenizer, messages, False)
        prompt = _template(tokenizer, messages[:-1], True)
        lengths.append(len(full))
        prefix_ok = full[: len(prompt)] == prompt
        if not prefix_ok:
            mismatches += 1
            # The trainer refuses ambiguous masks; do not report plausible but
            # incorrect assistant-truncation estimates for these examples.
            continue
        response_count = len(full) - len(prompt)
        response_lengths.append(response_count)
        for limit in limits:
            kept_assistant = max(0, min(len(full), limit) - len(prompt))
            removed = response_count - kept_assistant
            item = truncation[str(limit)]
            item["examples"] += 1
            item["assistant_tokens_total"] += response_count
            item["assistant_tokens_removed"] += removed
            item["any_assistant_tokens_removed"] += int(removed > 0)
    return {
        "data_file": data.name,
        "record_count": len(lengths),
        "prompt_full_prefix_mismatches": mismatches,
        "full_sequence_tokens": {
            "p50": _percentile(lengths, 0.50),
            "p75": _percentile(lengths, 0.75),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths, default=0),
        },
        "assistant_response_tokens": {
            "p50": _percentile(response_lengths, 0.50),
            "p75": _percentile(response_lengths, 0.75),
            "p90": _percentile(response_lengths, 0.90),
            "p95": _percentile(response_lengths, 0.95),
            "max": max(response_lengths, default=0),
        },
        "truncation_by_max_sequence_length": {
            limit: {
                **values,
                "examples_with_assistant_truncation_pct": round(
                    100
                    * int(values["any_assistant_tokens_removed"])
                    / max(1, int(values["examples"])),
                    2,
                ),
                "assistant_tokens_removed_pct": round(
                    100
                    * int(values["assistant_tokens_removed"])
                    / max(1, int(values["assistant_tokens_total"])),
                    2,
                ),
            }
            for limit, values in truncation.items()
        },
        "mask_policy": (
            "Exact tokenizer chat-template prefix required; ambiguous records are not trained."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-site", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/token_analysis.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runtime_rocm import bootstrap_torch

    bootstrap_torch(args.runtime_site)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    result = analyze(args.data, processor.tokenizer, [256, 384, 512, 768, 1024])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
