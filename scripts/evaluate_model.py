"""Offline base/adapter evaluator; raw completions remain in ignored local output."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_rocm import bootstrap_torch  # noqa: E402

from cndriveai.dataset import evaluation_messages, heldout_records  # noqa: E402
from cndriveai.scoring import score_answer  # noqa: E402


def format_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in evaluation_messages(prompt)
    ]
    try:
        return str(
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )
    except TypeError:
        return str(
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )


def summarize(items: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    completed = [item for item in items if not item["scores"]["truncated_output"]]
    metric_names = [
        "technical_correctness",
        "evidence_grounding",
        "hallucination_control",
        "uncertainty_discipline",
        "code_correctness",
        "command_safety",
        "usefulness",
        "output_completeness",
    ]
    categories: dict[str, Any] = {}
    for category in sorted({item["category"] for item in items}):
        group = [item for item in completed if item["category"] == category]
        categories[category] = {
            "scored_count": len(group),
            "truncated_count": sum(
                item["category"] == category and item["scores"]["truncated_output"]
                for item in items
            ),
            "means": {
                metric: round(
                    statistics.mean(
                        item["scores"][metric]
                        for item in group
                        if isinstance(item["scores"][metric], (int, float))
                    ),
                    3,
                )
                if any(isinstance(item["scores"][metric], (int, float)) for item in group)
                else None
                for metric in metric_names
            },
        }
    return {
        "model": model_name,
        "task_count": len(items),
        "scored_non_truncated_count": len(completed),
        "truncated_count": len(items) - len(completed),
        "incorrect_non_truncated_count": sum(
            item["scores"]["incorrect_output"] is True for item in completed
        ),
        "latency_seconds_mean": round(
            statistics.mean(item["scores"]["latency_seconds"] for item in items), 3
        ),
        "dimensions_mean_non_truncated": {
            metric: round(
                statistics.mean(
                    item["scores"][metric]
                    for item in completed
                    if isinstance(item["scores"][metric], (int, float))
                ),
                3,
            )
            if any(isinstance(item["scores"][metric], (int, float)) for item in completed)
            else None
            for metric in metric_names
        },
        "categories": categories,
        "scoring_note": (
            "Deterministic lexical/syntax rubric, not a human or model judge; "
            "inspect false positives/negatives before promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--runtime-site", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    torch = bootstrap_torch(args.runtime_site)
    from transformers import AutoModelForImageTextToText, AutoProcessor, QuantoConfig

    processor = AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        quantization_config=QuantoConfig(weights="int4"),
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    cases = heldout_records()[: args.limit]
    items: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for index, task in enumerate(cases, 1):
            text = format_prompt(processor, task["prompt"])
            inputs = processor(text=[text], return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            started = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
            elapsed = time.perf_counter() - started
            prompt_length = int(inputs["input_ids"].shape[-1])
            generated_ids = generated[0][prompt_length:]
            answer = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            eos = processor.tokenizer.eos_token_id
            ended_with_eos = bool(len(generated_ids)) and int(generated_ids[-1]) == eos
            scores = score_answer(
                task, answer, len(generated_ids), args.max_new_tokens, ended_with_eos, elapsed
            )
            row = {
                "id": task["id"],
                "category": task["category"],
                "answer": answer,
                "scores": scores,
            }
            items.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if index % args.log_every == 0 or index == len(cases):
                truncated_count = sum(item["scores"]["truncated_output"] for item in items)
                print(
                    f"evaluated {index}/{len(cases)}; trunc={truncated_count}",
                    flush=True,
                )
    summary = summarize(items, "base" if not args.adapter else "adapter")
    summary["configuration"] = {
        "base_model": "Qwen/Qwen3.5-9B",
        "base_snapshot_revision": args.model_path.name,
        "quantization": "Quanto int4 weights",
        "compute_dtype": "bfloat16",
        "generation": "greedy, do_sample=false",
        "max_new_tokens": args.max_new_tokens,
        "adapter_loaded": args.adapter is not None,
        "prompt_contract": (
            "same CNDriveAI system message used for training plus held-out user prompt"
        ),
    }
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
