"""Conservative single-GPU QLoRA runner for the isolated CNDriveAI lineage."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_rocm import bootstrap_torch  # noqa: E402

from cndriveai.training import encode_chat  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _memory(torch: Any) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "free_gib": round(free / 1024**3, 3),
        "total_gib": round(total / 1024**3, 3),
        "allocated_gib": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-site", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=935)
    args = parser.parse_args()
    if not args.model_path.is_dir() or not args.train.is_file():
        raise SystemExit("base model and training file must exist locally")
    if args.out.exists() or args.model_path.resolve() in args.out.resolve().parents:
        raise SystemExit("refusing existing output or output inside immutable base snapshot")
    torch = bootstrap_torch(args.runtime_site)

    if not torch.cuda.is_available():
        raise SystemExit(
            "This pilot requires the already validated local ROCm GPU; CPU training refused"
        )
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor, QuantoConfig

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    processor = AutoProcessor.from_pretrained(
        str(args.model_path), local_files_only=True, trust_remote_code=False
    )
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        quantization_config=QuantoConfig(weights="int4"),
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    names = [name for name, _ in model.named_modules()]
    targets = [suffix for suffix in TARGETS if any(name.endswith(suffix) for name in names)]
    if targets != list(TARGETS):
        raise SystemExit(f"expected validated Qwen projection suffixes; found {targets}")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=targets,
        ),
    )
    trainable_named = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable_named or any("lora_" not in name for name, _ in trainable_named):
        raise SystemExit("trainable parameter audit failed; only LoRA parameters may be trained")
    if any(
        "lora_" in name and not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if "lora_" in name
    ):
        raise SystemExit("one or more LoRA parameters are unexpectedly frozen")

    records = _read_jsonl(args.train)
    encoded = [encode_chat(record, tokenizer, args.max_seq_len) for record in records]
    model_device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_named], lr=args.learning_rate
    )
    metadata: dict[str, Any] = {
        "status": "started",
        "base_model": "Qwen/Qwen3.5-9B",
        "base_snapshot_revision": args.model_path.name,
        "quantization": "Quanto int4 weights, bfloat16 compute",
        "lora_target_modules": targets,
        "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable_named),
        "trainable_name_audit": "all trainable names contain lora_",
        "record_count": len(records),
        "max_seq_len": args.max_seq_len,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.grad_accum,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "initial_gpu_memory": _memory(torch),
    }
    rng = random.Random(args.seed)
    losses: list[float] = []
    step = micro = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    model.train()
    while step < args.max_steps:
        order = list(range(len(encoded)))
        rng.shuffle(order)
        for index in order:
            item = encoded[index]
            ids = item["input_ids"]
            labels = item["labels"]
            attention = [1] * len(ids)
            batch = {
                "input_ids": torch.tensor([ids], dtype=torch.long, device=model_device),
                "attention_mask": torch.tensor([attention], dtype=torch.long, device=model_device),
                "labels": torch.tensor([labels], dtype=torch.long, device=model_device),
            }
            loss = model(**batch).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at micro-step {micro}")
            (loss / args.grad_accum).backward()
            losses.append(float(loss.detach().item()))
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable_named], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == args.max_steps:
                    print(f"step {step}/{args.max_steps} loss={losses[-1]:.4f}", flush=True)
                if step >= args.max_steps:
                    break
    metadata.update(
        {
            "status": "trained",
            "steps_completed": step,
            "training_seconds": round(time.perf_counter() - started, 3),
            "first_loss": round(losses[0], 6),
            "last_loss": round(losses[-1], 6),
            "peak_gpu_reserved_gib": round(torch.cuda.max_memory_reserved(0) / 1024**3, 3),
            "final_gpu_memory": _memory(torch),
        }
    )
    args.out.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.out, safe_serialization=True)
    metadata["saved_files"] = sorted(path.name for path in args.out.iterdir() if path.is_file())
    if not (args.out / "adapter_model.safetensors").is_file():
        raise RuntimeError("adapter-only safetensors save verification failed")
    if any(path.suffix in {".bin", ".pt", ".pth"} for path in args.out.iterdir()):
        raise RuntimeError("unexpected full model weight file in adapter output")
    metadata["status"] = "adapter_saved"
    (args.out / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
