# Training

## Data boundaries

`scripts/generate_data.py` creates `data/train.jsonl` and `data/heldout.jsonl` from authored synthetic cases. It does not ingest CNDriveTrust data, customer identifiers, personal datasets, or SelimPyCoder training files. The held-out set is generated independently and is not used for optimization.

## Reproducible GPU pilot

Use the pinned Qwen3.5-9B local snapshot and an isolated ROCm Python environment. The tested local machine was Windows with an RX 9070 XT; WSL did not expose `/dev/kfd` or `/dev/dri`, so the validated training runtime was native Windows ROCm. Avoid touching the ComfyUI Python environment.

Example commands (replace paths for the host; keep outputs outside Git):

```powershell
python scripts/generate_data.py
python scripts/analyze_tokens.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --data data/train.jsonl --out .local_runs/token-analysis.json
python scripts/evaluate_model.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --output .local_runs/base.jsonl --metrics-out results/base_metrics.json
python scripts/train_qwen35_qlora.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --train data/train.jsonl --out <PRIVATE_ADAPTER_PATH> --max-steps 20 --grad-accum 4 --max-seq-len <MEASURED_LENGTH>
python scripts/evaluate_model.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --adapter <PRIVATE_ADAPTER_PATH> --output .local_runs/adapter.jsonl --metrics-out results/adapter_metrics.json
```

Run the exact same 100 held-out tasks before and after training. `results/*_metrics.json` contains aggregates only; per-task answers are local ignored files. Report base, adapter, and delta by dimension and category, plus truncation and latency separately. Do not promote on loss alone.

## Trainer guardrails

- Qwen chat template is tokenized both with and without the assistant response.
- The assistant prompt token sequence must be an exact prefix of the full chat sequence; otherwise the record fails closed.
- All prompt labels are `-100`; only assistant response tokens contribute loss.
- Right truncation is measured first; an example with its entire answer removed is rejected.
- The base snapshot is opened read-only by convention and output must be separate.
- Quantized base parameters are frozen; all trainable names must contain `lora_`.
- Only the PEFT adapter is saved; tokenizer and base weights are not copied.
- An existing output path is rejected.

## Completed CNDriveAI synthetic pilot

The separate CNDriveAI adapter was trained on 104 authored synthetic examples using the pinned Qwen3.5-9B snapshot. Training used Quanto int4 weights, BF16 compute, LoRA on q/k/v/o and gate/up/down projections, 20 optimizer steps, gradient accumulation 4, sequence length 384, learning rate 1e-5, and seed 935. All 14,548,992 trainable parameters had `lora_` names; base weights remained frozen. Training took 102.057 seconds and peak reserved VRAM was 10.303 GiB of 15.922 GiB. Loss changed from 3.238710 to 3.016845; this is not evidence of useful behavior by itself.

The tokenizer measured full-sequence p50/p75/p90/p95/max at 220/239/278/286/322 tokens and assistant-only response lengths at 57/71/114/118/153. A 256-token cap would remove 552 assistant tokens across 20 examples (8.0% of answer tokens); 384 removed none, so 384 was selected. The adapter reloaded and generated successfully. It remains outside Git at the owner's local adapter store. Its 58,233,376-byte weight file has SHA-256 `47180f0dd14b2b52f5ef4ae66a906b63c2d1c09ce49f79dfe1b2dabc11db9e61`.

The 100-task aligned base/adapter comparison is mixed and **does not pass promotion gates**. See `PILOT-RESULTS.md` and `results/base_adapter_comparison.json`. The prior SelimPyCoder-oriented adapter experiment is separate; it was not used, copied, merged, or modified.
