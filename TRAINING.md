# Training

## Data

`scripts/generate_data.py` builds `data/train.jsonl` and `data/heldout.jsonl` from authored synthetic cases. No product data, customer identifiers, or personal datasets are used. The held-out set is generated separately and never used for training.

## Running it

Use the pinned Qwen3.5-9B local snapshot and an isolated ROCm Python environment. Training was run on native Windows ROCm with an RX 9070 XT.

Replace the paths for your host and keep outputs outside Git:

```powershell
python scripts/generate_data.py
python scripts/analyze_tokens.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --data data/train.jsonl --out .local_runs/token-analysis.json
python scripts/evaluate_model.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --output .local_runs/base.jsonl --metrics-out results/base_metrics.json
python scripts/train_qwen35_qlora.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --train data/train.jsonl --out <PRIVATE_ADAPTER_PATH> --max-steps 20 --grad-accum 4 --max-seq-len <MEASURED_LENGTH>
python scripts/evaluate_model.py --model-path <LOCAL_QWEN_SNAPSHOT> --runtime-site <ISOLATED_SITE_PACKAGES> --adapter <PRIVATE_ADAPTER_PATH> --output .local_runs/adapter.jsonl --metrics-out results/adapter_metrics.json
```

Run the same 100 held-out tasks before and after training and compare by category. Loss alone is never the deciding metric.

## Trainer guardrails

- Only assistant response tokens contribute to loss. Prompt labels are `-100`.
- The prompt tokens must be an exact prefix of the full chat sequence, or the record is rejected.
- Truncation is measured first. An example that would lose its whole answer is rejected.
- Quantized base weights are frozen, and every trainable parameter must be a `lora_` parameter.
- Only the adapter is saved. Tokenizer and base weights are not copied.
- An existing output path is rejected.
