# CNDriveAI

An offline engineering assistant for storage and hardware diagnostics, Linux troubleshooting, and Python/Bash help. It runs locally and keeps its answers grounded in collected evidence.

## What it does

- Explains drive and hardware evidence and compares results
- Suggests next diagnostic steps
- Prepares review-only patches for a human to approve

It is advisory only. It never picks a destructive target, erases, formats, flashes firmware, or deploys code.

## What's inside

- Synthetic training and evaluation datasets
- Local BM25 retrieval
- Evaluation tooling
- QLoRA training runner

## Quick start

```powershell
python -m pip install -e ".[dev]"
python scripts/generate_data.py
python -m pytest
```

Training needs the optional GPU packages and a local base model. No model weights or credentials are stored in this repo. See [TRAINING.md](TRAINING.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## License

Apache 2.0
