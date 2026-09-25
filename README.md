# CNDriveAI

**Offline, evidence-grounded engineering assistant foundations** for storage and hardware diagnostics, Linux troubleshooting, and safe Python/Bash help.

This repository contains synthetic-only datasets, context contracts, a local BM25 retrieval prototype, evaluation tooling, and a conservative QLoRA runner. It is not a deployed CNDriveTrust integration and does not execute generated commands.

## Current integration basis

The optional interface is aligned to the public CNDriveTrust `v2.3.0` source contract, commit [`13c006b`](https://github.com/SelimKandaz/CNDriveTrust/commit/13c006b8c030b14c9f4742dd7dfc3668d252794e). CNDriveTrust remains fully operational if CNDriveAI is absent. The normalizer supports the documented evidence schema 2.0 and health-summary schema 2.1; it is a standalone read-only adapter, not imported into the product.

The integration boundary is advisory: the product's own rules and disposition stay authoritative. CNDriveAI can explain, compare, suggest diagnostics, and prepare review-only patches in a staging area. It cannot select a destructive target, run erase/sanitize/format, alter namespaces, flash firmware, or deploy code.

## Quick start

```powershell
python -m pip install -e ".[dev]"
python scripts/generate_data.py
python -m pytest
python scripts/audit_publication.py
```

Training/evaluation require the optional GPU packages and a local Qwen3.5-9B snapshot. No weights, Hugging Face cache, adapter, ROCm runtime, or credentials belong in this repository. See [TRAINING.md](TRAINING.md) and [OFFLINE-DEPLOYMENT.md](OFFLINE-DEPLOYMENT.md).

## Repository status

- CNDriveTrust live integration: **not deployed or tested against production data**.
- Training and evaluation data: generated, synthetic-only.
- CNDriveAI QLoRA pilot: trained, saved, reloaded, and evaluated locally; **not promoted** because gains were small/mixed and safety/code behavior remains inadequate.
- Model results: aggregate metrics only; evaluator scores are deterministic rubric proxies, not human judgments. See [pilot results](PILOT-RESULTS.md).
- Runtime policy: offline by default; local files/Unix IPC only after host-side authorization.

## Documents

- [Architecture](ARCHITECTURE.md)
- [Training and reproducibility](TRAINING.md)
- [Pipeline audit](TRAINING-PIPELINE-AUDIT.md)
- [Dataset card](DATASET-CARD.md)
- [Model card](MODEL-CARD.md)
- [Evaluation](EVALUATION.md)
- [QLoRA pilot results](PILOT-RESULTS.md)
- [Safety](SAFETY.md)
- [Offline deployment](OFFLINE-DEPLOYMENT.md)
- [Work-computer integration handoff](WORK-INTEGRATION-HANDOFF.md)
- [Licenses](LICENSES.md)
