# Model Card

## Base model

- Upstream: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`
- Intended use: offline, evidence-grounded engineering assistance with current context and reviewed retrieval
- License: Apache-2.0 per the upstream model card. Get the weights from the official source and keep its notices.
- Base weights and caches are not part of this repository.

## Adapter

A QLoRA adapter is trained on the synthetic dataset in this repo. Adapter weights are kept outside Git. The base model is the default.

## Use

Deterministic product rules stay authoritative. Outputs are reviewed by a human, and no tool execution is connected.
