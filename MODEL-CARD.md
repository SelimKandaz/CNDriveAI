# Model Card

## Base model

- Upstream: `Qwen/Qwen3.5-9B`.
- Local development revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Intended use: offline evidence-grounded engineering assistance, with authoritative current context and reviewed retrieval.
- License: upstream model card currently declares Apache-2.0; acquire weights from the official model source and retain its notices.
- Base weights and Hugging Face cache are not part of this repository.

## Adapter status

The separate CNDriveAI synthetic-data pilot was trained and saved outside Git at a private local path. It is a research artifact, **not promoted** and not approved for work-computer transfer or production. On 98 paired, non-truncated held-out tasks, rubric deltas were mixed: technical correctness -0.003, evidence grounding -0.010, hallucination control +0.020, uncertainty discipline -0.010, and command safety -0.010. The adapter produced no truncated answers versus two for the base, but a manual review found a destructive-command answer that was not sufficiently fail-closed. Python AST validity improved from 2/13 to 4/13 paired tasks, still too low for practical code assistance. See [PILOT-RESULTS.md](PILOT-RESULTS.md).

Private artifact metadata: 20 optimizer steps; 14,548,992 trainable LoRA parameters; adapter weights 58,233,376 bytes; SHA-256 `47180f0dd14b2b52f5ef4ae66a906b63c2d1c09ce49f79dfE1B2DABC11DB9E61`. No weights are included in this repository. The earlier SelimPyCoder adapter remains a separate lineage and was not modified, merged, or reused.

## Limitations

The base and pilot adapter can hallucinate, misread unsupported/unknown states, omit requested code, or weaken a safety refusal. Deterministic product rules remain authoritative. Outputs require human review; no tool execution is connected. CPU inference and GGUF conversion are deployment goals to validate on the target Linux server, not a completed deployment claim.
