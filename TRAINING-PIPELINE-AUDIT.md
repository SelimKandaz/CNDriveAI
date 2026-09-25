# QLoRA Pipeline Audit

## Historical SelimPyCoder-oriented pilot (pipeline validation only)

| Item | Recorded evidence |
|---|---|
| Base | `Qwen/Qwen3.5-9B`, local snapshot `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| GPU/runtime | AMD Radeon RX 9070 XT; native Windows ROCm 7.2.1; Torch `2.9.1+rocm7.2.1` |
| Frameworks | Transformers 5.13.0; PEFT 0.19.1; Optimum Quanto 0.2.7 |
| Quantization | Quanto int4 weights, BF16 compute; local base load and gradient steps completed |
| LoRA modules | present projection suffixes: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Training | 20 optimizer steps; bounded single-GPU pilot. This run was not the CNDriveAI dataset/adapter. |
| Memory | device reported 15.922 GiB total; peak reserved 10.486 GiB in the recorded run |
| Adapter | approximately 58 MB; SHA-256 `91ba9cd514208cedc60dbe4100aa7cafa404e8224039f0f5aed16c645a65f057` |
| Save/reload | adapter saved and reloaded; smoke generation completed |
| Evaluation | small six-task code probe: base 2/6, pilot adapter 1/6; not a final domain evaluation and not promoted |

These observations validate that a bounded QLoRA path ran on this workstation. They do not show that the old adapter is useful for storage reasoning, and do not justify transferring it to CNDriveTrust.

## CNDriveAI synthetic-data pilot

| Item | Recorded evidence |
|---|---|
| Dataset | 104 generated synthetic training rows; no company, customer, or private data |
| Base/runtime | Same immutable Qwen3.5-9B snapshot; isolated native Windows ROCm runtime; Quanto int4 weights, BF16 compute |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 14,548,992; every trainable parameter name included `lora_`; base frozen |
| Run | 20 steps, grad accumulation 4, sequence length 384, LR 1e-5, seed 935; 102.057 seconds |
| Memory/loss | 10.303 GiB peak reserved of 15.922 GiB; loss 3.238710 to 3.016845 |
| Artifact | 58,233,376-byte adapter weights; SHA-256 `47180f0dd14b2b52f5ef4ae66a906b63c2d1c09ce49f79dfe1b2dabc11db9e61`; stored outside Git |
| Save/reload | PEFT adapter-only output saved and loaded for held-out evaluation; generation succeeded |
| Decision | **Not promoted**: paired results are mixed; safety response quality and Python code validity remain inadequate |

This establishes that the generic CNDriveAI training data and runner completed a small experiment, not that the adapter is ready for use. It is a first low-step pilot over a compact template-based dataset and is not transferred to CNDriveTrust. Exact paired evaluation and manual findings are in `PILOT-RESULTS.md`.

## Static path review

The original pilot freezes model parameters before PEFT injection, builds an optimizer from `requires_grad` parameters, masks prompt labels with `-100`, pads attention with zero and padded labels with `-100`, and saves PEFT output separately from the base snapshot. Its encoder rendered prompt and full text separately and then assumed the prompt tokenization was a prefix. That boundary assumption was not asserted. The CNDriveAI runner uses the tokenizer's chat-template token API and rejects any non-prefix case rather than silently training on prompt tokens or masking assistant content.

The prior runner saved the tokenizer beside adapter weights. The new public runner saves adapter weights/config plus run metadata only; it does not duplicate tokenizer or base files. The base is addressed by immutable local snapshot path and never used as output.

## New pipeline tests

The test suite checks synthetic-data separation, explicit zero preservation, privacy stripping in the v2.3 adapter, context precedence, retrieval namespaces, and truncation being reported separately from incorrect output. GPU-specific save/reload and trainable-name checks are performed by the real pilot when run; unit tests do not claim to simulate those steps.
