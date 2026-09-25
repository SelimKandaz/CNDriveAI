# Dataset Card

## Purpose

Teach an offline engineering assistant to separate measurement from inference, preserve provenance and timestamps, handle missing/query-failed/unsupported states, express uncertainty, and provide bounded read-only diagnostics. Basic Python/Bash syntax is not the training objective.

## Sources and privacy

All examples are authored, deterministic, synthetic text. No company implementation, customer data, serial-number inventory, purchase-order data, personal logs, model outputs, or raw device captures are used. Examples use synthetic names/values. There are no third-party datasets or real identifiers.

## Split and composition

The generator currently creates 104 training examples from 26 authored synthetic cases, each with four surface phrasings. The independent held-out suite contains 100 tasks:

| Category | Count |
|---|---:|
| Evidence, health, provenance | 25 |
| Program/context interpretation | 20 |
| Linux/storage troubleshooting | 20 |
| Python | 15 |
| Bash/manual diagnostics | 10 |
| Safety/uncertainty | 10 |

The held-out task IDs and prompts are disjoint from training prompts. The suite is compact and synthetic; it is a regression screen, not a broad industry benchmark or substitute for expert review. Each category reuses a small set of authored templates with rotated framing, so the 100 rows are not 100 fully independent scenarios. Some evaluator rubrics are lexical and can mis-score semantically correct paraphrases.

## Known limitations

The dataset is intentionally small. It does not encode CNDriveTrust production policy beyond the public interface concepts, does not teach vendor-specific decoder behavior, and does not authorize execution. Expand only with approved synthetic or explicitly sanitized cases and keep a new held-out split.
