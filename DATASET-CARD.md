# Dataset Card

## Purpose

Teach an offline engineering assistant to separate measurement from inference, keep provenance and timestamps, handle missing, failed, and unsupported states, express uncertainty, and suggest bounded read-only diagnostics.

## Sources and privacy

Every example is authored, synthetic text. No product implementation, customer data, serial or purchase-order data, personal logs, model outputs, or raw device captures are used.

## Split

104 training examples, built from 26 authored cases with four phrasings each. The held-out suite has 100 tasks:

| Category | Count |
|---|---:|
| Evidence, health, provenance | 25 |
| Program/context interpretation | 20 |
| Linux/storage troubleshooting | 20 |
| Python | 15 |
| Bash/manual diagnostics | 10 |
| Safety/uncertainty | 10 |

Held-out prompts never appear in training. The suite works as a regression screen alongside manual review.

## Extending it

Add only synthetic or explicitly sanitized cases, and keep a fresh held-out split.
