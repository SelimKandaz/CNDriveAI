# Architecture

## Boundary

```text
CNDriveTrust (independent, authoritative, host LAN/Central enabled)
  finalized normalized JSON + manifest
             |
             v
approved local staging / privacy filter (read-only input)
             |
             v
CNDriveAI (dedicated Linux user; Internet, DNS, direct LAN denied)
  context adapter -> SQLite FTS5/BM25 -> Qwen3.5 CPU inference
             |
             v
advice / comparisons / diagnostics / patch proposal in staging
             |
             v
human review; product rules remain authoritative
```

CNDriveAI is optional and never becomes a runtime dependency of CNDriveTrust. The adapter in `src/cndriveai/cndrivetrust.py` reads only normalized evidence and summary objects. It does not import CNDriveTrust code or read its result tree.

## Data format

Normalized fields follow `{value, status, source, time, unit?}`. A numeric zero and a missing or failed query are different states and stay different. The adapter reads evidence artifacts (schema `2.0`) and Health & Usage Summary artifacts (schema `2.1`), which carry separate `media_health`, `endurance`, `error_state`, `usage`, `performance`, and `history_trust` dimensions.

Health values can be current while performance comes from the last preserved full-read result. Each source and time is kept, and historical performance is never relabeled as live. The product's disposition and policy are passed through as facts, never recalculated by the model.

## Context precedence

1. Current run and evidence from the host
2. Current product rules and configuration supplied by the host
3. Local project documentation
4. Historical runs supplied for comparison
5. Vendor and reference material
6. General pretrained knowledge

Conflicts are kept with their provenance. The model describes the conflict instead of quietly resolving it.

## Retrieval

SQLite FTS5/BM25 gives a small, CPU-only, offline index filtered by namespace. Structured JSON lookup is direct and typed. No vector database or embedding service is needed.
