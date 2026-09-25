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

CNDriveAI is optional and must not become a runtime dependency of CNDriveTrust. The v2.3.0 interface says the product can provide normalized JSON/manifests, explicitly requested raw evidence, local documentation/source/runbooks, and preserved reports. The adapter in `src/cndriveai/cndrivetrust.py` consumes only normalized evidence/summary objects; it does not import CNDriveTrust code or read its result tree.

## Release-aligned data

The public v2.3.0 contract records normalized fields as `{value, status, source, time, unit?}`. Numeric zero and a missing/failed query are different states and remain distinct. The adapter recognizes evidence artifacts with schema `2.0` and Health & Usage Summary artifacts with schema `2.1`; the latter has separate `media_health`, `endurance`, `error_state`, `usage`, `performance`, and `history_trust` dimensions.

Health values can be current while performance may be sourced from the last preserved full-read result. Preserve each source and time; never relabel historical performance as live. Current product disposition and policy are carried as facts, never recalculated or replaced by model output.

## Context precedence

1. Current run/evidence from the host.
2. Current product rules/configuration explicitly supplied by the host.
3. Local project documentation.
4. Historical runs explicitly supplied for comparison.
5. Vendor/reference material.
6. General pretrained knowledge.

Conflicts are retained with provenance. The model must describe the conflict, not silently resolve it from prior knowledge.

## Retrieval

SQLite FTS5/BM25 provides a small CPU-only, offline index, filtered by namespace. Structured JSON lookup is direct and typed. No vector database or embedding service is required. A future embedding provider may be optional, but it cannot cross project namespaces or override current evidence.

## Release status

The reviewed CNDriveTrust public tag is `v2.3.0`; its AI menu entry is a placeholder and installs no model/runtime. The adapter here is a contract bridge prototype only. No production data, target host, or Central connection has been used.
