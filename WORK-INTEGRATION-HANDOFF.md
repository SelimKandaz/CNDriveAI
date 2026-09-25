# Work-Computer Integration Handoff

## Known from the source inspected

The public `SelimKandaz/CNDriveTrust` `main` at commit `13c006b8c030b14c9f4742dd7dfc3668d252794e` is tagged `v2.3.0`. Its public AI contract keeps CNDriveAI optional; product logic runs without a model/runtime. Permitted inputs include normalized JSON/manifests, raw evidence only when explicitly requested, local project docs/source/runbooks/known issues, and preserved reports. Permitted outputs are explanations, diagnostic suggestions, comparisons, code suggestions, and patch files in a future staging location.

The public normalized evidence envelope is `{value, status, source, time, unit?}`. Evidence result schema is 2.0; health summary schema is 2.1 and returns six dimensions (`media_health`, `endurance`, `error_state`, `usage`, `performance`, `history_trust`) plus `overall` and assessment limits. The run's final disposition is produced by CNDriveTrust. Health can be current while performance is explicitly taken from a preserved last-test result. Central sync is a separate offline-first concern: a valid local result remains valid while delivery is pending.

The source inspected is the repository snapshot, not a live work-computer installation. Confirm that the deployed release/configuration matches before adapting; do not assume the deployed tree is identical to `main`.

## Integration sequence for the work computer

1. Record the installed release/tag/commit and compare deployed files against the approved release. Inspect the AI placeholder contract and menu integration, but keep the product usable without CNDriveAI.
2. Confirm exact normalized evidence/health-summary schemas and freshness behavior from the deployed code. Add explicit schema versions and fixture tests to the integration adapter.
3. Export one synthetic or approved sanitized artifact to a staging location. Test adapter behavior for numeric zero, `UNAVAILABLE`, `QUERY_FAILED`, `UNSUPPORTED`, and historical performance. Never use a real customer example in public tests.
4. Keep raw `values` provenance (`source`, `time`, `status`, `unit`) and six summary dimensions. Never convert missing/failed to zero or recalculate `overall`/`analysis.disposition` in the model layer.
5. Scope retrieval to approved local documentation/runbooks and namespace per deployment. Do not index serial/PO inventories by default. Preserve source/hash/access metadata.
6. Install CPU inference as a separate optional service. It receives context only by approved local staging or Unix IPC and has no Central credentials, product write access, or command executor.
7. Preserve host and CNDriveTrust network access. Deny Internet, DNS, and direct LAN only for the dedicated CNDriveAI service. Verify Central sync while the AI boundary is active.
8. Test service absence, timeout, malformed context, schema drift, and model failure. In each case CNDriveTrust must still finish its normal evidence/report/Central workflow.
9. Require human review for all explanations that affect disposition, all diagnostic commands, and all code/patch suggestions. Product gates remain the only authority for target protection and operation safety.
10. Benchmark CPU inference and any optional adapter training separately. Never train on customer or company evidence without separate data governance approval.

## Artifacts to transfer

- This sanitized CNDriveAI source repository and synthetic train/held-out data.
- A base model acquired under the approved model license through a separate verified transfer; do not transfer a Hugging Face cache as a substitute for a manifest.
- Do not transfer the current CNDriveAI adapter pilot: it did not pass promotion gates. Transfer a future version only after it passes an expanded held-out suite, destructive-action refusal review, and Python/Bash regression review. Adapter weights remain intentionally outside GitHub.
- `OFFLINE-DEPLOYMENT.md`, model revision/hash manifest, and runtime/conversion instructions.
- No real run bundles, serial/PO lists, customer reports, credentials, tokens, host paths, Central configuration, or raw command evidence.

## Must be learned on the work computer

The local product/source tree, actual deployed release, approved local documentation, policy/config generation, Central paths/credentials, Linux service/user policy, firewall tooling, CPU hardware and inference SLOs, data retention requirements, and authorization for any specific sanitized training/evaluation corpus. None of these are guessed here.

## Stop conditions

Do not enable runtime integration if the deployed schema differs, the staging boundary cannot be made read-only, service network isolation interferes with Central, or product authority can be changed by model output. In those cases keep the placeholder and report the exact mismatch.
