# Safety Policy

- Evidence, source, status, units, and time must remain attached to every fact.
- Zero is a reported value; missing, failed, unsupported, blocked, and not-evaluated are separate states.
- A plausible explanation is not a proven cause. Do not infer intent, seller action, fraud, or tampering without evidence.
- Product-generated dispositions and deterministic safety checks are authoritative.
- Model output cannot select targets, execute commands, write production files, change namespaces, flash firmware, erase, sanitize, format, or deploy.
- Diagnostics must label risk. Read-only commands are preferred; state-changing and destructive operations must never be presented as routine diagnostics.
- Proposed code changes are files in a separate staging area and require human review and normal tests.
- Historical evidence stays labeled historical. Current Health & Usage fields and last-preserved performance measurements may have different freshness.
- The CNDriveAI service is denied Internet, DNS, and direct LAN access by default. The host and independent CNDriveTrust/Central connectivity remain enabled.
- Only an approved local staging-file or Unix IPC boundary may pass normalized context to the isolated service.
- Raw evidence is excluded by default and may be admitted only for an explicit, audited review request.

This is a software policy, not a hardened runtime sandbox. Production deployment still needs a dedicated Linux user, systemd sandboxing, filesystem allowlists, resource limits, firewall policy, and integration tests.
