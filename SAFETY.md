# Safety Policy

- Every fact keeps its evidence, source, status, unit, and time.
- Zero is a reported value. Missing, failed, unsupported, blocked, and not-evaluated are separate states.
- A plausible explanation is not a proven cause. No claims of fraud or tampering without evidence.
- Product dispositions and deterministic safety checks are authoritative.
- Model output cannot select targets, run commands, write production files, change namespaces, flash firmware, erase, sanitize, format, or deploy.
- Diagnostics are labeled by risk. Read-only commands come first, and destructive operations are never presented as routine.
- Code changes are proposed as files in a separate staging area and go through human review and normal tests.
- Historical evidence stays labeled as historical.
- The CNDriveAI service has no Internet, DNS, or direct LAN access by default. The host and CNDriveTrust Central connectivity are unaffected.
- Context reaches the service only through an approved local staging file or Unix IPC.
- Raw evidence is excluded by default and only admitted for an explicit, audited review.

This policy is enforced in software. A production deployment adds a dedicated Linux user, systemd sandboxing, filesystem allowlists, resource limits, and firewall rules on top of it.
