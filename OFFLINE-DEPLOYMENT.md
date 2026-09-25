# Offline CPU Deployment Design

## Target shape

Linux host with normal host networking; CNDriveTrust retains LAN and Central access. A separate CNDriveAI service runs as a dedicated unprivileged user and is denied Internet, DNS, and direct LAN access. It reads only an explicitly staged normalized context file or a tightly scoped Unix-domain socket. It cannot write into CNDriveTrust's result tree or Central queue.

## Inference artifact

Target model: Qwen3.5-9B plus an accepted CNDriveAI adapter (or a separately validated merged artifact). Production should not require ROCm, CUDA, PyTorch, Transformers, Ollama, or Hugging Face connectivity. The preferred CPU experiment is GGUF with `llama.cpp`, but conversion and exact multimodal/text architecture support must be verified against the pinned Qwen3.5 revision before selecting a production build. Do not download or convert on an enterprise server without a capacity plan.

## Reproducible conversion acceptance

1. Acquire the exact upstream snapshot through an approved offline transfer and verify revision/files/hashes.
2. Pin a `llama.cpp` release/commit and inspect its Qwen3.5 architecture support and converter behavior.
3. Convert a copied model tree in scratch storage; never mutate the upstream model cache.
4. Run a small quantization (e.g. Q4_K_M candidate) and verify model loads, context length, tokenizer/chat template, and representative text-only tasks.
5. Compare at least the same held-out suite against the approved source-format model: correctness proxies, truncation, latency, RAM, CPU threads, and concurrent service impact.
6. Verify adapter conversion/merge explicitly; if unsupported or quality regresses, do not ship the artifact.
7. Package exact runtime commit, conversion commands, hashes, license notices, and rollback artifact.

## Systemd/network controls

The production integrator must design and test a dedicated unit with no root, no writable product paths, read-only model/index directories, bounded writable staging/cache, private temporary storage, resource limits, and `NoNewPrivileges`. Block outbound network and DNS and deny direct LAN. Permit only a minimal local file drop or Unix IPC from an authorized producer. Do not apply a host-wide network deny: that would break CNDriveTrust Central. Keep AI service failure independent of product operation.

## Current verification boundary

This repository does not contain an enterprise host, GGUF model, adapter, service unit, firewall change, or target performance result. CPU-only deployment is a design plus acceptance procedure, not verified deployment.
