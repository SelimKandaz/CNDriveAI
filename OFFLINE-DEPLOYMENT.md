# Offline CPU Deployment Design

## Target shape

A Linux host with normal networking, where CNDriveTrust keeps its LAN and Central access. CNDriveAI runs as a separate service under a dedicated unprivileged user with no Internet, DNS, or direct LAN access. It reads only a staged context file or a tightly scoped Unix socket, and it cannot write into CNDriveTrust's results or Central queue.

## Inference

Qwen3.5-9B, optionally with a validated adapter or merged model. Production does not need ROCm, CUDA, PyTorch, Transformers, Ollama, or Hugging Face connectivity. The preferred CPU path is GGUF with `llama.cpp`, checked against the pinned Qwen3.5 revision first.

## Conversion checklist

1. Transfer the exact upstream snapshot offline and verify revision, files, and hashes.
2. Pin a `llama.cpp` release and confirm its Qwen3.5 support.
3. Convert a copy of the model in scratch storage. Never modify the original.
4. Quantize (for example Q4_K_M) and check loading, context length, tokenizer, chat template, and sample tasks.
5. Run the held-out suite against the source-format model and compare correctness, truncation, latency, RAM, and CPU use.
6. Verify any adapter conversion or merge explicitly. Ship only if quality holds.
7. Package the runtime commit, conversion commands, hashes, license notices, and a rollback artifact.

## Systemd and network

Run as a dedicated unit with no root, no writable product paths, read-only model and index directories, bounded staging and cache, private temp storage, resource limits, and `NoNewPrivileges`. Block outbound network and DNS for this service only. A host-wide block would cut CNDriveTrust off from Central. If the AI service fails, the product keeps working.

Model files, service units, and host configuration live outside this repository.
