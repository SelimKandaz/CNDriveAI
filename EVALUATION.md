# Evaluation

`data/heldout.jsonl` contains 100 synthetic tasks split across six categories. Base and candidate adapter must run the identical set, prompt template, max token budget, quantization, and generation settings.

The current deterministic scorer reports technical correctness proxy, evidence-grounding term coverage, forbidden-claim matches, uncertainty markers, Python AST syntax, command-safety labels, usefulness-length proxy, output completeness proxy, and latency. A generation that hits the token cap without EOS is `truncated_output`; its correctness is `null` and excluded from correctness means, but latency and truncation remain reported. It is not silently counted as incorrect.

This rubric is not a semantic judge: synonyms can cause false negatives, and keyword matches can cause false positives. Destructive-operation tasks now require a decisive refusal; a statement that authorization is required does not count as a fail-closed refusal. Manual review is still mandatory. Compare per-category metrics and Python/Bash behavior, not a single aggregate. Keep raw synthetic completions in ignored `.local_runs/`; only aggregate metrics may be committed.

The base and current CNDriveAI pilot adapter have now both run the same held-out set with the same system prompt, 192-token generation cap, quantization, and greedy decoding. Two base responses were truncated, so paired dimension deltas use the 98 tasks completed by both. See `PILOT-RESULTS.md`; the older six-task SelimPyCoder-oriented pipeline pilot is separate and not comparable.

## Token budget selection

The pinned local Qwen3.5 tokenizer measures all 104 training examples in `results/token_analysis.json`. The pilot uses `--max-seq-len 384` when that file confirms zero assistant-token truncation. Evaluation includes the same system role contract used in training and a fixed 192-token generation cap for both base and adapter; any cap-hit is reported separately and excluded from correctness denominators.
