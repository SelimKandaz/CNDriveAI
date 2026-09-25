"""Token-boundary and assistant-only supervision invariants."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def apply_template(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> list[int]:
    options = {"tokenize": True, "add_generation_prompt": generation}
    try:
        result = tokenizer.apply_chat_template(messages, enable_thinking=False, **options)
    except TypeError:
        result = tokenizer.apply_chat_template(messages, **options)
    return token_ids(result)


def encode_chat(record: dict[str, Any], tokenizer: Any, max_seq_len: int) -> dict[str, list[int]]:
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be positive")
    messages = record.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or messages[-1].get("role") != "assistant"
    ):
        raise ValueError("chat record must end with an assistant message")
    full = apply_template(tokenizer, messages, generation=False)
    prompt = apply_template(tokenizer, messages[:-1], generation=True)
    if full[: len(prompt)] != prompt:
        raise ValueError(f"chat-template prefix mismatch in {record.get('id', '<unknown>')}")
    if len(full) <= len(prompt):
        raise ValueError(f"assistant response is empty in {record.get('id', '<unknown>')}")
    kept = full[:max_seq_len]
    prompt_len = min(len(prompt), len(kept))
    labels = [-100] * prompt_len + kept[prompt_len:]
    if not any(label != -100 for label in labels):
        raise ValueError(
            f"assistant response entirely truncated in {record.get('id', '<unknown>')}"
        )
    return {"input_ids": kept, "labels": labels}
