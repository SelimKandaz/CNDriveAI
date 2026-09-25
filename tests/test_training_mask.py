import pytest

from cndriveai.training import encode_chat


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
        if add_generation_prompt:
            return [11, 12, 13]
        return [11, 12, 13, 21, 99]


def test_only_assistant_tokens_are_supervised_and_eos_is_kept():
    record = {
        "id": "synthetic",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }
    encoded = encode_chat(record, FakeTokenizer(), 10)
    assert encoded["input_ids"] == [11, 12, 13, 21, 99]
    assert encoded["labels"] == [-100, -100, -100, 21, 99]


def test_prompt_prefix_mismatch_fails_closed():
    class MismatchTokenizer(FakeTokenizer):
        def apply_chat_template(
            self, messages, tokenize, add_generation_prompt, enable_thinking=False
        ):
            return [11, 42] if add_generation_prompt else [11, 12, 21]

    record = {
        "id": "mismatch",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }
    with pytest.raises(ValueError, match="prefix mismatch"):
        encode_chat(record, MismatchTokenizer(), 10)


def test_fully_truncated_assistant_answer_is_rejected():
    record = {
        "id": "long",
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
    }
    with pytest.raises(ValueError, match="entirely truncated"):
        encode_chat(record, FakeTokenizer(), 3)
