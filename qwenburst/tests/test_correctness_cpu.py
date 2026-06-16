from __future__ import annotations

from qwenburst.correctness import _encode_answer, recall_prompt


class ToyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        assert not add_special_tokens
        return [ord(ch) for ch in text]


class ToyEngine:
    tokenizer = ToyTokenizer()


def test_recall_prompt_contains_one_secret_and_final_question():
    prompt = recall_prompt("NX-1742-ALPHA", filler_repeats=2)
    assert prompt.count("NX-1742-ALPHA") == 1
    assert "Final question" in prompt
    assert "Answer only the code" in prompt


def test_encode_answer_rejects_empty_candidate():
    assert _encode_answer(ToyEngine(), "AB") == [65, 66]
