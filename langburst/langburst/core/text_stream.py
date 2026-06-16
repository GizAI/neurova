from __future__ import annotations

from typing import Any


class StreamingTextDecoder:
    """Incremental tokenizer decoder for SSE/CLI streaming.

    Byte-fallback tokenizers can emit U+FFFD replacement characters when one
    token is decoded in isolation. Streaming should emit deltas from the decoded
    token prefix instead.
    """

    def __init__(self, tokenizer: Any, *, skip_special_tokens: bool = False) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_ids: list[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        self.token_ids.append(int(token_id))
        return self._delta(final=False)

    def flush(self) -> str:
        return self._delta(final=True)

    def _decode(self) -> str:
        try:
            return str(
                self.tokenizer.decode(
                    self.token_ids,
                    skip_special_tokens=self.skip_special_tokens,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(self.tokenizer.decode(self.token_ids, skip_special_tokens=self.skip_special_tokens))

    def _delta(self, *, final: bool) -> str:
        decoded = self._decode()
        visible = decoded if final else decoded.rstrip("\ufffd")
        visible = visible.replace("\ufffd", "")
        if len(visible) <= len(self.emitted):
            return ""
        if not visible.startswith(self.emitted):
            prefix_len = 0
            for a, b in zip(visible, self.emitted):
                if a != b:
                    break
                prefix_len += 1
            self.emitted = self.emitted[:prefix_len]
        delta = visible[len(self.emitted) :]
        self.emitted = visible
        return delta
