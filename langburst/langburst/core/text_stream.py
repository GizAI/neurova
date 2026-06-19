from __future__ import annotations

import re
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


class ThinkingTextFilter:
    """Remove Qwen visible-thinking spans without changing model logits.

    Qwen chat templates may place or emit ``<think>`` / ``</think>`` boundary
    tokens even when the caller asked for a non-thinking answer. Suppressing
    those token IDs at sampling time changes the model distribution and can
    make long prompts collapse into instruction-copying. This filter keeps the
    token stream intact and only removes the reasoning span from API text.
    """

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._buffer = ""
        self._hiding = False
        self._keep = max(len(self.OPEN), len(self.CLOSE)) - 1

    def push(self, text: str, *, final: bool = False) -> str:
        if not self.enabled or not text and not final:
            return text
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            if self._hiding:
                close_idx = self._buffer.find(self.CLOSE)
                if close_idx < 0:
                    if final:
                        self._buffer = ""
                    elif len(self._buffer) > self._keep:
                        self._buffer = self._buffer[-self._keep :]
                    break
                self._buffer = self._buffer[close_idx + len(self.CLOSE) :].lstrip()
                self._hiding = False
                continue

            open_idx = self._buffer.find(self.OPEN)
            close_idx = self._buffer.find(self.CLOSE)
            candidates = [idx for idx in (open_idx, close_idx) if idx >= 0]
            if not candidates:
                if final:
                    out.append(self._buffer)
                    self._buffer = ""
                elif len(self._buffer) > self._keep:
                    out.append(self._buffer[:-self._keep])
                    self._buffer = self._buffer[-self._keep :]
                break

            idx = min(candidates)
            out.append(self._buffer[:idx])
            if open_idx == idx:
                self._buffer = self._buffer[idx + len(self.OPEN) :]
                self._hiding = True
            else:
                self._buffer = self._buffer[idx + len(self.CLOSE) :].lstrip()
        return "".join(out)


def hide_thinking_text(text: str) -> str:
    return ThinkingTextFilter(enabled=True).push(text, final=True)


class StopTextFilter:
    """Streaming text stop filter that does not leak split stop strings."""

    def __init__(self, stop_strings: tuple[str, ...] = ()) -> None:
        self.stop_strings = tuple(s for s in stop_strings if s)
        self._buffer = ""
        self._stopped = False
        self._keep = max((len(s) for s in self.stop_strings), default=1) - 1

    @property
    def stopped(self) -> bool:
        return self._stopped

    def push(self, text: str, *, final: bool = False) -> str:
        if self._stopped:
            return ""
        if not self.stop_strings:
            return text
        self._buffer += text
        stop_idx = self._first_stop_index(self._buffer)
        if stop_idx is not None:
            out = self._buffer[:stop_idx]
            self._buffer = ""
            self._stopped = True
            return out
        if final:
            out = self._buffer
            self._buffer = ""
            return out
        if len(self._buffer) <= self._keep:
            return ""
        out = self._buffer[: -self._keep]
        self._buffer = self._buffer[-self._keep :]
        return out

    def _first_stop_index(self, text: str) -> int | None:
        found: list[int] = []
        for stop in self.stop_strings:
            idx = text.find(stop)
            if idx >= 0:
                found.append(idx)
        return min(found) if found else None


_LEADING_ROLE_ECHO_RE = re.compile(
    r"^\s*(?:[*_`>#\-\s]{0,16})?(?:user|system)\s+(?:assistant)\s*",
    re.IGNORECASE,
)


class LeadingRoleEchoFilter:
    """Remove model-emitted ChatML role labels at the very start only."""

    def __init__(self) -> None:
        self._buffer = ""
        self._decided = False
        self._max_probe = 128

    def push(self, text: str, *, final: bool = False) -> str:
        if self._decided:
            return text
        self._buffer += text
        match = _LEADING_ROLE_ECHO_RE.match(self._buffer)
        if match is not None:
            self._decided = True
            out = self._buffer[match.end() :].lstrip()
            self._buffer = ""
            return out
        if final or len(self._buffer) >= self._max_probe:
            self._decided = True
            out = self._buffer
            self._buffer = ""
            return out
        return ""
