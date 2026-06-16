from __future__ import annotations

import json
import random
from itertools import cycle
from pathlib import Path
from typing import Iterable, Protocol


class EncodesText(Protocol):
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]: ...


def iter_texts(paths: list[Path]) -> Iterable[str]:
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        yield line
                        continue
                    yield _json_text(obj)
        else:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line


def _json_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("text", "content", "prompt", "input", "output", "response", "completion"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if isinstance(obj.get("messages"), list):
            parts = []
            for message in obj["messages"]:
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    role = message.get("role", "user")
                    parts.append(f"{role}: {message['content'].strip()}")
            if parts:
                return "\n".join(parts)
        parts = []
        for value in obj.values():
            if isinstance(value, str) and value.strip() and value[:1] not in "[{":
                parts.append(value.strip())
        return "\n".join(parts)
    return json.dumps(obj, ensure_ascii=False)


def iter_packed_token_batches(
    tokenizer: EncodesText,
    paths: list[Path],
    seq_len: int,
    batch_size: int,
    device: str,
    *,
    shuffle_texts: bool = False,
    seed: int = 0,
    max_text_chars: int | None = 65536,
    max_text_tokens: int | None = 120000,
):
    import torch

    texts = list(iter_texts(paths))
    if not texts:
        raise ValueError("no training text found")
    if shuffle_texts:
        rng = random.Random(seed)
        rng.shuffle(texts)
    stream = cycle(texts)
    buffer: list[int] = []
    needed = batch_size * (seq_len + 1)
    while True:
        while len(buffer) < needed:
            text = next(stream)
            for ids in _encode_bounded_text(
                tokenizer,
                text,
                max_text_chars=max_text_chars,
                max_text_tokens=max_text_tokens,
            ):
                buffer.extend(ids)
        rows = []
        for _ in range(batch_size):
            rows.append(buffer[: seq_len + 1])
            del buffer[: seq_len + 1]
        yield torch.tensor(rows, device=device, dtype=torch.long)


def _split_long_texts(texts: Iterable[str], *, max_text_chars: int | None) -> Iterable[str]:
    if max_text_chars is None or max_text_chars <= 0:
        yield from texts
        return
    for text in texts:
        if len(text) <= max_text_chars:
            yield text
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + max_text_chars)
            if end < len(text):
                split_at = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                if split_at > start + max_text_chars // 2:
                    end = split_at
            chunk = text[start:end].strip()
            if chunk:
                yield chunk
            start = end


def _encode_bounded_text(
    tokenizer: EncodesText,
    text: str,
    *,
    max_text_chars: int | None,
    max_text_tokens: int | None,
) -> Iterable[list[int]]:
    stack = list(_split_long_texts([text], max_text_chars=max_text_chars))
    while stack:
        chunk = stack.pop(0)
        ids = tokenizer.encode(chunk, add_bos=True, add_eos=True)
        if max_text_tokens is not None and max_text_tokens > 0 and len(ids) > max_text_tokens and len(chunk) > 1:
            left, right = _split_text_once(chunk)
            stack.insert(0, right)
            stack.insert(0, left)
            continue
        yield ids


def _split_text_once(text: str) -> tuple[str, str]:
    mid = max(1, len(text) // 2)
    split_at = max(text.rfind("\n", 0, mid), text.rfind(" ", 0, mid))
    if split_at <= 0:
        split_at = min(len(text) - 1, mid)
    left = text[:split_at].strip()
    right = text[split_at:].strip()
    if not left or not right:
        return text[:mid].strip(), text[mid:].strip()
    return left, right
