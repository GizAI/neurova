from __future__ import annotations

from collections import Counter

import torch
import torch.nn.functional as F


def apply_repetition_penalty(logits: torch.Tensor, generated_ids: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or not generated_ids:
        return logits
    out = logits.clone()
    for token_id in set(generated_ids):
        value = out[token_id]
        out[token_id] = value / penalty if value > 0 else value * penalty
    return out


def banned_ngram_tokens(generated_ids: list[int], ngram_size: int) -> set[int]:
    if ngram_size <= 0 or len(generated_ids) + 1 < ngram_size:
        return set()
    prefix = tuple(generated_ids[-(ngram_size - 1) :])
    banned: set[int] = set()
    for i in range(len(generated_ids) - ngram_size + 1):
        gram = tuple(generated_ids[i : i + ngram_size])
        if gram[:-1] == prefix:
            banned.add(gram[-1])
    return banned


def filter_logits(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
    generated_ids: list[int] | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    logits = apply_repetition_penalty(logits, generated_ids or [], repetition_penalty)
    if generated_ids and no_repeat_ngram_size > 0:
        banned = banned_ngram_tokens(generated_ids, no_repeat_ngram_size)
        if banned:
            logits = logits.clone()
            logits[list(banned)] = -torch.inf
    if top_k > 0 and top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits = logits.clone()
        logits[logits < cutoff] = -torch.inf
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        keep = torch.cumsum(probs, dim=-1) <= top_p
        keep[0] = True
        filtered = torch.full_like(logits, -torch.inf)
        filtered[sorted_idx[keep]] = logits[sorted_idx[keep]]
        logits = filtered
    return logits


def repetition_stats(text: str) -> dict[str, float | int]:
    import re

    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repeated_4gram_max": 0, "longest_word_run": 0}
    distinct_1 = len(set(words)) / len(words)
    bigrams = list(zip(words, words[1:]))
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))
    grams4 = [tuple(words[i : i + 4]) for i in range(max(0, len(words) - 3))]
    repeated_4gram_max = max(Counter(grams4).values(), default=0)
    longest = 1
    cur = 1
    for prev, word in zip(words, words[1:]):
        cur = cur + 1 if word == prev else 1
        longest = max(longest, cur)
    return {
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "repeated_4gram_max": repeated_4gram_max,
        "longest_word_run": longest,
    }
