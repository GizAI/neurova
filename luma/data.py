from __future__ import annotations

import random
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import ByteTokenizer


NAMES = ["Mina", "Joon", "Ara", "Noah", "Yuna", "Sora", "Liam", "Eun"]
OBJECTS = ["blue key", "red notebook", "silver coin", "green map", "black card"]
PLACES = ["busan", "seoul", "lab7", "mars room", "quiet library"]
COLORS = ["cyan", "amber", "violet", "white", "orange"]
ANIMALS = ["cat", "dog", "horse", "otter", "eagle"]
IGNORE_INDEX = -100


def make_memory_story(rng: random.Random) -> str:
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    color = rng.choice(COLORS)
    code = f"{rng.choice(['AX', 'LM', 'QK', 'NV'])}-{rng.randint(100, 999)}"
    distractors = [
        f"A weather note says the sky looked {rng.choice(COLORS)}.",
        f"A draft report mentioned a {rng.choice(ANIMALS)} near the station.",
        f"Someone moved a spare {rng.choice(OBJECTS)} to {rng.choice(PLACES)}.",
        "The assistant must answer from the remembered facts only.",
    ]
    rng.shuffle(distractors)
    question = rng.choice(
        [
            (f"What object belongs to {name}?", obj),
            (f"Where should {name} go?", place),
            (f"What is {name}'s color?", color),
            (f"What is {name}'s code?", code),
        ]
    )
    return (
        f"Memory page:\n"
        f"{name} owns the {obj}.\n"
        f"{name} should go to {place}.\n"
        f"{name}'s color is {color}.\n"
        f"{name}'s code is {code}.\n"
        + "\n".join(distractors)
        + f"\nQuestion: {question[0]}\nAnswer: {question[1]}\n"
    )


@dataclass
class SyntheticMemoryDataset:
    tokenizer: ByteTokenizer
    seq_len: int = 256
    seed: int = 17

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + random.randint(0, 10_000_000))
        rows: list[list[int]] = []
        for _ in range(batch_size):
            ids = self.tokenizer.encode(make_memory_story(rng))
            if len(ids) < self.seq_len + 1:
                ids.extend([self.tokenizer.eos_id] * (self.seq_len + 1 - len(ids)))
            start = 0 if len(ids) == self.seq_len + 1 else rng.randint(0, len(ids) - self.seq_len - 1)
            rows.append(ids[start : start + self.seq_len + 1])
        x = torch.tensor([row[:-1] for row in rows], dtype=torch.long, device=device)
        y = torch.tensor([row[1:] for row in rows], dtype=torch.long, device=device)
        return x, y


def _read_text_record(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        for key in ("text", "content", "completion", "answer", "prompt"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if isinstance(obj.get("messages"), list):
            parts = []
            for msg in obj["messages"]:
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    parts.append(msg["content"])
            return "\n".join(parts)
        return stripped
    return stripped


def _find_subsequence(values: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(values):
        return -1
    last = len(values) - len(needle)
    for start in range(last + 1):
        if values[start : start + len(needle)] == needle:
            return start
    return -1


@dataclass
class PackedTextDataset:
    tokenizer: ByteTokenizer
    paths: list[Path]
    seq_len: int = 512
    max_records: int = 0
    seed: int = 17

    def __post_init__(self) -> None:
        ids: list[int] = []
        records = 0
        for path in self.paths:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    text = _read_text_record(line)
                    if not text:
                        continue
                    ids.extend(self.tokenizer.encode(text + "\n", add_bos=False, add_eos=True))
                    records += 1
                    if self.max_records and records >= self.max_records:
                        break
            if self.max_records and records >= self.max_records:
                break
        if len(ids) < self.seq_len + 2:
            raise ValueError(f"not enough tokens loaded from {self.paths}: {len(ids)}")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.records = records

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        high = len(self.ids) - self.seq_len - 1
        starts = torch.randint(0, high, (batch_size,))
        rows = torch.stack([self.ids[int(start) : int(start) + self.seq_len + 1] for start in starts])
        rows = rows.to(device=device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:]


@dataclass
class RecordTextDataset:
    tokenizer: ByteTokenizer
    paths: list[Path]
    seq_len: int = 512
    max_records: int = 0
    answer_only: bool = False

    def __post_init__(self) -> None:
        rows: list[list[int]] = []
        labels: list[list[int]] = []
        answer_marker = self.tokenizer.encode("Answer:", add_bos=False, add_eos=False)
        for path in self.paths:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    text = _read_text_record(line)
                    if not text:
                        continue
                    ids = self.tokenizer.encode(text + "\n")
                    if len(ids) < self.seq_len + 1:
                        ids.extend([self.tokenizer.pad_id] * (self.seq_len + 1 - len(ids)))
                    row = ids[: self.seq_len + 1]
                    label = row[1:].copy()
                    label = [IGNORE_INDEX if token_id == self.tokenizer.pad_id else token_id for token_id in label]
                    if self.answer_only:
                        marker_pos = _find_subsequence(row, answer_marker)
                        if marker_pos >= 0:
                            answer_start = marker_pos + len(answer_marker)
                            label = [
                                token_id if label_idx + 1 >= answer_start else IGNORE_INDEX
                                for label_idx, token_id in enumerate(label)
                            ]
                    rows.append(row)
                    labels.append(label)
                    if self.max_records and len(rows) >= self.max_records:
                        break
            if self.max_records and len(rows) >= self.max_records:
                break
        if not rows:
            raise ValueError(f"no records loaded from {self.paths}")
        self.rows = torch.tensor(rows, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.records = len(rows)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.records, (batch_size,))
        rows = self.rows[idx].to(device=device, non_blocking=True)
        labels = self.labels[idx].to(device=device, non_blocking=True)
        return rows[:, :-1], labels
