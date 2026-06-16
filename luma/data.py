from __future__ import annotations

import random
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .tokenizer import LUMATokenizer
from .chat_format import assistant_start_marker, chatml


NAMES = ["Mina", "Joon", "Ara", "Noah", "Yuna", "Sora", "Liam", "Eun"]
OBJECTS = ["blue key", "red notebook", "silver coin", "green map", "black card"]
PLACES = ["busan", "seoul", "lab7", "mars room", "quiet library"]
COLORS = ["cyan", "amber", "violet", "white", "orange"]
ANIMALS = ["cat", "dog", "horse", "otter", "eagle"]
WORDS = [
    "amber", "brisk", "cedar", "delta", "ember", "fable", "glade", "harbor",
    "ion", "juniper", "lumen", "mosaic", "nova", "onyx", "quartz", "raven",
]
IGNORE_INDEX = -100


def _tokenizer_cache_tag(tokenizer: LUMATokenizer) -> str:
    name = getattr(tokenizer, "name", tokenizer.__class__.__name__)
    return f"{name}-v{tokenizer.vocab_size}"


def _paths_cache_tag(paths: list[Path], max_records: int, extra: str) -> str:
    h = hashlib.sha256()
    h.update(extra.encode("utf-8"))
    h.update(str(max_records).encode("utf-8"))
    for path in paths:
        stat = path.stat()
        h.update(str(path.resolve()).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
    return h.hexdigest()[:24]


def _cache_path(cache_dir: Path | None, name: str, tag: str) -> Path | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}-{tag}.pt"


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
    tokenizer: LUMATokenizer
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


def make_slot_proof_story(rng: random.Random, gap_lines: int = 8) -> tuple[str, str]:
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    color = rng.choice(COLORS)
    code = f"{rng.choice(['AX', 'LM', 'QK', 'NV'])}-{rng.randint(100, 999)}"
    facts = {
        "object": obj,
        "place": place,
        "color": color,
        "code": code,
    }
    key, answer = rng.choice(list(facts.items()))
    question = {
        "object": f"What object belongs to {name}?",
        "place": f"Where should {name} go?",
        "color": f"What is {name}'s color?",
        "code": f"What is {name}'s code?",
    }[key]
    gap = "\n".join(
        "Irrelevant note: " + " ".join(rng.choice(WORDS) for _ in range(12)) + "."
        for _ in range(gap_lines)
    )
    user = (
        "Memory page:\n"
        f"{name} owns the {obj}.\n"
        f"{name} should go to {place}.\n"
        f"{name}'s color is {color}.\n"
        f"{name}'s code is {code}.\n"
        f"{gap}\n"
        f"Question: {question}"
    )
    return chatml("You are LUMA, a precise memory assistant.", user), answer


@dataclass
class SyntheticSlotProofDataset:
    tokenizer: LUMATokenizer
    seq_len: int = 512
    gap_lines: int = 8
    seed: int = 4242
    records: int = 1_000_000

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + random.randint(0, 10_000_000))
        rows: list[list[int]] = []
        labels: list[list[int]] = []
        for _ in range(batch_size):
            prompt, answer = make_slot_proof_story(rng, self.gap_lines)
            prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
            ids = self.tokenizer.encode(prompt + answer + "<|im_end|>\n", add_bos=True, add_eos=True)
            if len(ids) < self.seq_len + 1:
                ids.extend([self.tokenizer.pad_id] * (self.seq_len + 1 - len(ids)))
            row = ids[: self.seq_len + 1]
            label = row[1:].copy()
            label = [IGNORE_INDEX if token_id == self.tokenizer.pad_id else token_id for token_id in label]
            answer_start = min(len(prompt_ids), self.seq_len)
            label = [
                token_id if label_idx + 1 >= answer_start else IGNORE_INDEX
                for label_idx, token_id in enumerate(label)
            ]
            rows.append(row)
            labels.append(label)
        x = torch.tensor([row[:-1] for row in rows], dtype=torch.long, device=device)
        y = torch.tensor(labels, dtype=torch.long, device=device)
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


def _answer_start_token_index(tokenizer: LUMATokenizer, text: str) -> int | None:
    marker = assistant_start_marker()
    marker_idx = text.rfind(marker)
    if marker_idx >= 0:
        prefix = text[: marker_idx + len(marker)]
        return len(tokenizer.encode(prefix, add_bos=True, add_eos=False))

    for marker in ("\nAnswer:", "Answer:"):
        marker_idx = text.rfind(marker)
        if marker_idx >= 0:
            prefix = text[: marker_idx + len(marker)]
            return len(tokenizer.encode(prefix, add_bos=True, add_eos=False))
    return None


@dataclass
class PackedTextDataset:
    tokenizer: LUMATokenizer
    paths: list[Path]
    seq_len: int = 512
    max_records: int = 0
    seed: int = 17
    cache_dir: Path | None = Path("luma/data/.luma_cache")

    def __post_init__(self) -> None:
        tag = _paths_cache_tag(
            self.paths,
            self.max_records,
            f"packed-seq{self.seq_len}-{_tokenizer_cache_tag(self.tokenizer)}",
        )
        cache = _cache_path(self.cache_dir, "packed", tag)
        if cache is not None and cache.exists():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            self.ids = payload["ids"].long()
            self.records = int(payload["records"])
            return
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
        if cache is not None:
            torch.save({"ids": self.ids, "records": self.records}, cache)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        high = len(self.ids) - self.seq_len - 1
        starts = torch.randint(0, high, (batch_size,))
        rows = torch.stack([self.ids[int(start) : int(start) + self.seq_len + 1] for start in starts])
        rows = rows.to(device=device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:]


@dataclass
class RecordTextDataset:
    tokenizer: LUMATokenizer
    paths: list[Path]
    seq_len: int = 512
    max_records: int = 0
    answer_only: bool = False
    cache_dir: Path | None = Path("luma/data/.luma_cache")

    def __post_init__(self) -> None:
        tag = _paths_cache_tag(
            self.paths,
            self.max_records,
            f"records-seq{self.seq_len}-answer{int(self.answer_only)}-{_tokenizer_cache_tag(self.tokenizer)}",
        )
        cache = _cache_path(self.cache_dir, "records", tag)
        if cache is not None and cache.exists():
            payload = torch.load(cache, map_location="cpu", weights_only=True)
            self.rows = payload["rows"].long()
            self.labels = payload["labels"].long()
            self.records = int(payload["records"])
            return
        rows: list[list[int]] = []
        labels: list[list[int]] = []
        for path in self.paths:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    text = _read_text_record(line)
                    if not text:
                        continue
                    ids = self.tokenizer.encode(text + "\n")
                    answer_start = 0
                    if self.answer_only:
                        maybe_answer_start = _answer_start_token_index(self.tokenizer, text)
                        if maybe_answer_start is None:
                            continue
                        answer_start = maybe_answer_start
                    if len(ids) < self.seq_len + 1:
                        ids.extend([self.tokenizer.pad_id] * (self.seq_len + 1 - len(ids)))
                    row = ids[: self.seq_len + 1]
                    label = row[1:].copy()
                    label = [IGNORE_INDEX if token_id == self.tokenizer.pad_id else token_id for token_id in label]
                    if self.answer_only and answer_start:
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
        if cache is not None:
            torch.save({"rows": self.rows, "labels": self.labels, "records": self.records}, cache)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.records, (batch_size,))
        rows = self.rows[idx].to(device=device, non_blocking=True)
        labels = self.labels[idx].to(device=device, non_blocking=True)
        return rows[:, :-1], labels


@dataclass
class WeightedMixedDataset:
    components: list[tuple[str, object, float]]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("WeightedMixedDataset requires at least one component")
        weights = torch.tensor([max(0.0, float(weight)) for _, _, weight in self.components], dtype=torch.float)
        if float(weights.sum()) <= 0:
            raise ValueError("mixed dataset weights must sum to a positive value")
        self.weights = weights / weights.sum()
        self.records = sum(int(getattr(dataset, "records", 0)) for _, dataset, _ in self.components)

    def summary(self) -> list[dict]:
        return [
            {
                "name": name,
                "weight": float(weight),
                "records": int(getattr(dataset, "records", 0)),
                "tokens": int(getattr(dataset, "rows", getattr(dataset, "ids", torch.empty(0))).numel()),
            }
            for (name, dataset, _), weight in zip(self.components, self.weights.tolist())
        ]

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        choices = torch.multinomial(self.weights, batch_size, replacement=True)
        xs: list[torch.Tensor] = []
        ys: list[torch.Tensor] = []
        for component_idx, (name, dataset, _) in enumerate(self.components):
            count = int((choices == component_idx).sum().item())
            if count == 0:
                continue
            x, y = dataset.batch(count, device)
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        order = torch.randperm(x.size(0), device=device)
        return x.index_select(0, order), y.index_select(0, order)
