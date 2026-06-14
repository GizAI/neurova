from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from saneflow.chat_format import ASSISTANT_HEADER, IM_END


def text_from_jsonl_line(line: str) -> str:
    line = line.strip()
    if not line:
        return ""
    if not line.startswith("{"):
        return line
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return ""
    text = row.get("text", "")
    return text if isinstance(text, str) else ""


def cache_key(paths: list[Path], tokenizer_name: str, seq_len: int, max_records: int, loss_mode: str, tokenizer_fingerprint: str = "") -> str:
    h = hashlib.sha256()
    h.update(tokenizer_name.encode())
    h.update(tokenizer_fingerprint.encode())
    h.update(str(seq_len).encode())
    h.update(str(max_records).encode())
    h.update(loss_mode.encode())
    for path in paths:
        h.update(str(path).encode())
        h.update(str(path.stat().st_mtime_ns if path.exists() else 0).encode())
        h.update(str(path.stat().st_size if path.exists() else 0).encode())
    return h.hexdigest()[:24]


def encode_text_with_mask(tokenizer, text: str, loss_mode: str) -> tuple[list[int], list[int]]:
    if loss_mode == "causal":
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids, [1] * len(ids)
    if loss_mode != "chatml_assistant":
        raise ValueError(f"unknown loss_mode: {loss_mode}")

    ids: list[int] = []
    mask: list[int] = []

    def add(fragment: str, enabled: bool) -> None:
        if not fragment:
            return
        frag_ids = tokenizer.encode(fragment, add_special_tokens=False)
        ids.extend(frag_ids)
        mask.extend([1 if enabled else 0] * len(frag_ids))

    pos = 0
    while pos < len(text):
        start = text.find(ASSISTANT_HEADER, pos)
        if start < 0:
            add(text[pos:], False)
            break
        answer_start = start + len(ASSISTANT_HEADER)
        add(text[pos:answer_start], False)
        end = text.find(IM_END, answer_start)
        if end < 0:
            add(text[answer_start:], True)
            break
        add(text[answer_start : end + len(IM_END)], True)
        pos = end + len(IM_END)
    return ids, mask


class TokenStreamDataset:
    def __init__(
        self,
        *,
        tokenizer,
        paths: list[Path],
        seq_len: int,
        max_records: int = 0,
        cache_dir: Path = Path("data/saneflow/.cache"),
        dataset_device: torch.device | None = None,
        loss_mode: str = "causal",
    ) -> None:
        self.seq_len = seq_len
        self.loss_mode = loss_mode
        self.dataset_device = dataset_device or torch.device("cpu")
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = json.dumps(
            {
                "vocab_size": len(tokenizer) if hasattr(tokenizer, "__len__") else None,
                "special_token_ids": getattr(tokenizer, "special_token_ids", {}),
            },
            sort_keys=True,
        )
        key = cache_key(paths, getattr(tokenizer, "name_or_path", "tokenizer"), seq_len, max_records, loss_mode, fingerprint)
        cache_path = cache_dir / f"tokens-{key}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                self.ids = payload["ids"]
                self.loss_mask = payload["loss_mask"]
            else:
                self.ids = payload
                self.loss_mask = torch.ones_like(self.ids, dtype=torch.bool)
            if self.dataset_device.type != "cpu":
                self.ids = self.ids.to(self.dataset_device, non_blocking=True)
                self.loss_mask = self.loss_mask.to(self.dataset_device, non_blocking=True)
            self.loss_positions = torch.nonzero(self.loss_mask, as_tuple=False).flatten()
            return

        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer must define eos_token_id")
        ids: list[int] = []
        loss_mask: list[int] = []
        records = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = text_from_jsonl_line(line)
                    text = text.strip()
                    if not text:
                        continue
                    record_ids, record_mask = encode_text_with_mask(tokenizer, text, loss_mode)
                    ids.extend(record_ids)
                    loss_mask.extend(record_mask)
                    ids.append(eos)
                    loss_mask.append(0)
                    records += 1
                    if max_records and records >= max_records:
                        break
            if max_records and records >= max_records:
                break
        if len(ids) < seq_len + 2:
            raise ValueError(f"not enough tokens: {len(ids)}")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.loss_mask = torch.tensor(loss_mask, dtype=torch.bool)
        torch.save({"ids": self.ids, "loss_mask": self.loss_mask}, cache_path)
        if self.dataset_device.type != "cpu":
            self.ids = self.ids.to(self.dataset_device, non_blocking=True)
            self.loss_mask = self.loss_mask.to(self.dataset_device, non_blocking=True)
        self.loss_positions = torch.nonzero(self.loss_mask, as_tuple=False).flatten()

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hi = self.ids.numel() - self.seq_len - 1
        sample_device = self.ids.device
        starts = torch.randint(0, hi, (batch_size,), device=sample_device)
        offsets = torch.arange(self.seq_len + 1, device=sample_device)
        rows = self.ids[starts[:, None] + offsets[None, :]]
        if sample_device != device:
            rows = rows.to(device, non_blocking=True)
        x = rows[:, :-1]
        y = rows[:, 1:]
        return x, y

    def batch_with_mask(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hi = self.ids.numel() - self.seq_len - 1
        sample_device = self.ids.device
        if self.loss_mode == "chatml_assistant" and self.loss_positions.numel() > 0:
            pick = torch.randint(0, self.loss_positions.numel(), (batch_size,), device=self.loss_positions.device)
            target_pos = self.loss_positions[pick].to(sample_device)
            lo = (target_pos - self.seq_len + 1).clamp_min(0)
            upper = target_pos.clamp_max(hi - 1)
            span = (upper - lo + 1).clamp_min(1)
            starts = lo + torch.floor(torch.rand(batch_size, device=sample_device) * span).long()
        else:
            starts = torch.randint(0, hi, (batch_size,), device=sample_device)
        offsets = torch.arange(self.seq_len + 1, device=sample_device)
        index = starts[:, None] + offsets[None, :]
        rows = self.ids[index]
        masks = self.loss_mask[index]
        if sample_device != device:
            rows = rows.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
        x = rows[:, :-1]
        y = rows[:, 1:]
        target_mask = masks[:, 1:]
        return x, y, target_mask


class SampleAlignedDataset:
    def __init__(
        self,
        *,
        tokenizer,
        paths: list[Path],
        seq_len: int,
        max_records: int = 0,
        dataset_device: torch.device | None = None,
        loss_mode: str = "causal",
    ) -> None:
        self.seq_len = seq_len
        self.loss_mode = loss_mode
        self.dataset_device = dataset_device or torch.device("cpu")
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        self.records: list[tuple[torch.Tensor, torch.Tensor]] = []
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer must define eos_token_id")
        records = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    text = text_from_jsonl_line(line)
                    text = text.strip()
                    if not text:
                        continue
                    ids, mask = encode_text_with_mask(tokenizer, text, loss_mode)
                    ids.append(eos)
                    mask.append(0)
                    if len(ids) < 2:
                        continue
                    if len(ids) > seq_len + 1:
                        if loss_mode == "chatml_assistant":
                            positions = [i for i, enabled in enumerate(mask) if enabled]
                            if positions:
                                hi = max(0, min(positions[-1], len(ids) - seq_len - 1))
                                lo = max(0, min(positions[0] - seq_len + 1, hi))
                                start = lo
                            else:
                                start = max(0, len(ids) - seq_len - 1)
                        else:
                            start = max(0, len(ids) - seq_len - 1)
                        ids = ids[start : start + seq_len + 1]
                        mask = mask[start : start + seq_len + 1]
                    self.records.append((torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)))
                    records += 1
                    if max_records and records >= max_records:
                        break
            if max_records and records >= max_records:
                break
        if not self.records:
            raise ValueError("no usable sample-aligned records")

    @property
    def ids(self) -> torch.Tensor:
        return torch.empty(sum(x.numel() for x, _ in self.records), dtype=torch.long)

    def _batch_indices(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, len(self.records), (batch_size,), device=self.dataset_device)

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        x, y, _ = self.batch_with_mask(batch_size, device)
        return x, y

    def batch_with_mask(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        picks = self._batch_indices(batch_size).tolist()
        rows = torch.full((batch_size, self.seq_len + 1), self.pad_id, dtype=torch.long)
        masks = torch.zeros((batch_size, self.seq_len + 1), dtype=torch.bool)
        for row_idx, rec_idx in enumerate(picks):
            ids, mask = self.records[int(rec_idx)]
            take = min(ids.numel(), self.seq_len + 1)
            rows[row_idx, :take] = ids[:take]
            if self.loss_mode == "causal":
                masks[row_idx, :take] = True
                masks[row_idx, take - 1 :] = False
            else:
                masks[row_idx, :take] = mask[:take]
        rows = rows.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        return rows[:, :-1], rows[:, 1:], masks[:, 1:]
