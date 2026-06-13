from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def state_summary(cache: dict[int, tuple[torch.Tensor, ...]]) -> dict[str, Any]:
    layers = {}
    total_bytes = 0
    for layer_idx, tensors in cache.items():
        shapes = []
        for tensor in tensors:
            shapes.append(list(tensor.shape))
            total_bytes += tensor.numel() * tensor.element_size()
        layers[str(layer_idx)] = shapes
    return {"layers": layers, "bytes": total_bytes}


def save_state(cache: dict[int, tuple[torch.Tensor, ...]], path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_cache = {
        int(layer_idx): tuple(t.detach().cpu() for t in tensors)
        for layer_idx, tensors in cache.items()
    }
    payload = {"cache": cpu_cache, "metadata": metadata or {}}
    torch.save(payload, path)
    path.with_suffix(".json").write_text(
        json.dumps({"summary": state_summary(cpu_cache), "metadata": metadata or {}}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_payload(path: Path) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict) and "cache" in raw:
        return raw
    return {"cache": raw, "metadata": {}}


def load_state(path: Path, device: str, dtype: torch.dtype) -> dict[int, tuple[torch.Tensor, ...]]:
    raw = load_payload(path)["cache"]
    restored = {}
    for layer_idx, tensors in raw.items():
        moved = []
        for tensor in tensors:
            if tensor.dtype.is_floating_point and tensor.dtype != torch.float32:
                moved.append(tensor.to(device=device, dtype=dtype))
            else:
                moved.append(tensor.to(device=device))
        restored[int(layer_idx)] = tuple(moved)
    return restored
