from __future__ import annotations

import numpy as np
import torch

from langburst.loader import (
    LowBitMarlinTensor,
    LowBitTensor,
    MARLIN_DIRECT_MAX_BATCH,
    clear_marlin_runtime_caches,
    marlin_cache_admitted,
    marlin_runtime_cache_bytes,
    marlin_should_cache_out,
)
from langburst.adapters.qwen36_tools.quantize import quantize_symmetric_lowbit


def dequant_cpu(packed: np.ndarray, scales: np.ndarray, cols: int, group_size: int) -> torch.Tensor:
    rows = packed.shape[0]
    out = torch.empty(rows, cols, dtype=torch.float32)
    for r in range(rows):
        for c in range(cols):
            byte = int(packed[r, c // 2])
            nib = (byte >> 4) & 0xF if c & 1 else byte & 0xF
            q = nib - 8
            out[r, c] = q * float(scales[r, c // group_size])
    return out


def test_lowbit_4bit_roundtrip_shape_and_error():
    torch.manual_seed(0)
    w = torch.randn(7, 257) * 0.2
    packed, scales, meta = quantize_symmetric_lowbit(w, group_size=128, bits=4)
    assert packed.shape == (7, 129)
    assert scales.shape == (7, 3)
    wdq = dequant_cpu(packed, scales, meta["cols"], meta["group_size"])
    mae = (wdq - w).abs().mean().item()
    assert mae < 0.03


def test_lowbit_4bit_gemv_cpu_reference():
    torch.manual_seed(1)
    w = torch.randn(16, 64)
    x = torch.randn(64)
    packed, scales, meta = quantize_symmetric_lowbit(w, group_size=32, bits=4)
    wdq = dequant_cpu(packed, scales, meta["cols"], meta["group_size"])
    y = wdq @ x
    assert y.shape == (16,)


def test_lowbit_tensor_supports_q3_without_model_code_changes():
    torch.manual_seed(2)
    w = torch.randn(5, 17) * 0.2
    packed, scales, meta = quantize_symmetric_lowbit(w, group_size=8, bits=3)
    t = LowBitTensor(
        name="q3_toy",
        qweight=torch.from_numpy(packed),
        scales=torch.from_numpy(scales),
        cols=meta["cols"],
        group_size=meta["group_size"],
        bits=meta["bits"],
    )
    assert packed.shape == (5, 7)
    assert t.row_dequant(0).shape == (17,)
    x = torch.randn(17, dtype=torch.float16)
    y = t.gemv(x)
    assert y.shape == (5,)
    assert torch.isfinite(y.float()).all()


def test_marlin_direct_batch_default_matches_t4_gate():
    assert MARLIN_DIRECT_MAX_BATCH == 256


def test_marlin_decode_small_cache_policy_bounds_prefill_batches(monkeypatch):
    monkeypatch.setenv("LANGBURST_MARLIN_OUT_CACHE_POLICY", "decode_small")
    monkeypatch.setenv("LANGBURST_MARLIN_OUT_CACHE_MAX_BATCH", "2")

    assert marlin_should_cache_out(1)
    assert marlin_should_cache_out(2)
    assert not marlin_should_cache_out(3)
    assert not marlin_should_cache_out(256)


def test_marlin_runtime_cache_budget_is_global(monkeypatch):
    monkeypatch.setenv("LANGBURST_MARLIN_OUT_CACHE_POLICY", "all")
    monkeypatch.setenv("LANGBURST_MARLIN_CACHE_MAX_MIB", "0")
    monkeypatch.setenv("LANGBURST_MARLIN_CACHE_MIN_FREE_MIB", "0")
    clear_marlin_runtime_caches()

    tensor = LowBitMarlinTensor(
        name="toy_marlin",
        qweight=torch.empty((1, 1), dtype=torch.int32),
        scales=torch.empty((1, 1), dtype=torch.float16),
        cols=16,
        group_size=128,
    )
    tensor._out_cache[1] = torch.empty((2, 8), dtype=torch.float16)
    assert tensor.runtime_cache_bytes() >= 32

    clear_marlin_runtime_caches()
    assert tensor.runtime_cache_bytes() == 0
    assert marlin_runtime_cache_bytes() == 0


def test_marlin_cache_admission_honors_budget(monkeypatch):
    monkeypatch.setenv("LANGBURST_MARLIN_OUT_CACHE_POLICY", "all")
    monkeypatch.setenv("LANGBURST_MARLIN_CACHE_MAX_MIB", "1")
    monkeypatch.setenv("LANGBURST_MARLIN_CACHE_MIN_FREE_MIB", "0")
    clear_marlin_runtime_caches()

    assert marlin_cache_admitted(1, device=torch.device("cpu"), new_bytes=512 * 1024)
    assert not marlin_cache_admitted(1, device=torch.device("cpu"), new_bytes=2 * 1024 * 1024)
