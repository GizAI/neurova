from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .ops import cuda_ops
from .tuning import lowbit_rows_per_cta


@dataclass
class LowBitTensor:
    name: str
    qweight: torch.Tensor
    scales: torch.Tensor
    cols: int
    group_size: int
    bits: int = 4

    def gemv(self, x: torch.Tensor) -> torch.Tensor:
        if self.qweight.device.type == "cuda":
            ops = cuda_ops()
            return ops.lowbit_gemv(
                self.qweight,
                self.scales,
                x.contiguous(),
                self.cols,
                self.group_size,
                self.bits,
                lowbit_rows_per_cta(),
            )
        dense = dequantize_lowbit_rows(self.qweight, self.scales, self.cols, self.group_size, self.bits)
        return torch.matmul(dense.to(device=x.device, dtype=x.dtype), x.contiguous())

    def row_dequant(self, row: int | torch.Tensor) -> torch.Tensor:
        """Dequantize one row, used for low-bit token embeddings."""
        if self.qweight.device.type != "cuda":
            return dequantize_lowbit_row(self.qweight, self.scales, row, self.cols, self.group_size, self.bits)
        ops = cuda_ops()
        if not torch.is_tensor(row):
            row = torch.tensor(int(row), device=self.qweight.device, dtype=torch.long)
        else:
            row = row.to(device=self.qweight.device, dtype=torch.long).reshape(())
        return ops.lowbit_row_dequant(self.qweight, self.scales, row, self.cols, self.group_size, self.bits)

    def gemm(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("gemm expects x [batch, cols]")
        if self.qweight.device.type == "cuda":
            ops = cuda_ops()
            return ops.lowbit_gemm(
                self.qweight,
                self.scales,
                x.contiguous(),
                self.cols,
                self.group_size,
                self.bits,
                lowbit_rows_per_cta(),
            )
        dense = dequantize_lowbit_rows(self.qweight, self.scales, self.cols, self.group_size, self.bits)
        return torch.matmul(x.contiguous().to(device=dense.device, dtype=dense.dtype), dense.t()).to(x.device)


@dataclass
class LowBitMarlinTensor:
    name: str
    qweight: torch.Tensor
    scales: torch.Tensor
    cols: int
    group_size: int
    bits: int = 4
    exec_bits: int = 4
    _out_cache: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _workspace: torch.Tensor | None = field(default=None, init=False, repr=False)

    def gemv(self, x: torch.Tensor) -> torch.Tensor:
        return self.gemm(x.reshape(1, -1).contiguous()).reshape(-1)

    def gemm(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("gemm expects x [batch, cols]")
        if self.qweight.device.type != "cuda":
            raise RuntimeError("Marlin tensors require CUDA")
        x = x.to(device=self.qweight.device, dtype=torch.float16).contiguous()
        batch = int(x.size(0))
        rows = int(self.qweight.size(1) // 2)
        out = self._out_cache.get(batch)
        if out is None or out.device != self.qweight.device or out.size(1) != rows:
            out = torch.empty((batch, rows), device=self.qweight.device, dtype=torch.float16)
            self._out_cache[batch] = out
        workspace_size = max(1, rows // 128 * 16)
        if self._workspace is None or self._workspace.device != self.qweight.device or self._workspace.numel() < workspace_size:
            self._workspace = torch.zeros((workspace_size,), device=self.qweight.device, dtype=torch.int32)
        cuda_ops().lowbit_marlin_gemm_out(
            self.qweight,
            self.scales,
            x,
            out,
            self._workspace,
            self.cols,
            self.group_size,
        )
        return out

    def row_dequant(self, row: int | torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Marlin layout does not support row_dequant; keep embeddings in rowwise layout")


def dequantize_lowbit_row(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    row: int | torch.Tensor,
    cols: int,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    if torch.is_tensor(row):
        row = int(row.item())
    qrow = qweight[int(row)]
    srow = scales[int(row)]
    vals = torch.empty((cols,), device=qweight.device, dtype=torch.float32)
    qmax = (1 << bits) - 1
    zero = 1 << (bits - 1)
    for c in range(cols):
        bit_pos = c * bits
        byte_i = bit_pos // 8
        shift = bit_pos % 8
        word = int(qrow[byte_i].item())
        if shift + bits > 8 and byte_i + 1 < qrow.numel():
            word |= int(qrow[byte_i + 1].item()) << 8
        q = ((word >> shift) & qmax) - zero
        vals[c] = q * float(srow[c // group_size].item())
    return vals.to(torch.float16)


def dequantize_lowbit_rows(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    cols: int,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    rows = qweight.shape[0]
    out = torch.empty((rows, cols), device=qweight.device, dtype=torch.float16)
    for r in range(rows):
        out[r] = dequantize_lowbit_row(qweight, scales, r, cols, group_size, bits)
    return out


@dataclass
class FP16Tensor:
    name: str
    value: torch.Tensor


class QuantizedStore:
    """QwenBurst quantized tensor store."""

    def __init__(self, root: str | Path, device: str | torch.device = "cuda"):
        self.root = Path(root)
        with open(self.root / "qwenburst_index.json", "r", encoding="utf-8") as f:
            self.index = json.load(f)
        self.device = torch.device(device)
        self.cache: dict[str, LowBitTensor | LowBitMarlinTensor | FP16Tensor] = {}

    def tensor(self, name: str) -> LowBitTensor | LowBitMarlinTensor | FP16Tensor:
        if name in self.cache:
            return self.cache[name]
        meta = self.index["tensors"][name]
        if meta["kind"] == "lowbit_symmetric_groupwise":
            bits = int(meta["bits"])
            rows = meta["rows"]
            packed_cols = meta["packed_cols"]
            n_groups = meta["n_groups"]
            q_np = np.memmap(self.root / meta["qweight"], dtype=np.uint8, mode="r", shape=(rows, packed_cols))
            s_np = np.memmap(self.root / meta["scales"], dtype=np.float16, mode="r", shape=(rows, n_groups))
            # Copy out of read-only mmap before constructing tensors.
            q = torch.from_numpy(np.array(q_np, copy=True)).to(self.device, non_blocking=True).contiguous()
            s = torch.from_numpy(np.array(s_np, copy=True)).to(self.device, non_blocking=True).contiguous()
            obj = LowBitTensor(name=name, qweight=q, scales=s, cols=meta["cols"], group_size=meta["group_size"], bits=bits)
        elif meta["kind"] == "lowbit_marlin_groupwise":
            rows = meta["cols"] // 16
            packed_cols = meta["packed_cols"]
            n_groups = meta["n_groups"]
            out_rows = meta["rows"]
            q_np = np.memmap(self.root / meta["qweight"], dtype=np.int32, mode="r", shape=(rows, packed_cols))
            s_np = np.memmap(self.root / meta["scales"], dtype=np.float16, mode="r", shape=(n_groups, out_rows))
            q = torch.from_numpy(np.array(q_np, copy=True)).to(self.device, non_blocking=True).contiguous()
            s = torch.from_numpy(np.array(s_np, copy=True)).to(self.device, non_blocking=True).contiguous()
            obj = LowBitMarlinTensor(
                name=name,
                qweight=q,
                scales=s,
                cols=meta["cols"],
                group_size=meta["group_size"],
                bits=int(meta.get("bits", 4)),
                exec_bits=int(meta.get("exec_bits", 4)),
            )
        elif meta["kind"] == "fp16_raw":
            arr = np.memmap(self.root / meta["path"], dtype=np.float16, mode="r", shape=tuple(meta["shape"]))
            val = torch.from_numpy(np.array(arr, copy=True)).to(self.device, non_blocking=True).contiguous()
            obj = FP16Tensor(name=name, value=val)
        else:
            raise ValueError(f"unknown tensor kind: {meta['kind']}")
        self.cache[name] = obj
        return obj

    def has(self, name: str) -> bool:
        return name in self.index.get("tensors", {})
