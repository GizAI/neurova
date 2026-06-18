from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .ops import cuda_ops
from .core.platform import resolve_index_file


from .tuning import DEFAULT_MARLIN_DIRECT_MAX_BATCH, lowbit_rows_per_cta, marlin_direct_max_batch


MARLIN_DIRECT_MAX_BATCH = DEFAULT_MARLIN_DIRECT_MAX_BATCH


def _marlin_out_cache_policy() -> str:
    return os.environ.get("LANGBURST_MARLIN_OUT_CACHE_POLICY", "off").strip().lower()


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
            with torch.cuda.device(self.qweight.device):
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
        with torch.cuda.device(self.qweight.device):
            return ops.lowbit_row_dequant(self.qweight, self.scales, row, self.cols, self.group_size, self.bits)

    def gemm(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("gemm expects x [batch, cols]")
        if self.qweight.device.type == "cuda":
            ops = cuda_ops()
            with torch.cuda.device(self.qweight.device):
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
    _argmax_cache: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _argmax_state_cache: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _argmax_sync: torch.Tensor | None = field(default=None, init=False, repr=False)
    _argmax_epoch: int = field(default=0, init=False, repr=False)
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
        if batch > marlin_direct_max_batch():
            # The current vendored Marlin path is deterministic for M=1 but not
            # accepted for large padded M on Ada. Preserve block-prefill
            # semantics by evaluating rows through the stable GEMM contract.
            rows = [self.gemm(row.reshape(1, -1)).reshape(-1).clone() for row in x]
            return torch.stack(rows, dim=0).contiguous()
        rows = int(self.qweight.size(1) // 2)
        cache_policy = _marlin_out_cache_policy()
        cache_out = cache_policy not in {"0", "false", "off", "none", "no_cache"} and (
            cache_policy != "decode_only" or batch == 1
        )
        out = self._out_cache.get(batch) if cache_out else None
        if out is None or out.device != self.qweight.device or out.size(1) != rows:
            out = torch.empty((batch, rows), device=self.qweight.device, dtype=torch.float16)
            if cache_out:
                self._out_cache[batch] = out
        workspace_size = max(1, rows // 128 * 16)
        if self._workspace is None or self._workspace.device != self.qweight.device or self._workspace.numel() < workspace_size:
            self._workspace = torch.zeros((workspace_size,), device=self.qweight.device, dtype=torch.int32)
        else:
            self._workspace[:workspace_size].zero_()
        with torch.cuda.device(self.qweight.device):
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

    def gemm_silu_packed(self, mixed: torch.Tensor, hidden: int, out: torch.Tensor | None = None) -> torch.Tensor:
        if mixed.ndim == 1:
            mixed = mixed.reshape(1, -1)
        if mixed.ndim != 2:
            raise ValueError("gemm_silu_packed expects mixed [batch, 2*hidden]")
        if int(hidden) != int(self.cols):
            raise ValueError("hidden must match this Marlin tensor's input cols")
        if mixed.size(1) != 2 * int(hidden):
            raise ValueError("mixed width must be 2*hidden")
        if self.qweight.device.type != "cuda":
            raise RuntimeError("Marlin tensors require CUDA")
        mixed = mixed.to(device=self.qweight.device, dtype=torch.float16).contiguous()
        batch = int(mixed.size(0))
        rows = int(self.qweight.size(1) // 2)
        if out is None:
            out = self._out_cache.get(batch)
            if out is None or out.device != self.qweight.device or out.size(1) != rows:
                out = torch.empty((batch, rows), device=self.qweight.device, dtype=torch.float16)
                self._out_cache[batch] = out
        else:
            if out.device != self.qweight.device or out.shape != (batch, rows) or out.dtype != torch.float16:
                raise ValueError("out must be fp16 CUDA tensor with shape [batch, rows]")
        workspace_size = max(1, rows // 128 * 16)
        if self._workspace is None or self._workspace.device != self.qweight.device or self._workspace.numel() < workspace_size:
            self._workspace = torch.zeros((workspace_size,), device=self.qweight.device, dtype=torch.int32)
        else:
            self._workspace[:workspace_size].zero_()
        with torch.cuda.device(self.qweight.device):
            cuda_ops().lowbit_marlin_gemm_silu_packed_out(
                self.qweight,
                self.scales,
                mixed,
                out,
                self._workspace,
                int(hidden),
                self.group_size,
            )
        return out

    def gemm_argmax(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("gemm_argmax expects x [batch, cols]")
        if self.qweight.device.type != "cuda":
            raise RuntimeError("Marlin tensors require CUDA")
        x = x.to(device=self.qweight.device, dtype=torch.float16).contiguous()
        batch = int(x.size(0))
        rows = int(self.qweight.size(1) // 2)
        cache_policy = _marlin_out_cache_policy()
        cache_out = cache_policy not in {"0", "false", "off", "none", "no_cache"} and (
            cache_policy != "decode_only" or batch == 1
        )
        scratch_out = self._out_cache.get(batch) if cache_out else None
        if scratch_out is None or scratch_out.device != self.qweight.device or scratch_out.size(1) != rows:
            scratch_out = torch.empty((batch, rows), device=self.qweight.device, dtype=torch.float16)
            if cache_out:
                self._out_cache[batch] = scratch_out
        workspace_size = max(1, rows // 128 * 16)
        if self._workspace is None or self._workspace.device != self.qweight.device or self._workspace.numel() < workspace_size:
            self._workspace = torch.zeros((workspace_size,), device=self.qweight.device, dtype=torch.int32)
        else:
            self._workspace[:workspace_size].zero_()
        if out is None:
            out = self._argmax_cache.get(batch)
            if out is None or out.device != self.qweight.device or out.numel() != batch:
                out = torch.empty((batch,), device=self.qweight.device, dtype=torch.long)
                self._argmax_cache[batch] = out
        else:
            if out.device != self.qweight.device or out.numel() != batch or out.dtype != torch.long:
                raise ValueError("argmax out must be int64 CUDA tensor with shape [batch]")
            out = out.reshape(batch)
        state = self._argmax_state_cache.get(batch)
        if state is None or state.device != self.qweight.device or state.numel() != batch:
            state = torch.empty((batch,), device=self.qweight.device, dtype=torch.long)
            self._argmax_state_cache[batch] = state
        if self._argmax_sync is None or self._argmax_sync.device != self.qweight.device:
            self._argmax_sync = torch.empty((2,), device=self.qweight.device, dtype=torch.int32)
        self._argmax_epoch = 1 if self._argmax_epoch >= 2_000_000_000 else self._argmax_epoch + 1
        with torch.cuda.device(self.qweight.device):
            cuda_ops().lowbit_marlin_gemm_argmax_out(
                self.qweight,
                self.scales,
                x,
                scratch_out,
                self._workspace,
                state,
                out,
                self._argmax_sync,
                self._argmax_epoch,
                self.cols,
                self.group_size,
            )
        return out

    def row_dequant(self, row: int | torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Marlin layout does not support row_dequant; keep embeddings in rowwise layout")

    def clear_runtime_cache(self) -> None:
        self._out_cache.clear()
        self._argmax_cache.clear()
        self._argmax_state_cache.clear()
        self._argmax_sync = None
        self._workspace = None


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
    """Model-neutral low-bit tensor store.

    The index name is `langburst_index.json`, so Qwen/Gemma adapters can share
    the tensor loader and kernels without model-family-specific store code.
    """

    def __init__(self, root: str | Path, device: str | torch.device = "cuda"):
        self.root = Path(root)
        with open(resolve_index_file(self.root), "r", encoding="utf-8") as f:
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

    def loaded_tensor_summary(self) -> dict[str, object]:
        counts: Counter[str] = Counter()
        bytes_by_kind: defaultdict[str, int] = defaultdict(int)
        bytes_by_group: defaultdict[str, int] = defaultdict(int)

        def group_for(name: str) -> str:
            low = name.lower()
            if low.startswith("mtp."):
                return "native_mtp"
            if low.startswith("model.visual") or ".visual." in low or "vision" in low:
                return "vision"
            if "embed_tokens" in low:
                return "text_embedding"
            if "lm_head" in low or "output.weight" in low:
                return "lm_head"
            if ".layers." in low or "language_model.layers" in low:
                return "text_layers"
            return "other"

        def add(kind: str, group: str, tensor: torch.Tensor | None) -> None:
            if tensor is None:
                return
            nbytes = int(tensor.numel() * tensor.element_size())
            bytes_by_kind[kind] += nbytes
            bytes_by_group[group] += nbytes

        for name, obj in self.cache.items():
            group = group_for(name)
            if isinstance(obj, LowBitTensor):
                kind = "lowbit_symmetric_groupwise"
                counts[kind] += 1
                add(kind, group, obj.qweight)
                add(kind, group, obj.scales)
            elif isinstance(obj, LowBitMarlinTensor):
                kind = "lowbit_marlin_groupwise"
                counts[kind] += 1
                add(kind, group, obj.qweight)
                add(kind, group, obj.scales)
                add("marlin_workspace", group, obj._workspace)
                for out in obj._out_cache.values():
                    add("marlin_output_cache", group, out)
            elif isinstance(obj, FP16Tensor):
                kind = "fp16_raw"
                counts[kind] += 1
                add(kind, group, obj.value)
        return {
            "loaded_tensors": len(self.cache),
            "counts_by_kind": dict(counts),
            "mib_by_kind": {k: round(v / (1024 * 1024), 2) for k, v in sorted(bytes_by_kind.items())},
            "mib_by_group": {k: round(v / (1024 * 1024), 2) for k, v in sorted(bytes_by_group.items())},
        }

    def clear_runtime_caches(self) -> None:
        for obj in self.cache.values():
            clear = getattr(obj, "clear_runtime_cache", None)
            if callable(clear):
                clear()

    def has(self, name: str) -> bool:
        return name in self.index.get("tensors", {})
