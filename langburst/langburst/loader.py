from __future__ import annotations

import json
import os
import weakref
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .ops import cuda_ops
from .core.platform import resolve_index_file


from .tuning import DEFAULT_MARLIN_DIRECT_MAX_BATCH, lowbit_rows_per_cta, marlin_direct_max_batch


MARLIN_DIRECT_MAX_BATCH = DEFAULT_MARLIN_DIRECT_MAX_BATCH
_MIB = 1024 * 1024
_MARLIN_RUNTIME_CACHE_REFS: list[weakref.ReferenceType["LowBitMarlinTensor"]] = []
_MARLIN_RUNTIME_CACHE_BYTES = 0


def _marlin_out_cache_policy() -> str:
    return os.environ.get("LANGBURST_MARLIN_OUT_CACHE_POLICY", "off").strip().lower()


def _marlin_out_cache_max_batch() -> int:
    raw = os.environ.get("LANGBURST_MARLIN_OUT_CACHE_MAX_BATCH", "2").strip()
    if not raw:
        return 2
    return max(1, int(raw))


def _env_mib(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _marlin_cache_max_bytes() -> int:
    return _env_mib("LANGBURST_MARLIN_CACHE_MAX_MIB", 384) * _MIB


def _marlin_cache_min_free_bytes() -> int:
    return _env_mib("LANGBURST_MARLIN_CACHE_MIN_FREE_MIB", 384) * _MIB


def marlin_should_cache_out(batch: int) -> bool:
    policy = _marlin_out_cache_policy()
    if policy in {"0", "false", "off", "none", "no_cache"}:
        return False
    if policy == "decode_only":
        return int(batch) == 1
    if policy in {"decode_small", "small", "bounded"}:
        return int(batch) <= _marlin_out_cache_max_batch()
    return True


def _tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return int(tensor.numel() * tensor.element_size())


def _dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16, torch.int16, torch.uint16}:
        return 2
    if dtype in {torch.float32, torch.int32, torch.uint32}:
        return 4
    if dtype in {torch.float64, torch.int64, torch.long}:
        return 8
    if dtype in {torch.int8, torch.uint8, torch.bool}:
        return 1
    return torch.empty((), dtype=dtype).element_size()


def _add_marlin_runtime_cache_bytes(delta: int) -> None:
    global _MARLIN_RUNTIME_CACHE_BYTES
    _MARLIN_RUNTIME_CACHE_BYTES = max(0, _MARLIN_RUNTIME_CACHE_BYTES + int(delta))


def _register_marlin_runtime_cache(tensor: "LowBitMarlinTensor") -> None:
    _MARLIN_RUNTIME_CACHE_REFS.append(weakref.ref(tensor))


def _iter_marlin_runtime_tensors() -> Iterator["LowBitMarlinTensor"]:
    live_refs: list[weakref.ReferenceType["LowBitMarlinTensor"]] = []
    for ref in _MARLIN_RUNTIME_CACHE_REFS:
        obj = ref()
        if obj is None:
            continue
        live_refs.append(ref)
        yield obj
    if len(live_refs) != len(_MARLIN_RUNTIME_CACHE_REFS):
        _MARLIN_RUNTIME_CACHE_REFS[:] = live_refs


def marlin_runtime_cache_bytes() -> int:
    return int(_MARLIN_RUNTIME_CACHE_BYTES)


def clear_marlin_runtime_caches() -> None:
    for obj in _iter_marlin_runtime_tensors():
        obj.clear_runtime_cache()


def marlin_runtime_cache_summary() -> dict[str, int]:
    return {
        "tensors": sum(1 for _ in _iter_marlin_runtime_tensors()),
        "bytes": marlin_runtime_cache_bytes(),
        "mib": round(marlin_runtime_cache_bytes() / _MIB),
        "max_mib": _marlin_cache_max_bytes() // _MIB,
        "min_free_mib": _marlin_cache_min_free_bytes() // _MIB,
    }


def marlin_cache_admitted(batch: int, *, device: torch.device, new_bytes: int = 0) -> bool:
    if not marlin_should_cache_out(batch):
        return False
    max_bytes = _marlin_cache_max_bytes()
    if max_bytes > 0 and marlin_runtime_cache_bytes() + int(new_bytes) > max_bytes:
        return False
    min_free = _marlin_cache_min_free_bytes()
    if min_free <= 0 or device.type != "cuda" or not torch.cuda.is_available():
        return True
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return int(free_bytes) - int(new_bytes) >= min_free


def _empty_cuda_tensor(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.empty(shape, device=device, dtype=dtype)
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
        clear_marlin_runtime_caches()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return torch.empty(shape, device=device, dtype=dtype)


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

    def __post_init__(self) -> None:
        _register_marlin_runtime_cache(self)

    def runtime_cache_bytes(self) -> int:
        total = _tensor_nbytes(self._workspace)
        total += _tensor_nbytes(self._argmax_sync)
        total += sum(_tensor_nbytes(tensor) for tensor in self._out_cache.values())
        total += sum(_tensor_nbytes(tensor) for tensor in self._argmax_cache.values())
        total += sum(_tensor_nbytes(tensor) for tensor in self._argmax_state_cache.values())
        return total

    def _put_out_cache(self, batch: int, tensor: torch.Tensor) -> None:
        old = self._out_cache.get(batch)
        _add_marlin_runtime_cache_bytes(_tensor_nbytes(tensor) - _tensor_nbytes(old))
        self._out_cache[batch] = tensor

    def _put_argmax_cache(self, batch: int, tensor: torch.Tensor) -> None:
        old = self._argmax_cache.get(batch)
        _add_marlin_runtime_cache_bytes(_tensor_nbytes(tensor) - _tensor_nbytes(old))
        self._argmax_cache[batch] = tensor

    def _put_argmax_state_cache(self, batch: int, tensor: torch.Tensor) -> None:
        old = self._argmax_state_cache.get(batch)
        _add_marlin_runtime_cache_bytes(_tensor_nbytes(tensor) - _tensor_nbytes(old))
        self._argmax_state_cache[batch] = tensor

    def _put_workspace(self, tensor: torch.Tensor | None) -> None:
        _add_marlin_runtime_cache_bytes(_tensor_nbytes(tensor) - _tensor_nbytes(self._workspace))
        self._workspace = tensor

    def _put_argmax_sync(self, tensor: torch.Tensor | None) -> None:
        _add_marlin_runtime_cache_bytes(_tensor_nbytes(tensor) - _tensor_nbytes(self._argmax_sync))
        self._argmax_sync = tensor

    def _cached_output(self, batch: int, rows: int, *, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        device = self.qweight.device
        cached = self._out_cache.get(batch)
        if cached is not None and cached.device == device and cached.shape == (batch, rows) and cached.dtype == dtype:
            return cached
        nbytes = batch * rows * _dtype_nbytes(dtype)
        if marlin_cache_admitted(batch, device=device, new_bytes=nbytes):
            out = _empty_cuda_tensor((batch, rows), device=device, dtype=dtype)
            self._put_out_cache(batch, out)
            return out
        return _empty_cuda_tensor((batch, rows), device=device, dtype=dtype)

    def _cached_argmax(self, batch: int) -> torch.Tensor:
        device = self.qweight.device
        cached = self._argmax_cache.get(batch)
        if cached is not None and cached.device == device and cached.numel() == batch:
            return cached
        nbytes = batch * _dtype_nbytes(torch.long)
        if marlin_cache_admitted(batch, device=device, new_bytes=nbytes):
            out = _empty_cuda_tensor((batch,), device=device, dtype=torch.long)
            self._put_argmax_cache(batch, out)
            return out
        return _empty_cuda_tensor((batch,), device=device, dtype=torch.long)

    def _cached_argmax_state(self, batch: int) -> torch.Tensor:
        device = self.qweight.device
        cached = self._argmax_state_cache.get(batch)
        if cached is not None and cached.device == device and cached.numel() == batch:
            return cached
        nbytes = batch * _dtype_nbytes(torch.long)
        if marlin_cache_admitted(batch, device=device, new_bytes=nbytes):
            state = _empty_cuda_tensor((batch,), device=device, dtype=torch.long)
            self._put_argmax_state_cache(batch, state)
            return state
        return _empty_cuda_tensor((batch,), device=device, dtype=torch.long)

    def _cached_workspace(self, workspace_size: int) -> torch.Tensor:
        device = self.qweight.device
        nbytes = int(workspace_size) * _dtype_nbytes(torch.int32)
        if (
            self._workspace is None
            or self._workspace.device != device
            or self._workspace.numel() < workspace_size
        ):
            if marlin_cache_admitted(1, device=device, new_bytes=nbytes):
                self._put_workspace(torch.zeros((workspace_size,), device=device, dtype=torch.int32))
                return self._workspace
            return torch.zeros((workspace_size,), device=device, dtype=torch.int32)
        self._workspace[:workspace_size].zero_()
        return self._workspace

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
        out = self._cached_output(batch, rows)
        workspace_size = max(1, rows // 128 * 16)
        workspace = self._cached_workspace(workspace_size)
        with torch.cuda.device(self.qweight.device):
            cuda_ops().lowbit_marlin_gemm_out(
                self.qweight,
                self.scales,
                x,
                out,
                workspace,
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
            out = self._cached_output(batch, rows)
        else:
            if out.device != self.qweight.device or out.shape != (batch, rows) or out.dtype != torch.float16:
                raise ValueError("out must be fp16 CUDA tensor with shape [batch, rows]")
        workspace_size = max(1, rows // 128 * 16)
        workspace = self._cached_workspace(workspace_size)
        with torch.cuda.device(self.qweight.device):
            cuda_ops().lowbit_marlin_gemm_silu_packed_out(
                self.qweight,
                self.scales,
                mixed,
                out,
                workspace,
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
        scratch_out = self._cached_output(batch, rows)
        workspace_size = max(1, rows // 128 * 16)
        workspace = self._cached_workspace(workspace_size)
        if out is None:
            out = self._cached_argmax(batch)
        else:
            if out.device != self.qweight.device or out.numel() != batch or out.dtype != torch.long:
                raise ValueError("argmax out must be int64 CUDA tensor with shape [batch]")
            out = out.reshape(batch)
        state = self._cached_argmax_state(batch)
        if self._argmax_sync is None or self._argmax_sync.device != self.qweight.device:
            if marlin_cache_admitted(batch, device=self.qweight.device, new_bytes=2 * 4):
                self._put_argmax_sync(_empty_cuda_tensor((2,), device=self.qweight.device, dtype=torch.int32))
            else:
                self._put_argmax_sync(None)
        sync = self._argmax_sync
        if sync is None:
            sync = _empty_cuda_tensor((2,), device=self.qweight.device, dtype=torch.int32)
        self._argmax_epoch = 1 if self._argmax_epoch >= 2_000_000_000 else self._argmax_epoch + 1
        with torch.cuda.device(self.qweight.device):
            cuda_ops().lowbit_marlin_gemm_argmax_out(
                self.qweight,
                self.scales,
                x,
                scratch_out,
                workspace,
                state,
                out,
                sync,
                self._argmax_epoch,
                self.cols,
                self.group_size,
            )
        return out

    def row_dequant(self, row: int | torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Marlin layout does not support row_dequant; keep embeddings in rowwise layout")

    def clear_runtime_cache(self) -> None:
        _add_marlin_runtime_cache_bytes(-self.runtime_cache_bytes())
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
                add("marlin_argmax_sync", group, obj._argmax_sync)
                for out in obj._out_cache.values():
                    add("marlin_output_cache", group, out)
                for out in obj._argmax_cache.values():
                    add("marlin_argmax_cache", group, out)
                for out in obj._argmax_state_cache.values():
                    add("marlin_argmax_state_cache", group, out)
            elif isinstance(obj, FP16Tensor):
                kind = "fp16_raw"
                counts[kind] += 1
                add(kind, group, obj.value)
        return {
            "loaded_tensors": len(self.cache),
            "counts_by_kind": dict(counts),
            "mib_by_kind": {k: round(v / (1024 * 1024), 2) for k, v in sorted(bytes_by_kind.items())},
            "mib_by_group": {k: round(v / (1024 * 1024), 2) for k, v in sorted(bytes_by_group.items())},
            "marlin_runtime_cache": marlin_runtime_cache_summary(),
        }

    def clear_runtime_caches(self) -> None:
        for obj in self.cache.values():
            clear = getattr(obj, "clear_runtime_cache", None)
            if callable(clear):
                clear()

    def has(self, name: str) -> bool:
        return name in self.index.get("tensors", {})
