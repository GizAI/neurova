from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

KVCacheDType = Literal["fp16", "fp8_e4m3", "int4", "int4_bdr"]
SUPPORTED_KV_CACHE_DTYPES: tuple[KVCacheDType, ...] = ("fp16", "fp8_e4m3", "int4", "int4_bdr")


def kv_buffer(shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Allocate KV storage without assuming every dtype supports CPU fill.

    PyTorch exposes float8 tensors on CPU, but common fill/zero kernels are not
    implemented there. Paged KV only reads positions that the scheduler has
    written, so uninitialized allocation is the correct low-level contract for
    fp8 storage. Non-fp8 buffers stay zeroed for existing fallback tests.
    """

    if dtype in (getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)):
        return torch.empty(shape, device=device, dtype=dtype)
    return torch.zeros(shape, device=device, dtype=dtype)


@dataclass(frozen=True)
class KVCacheSpec:
    """Single runtime contract for attention KV storage.

    external serving engine exposes this as ``kv_cache_dtype`` plus per-attention-layer
    ``k_scale``/``v_scale``. LangBurst keeps the same boundary, but extends it
    with serving-compatible INT4/BDR metadata. Model code must ask this spec for
    storage dtype, packed dimensions, byte accounting, and rotation settings
    instead of branching on string names.
    """

    dtype: KVCacheDType = "fp16"
    k_scale: float = 1.0
    v_scale: float = 1.0
    hadamard_order: int = 128
    rotate_v: bool = False

    @classmethod
    def resolve(
        cls,
        value: str | None = None,
        *,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        hadamard_order: int | None = None,
        rotate_v: bool = False,
    ) -> "KVCacheSpec":
        raw = (value or "fp16").strip().lower().replace("-", "_")
        aliases = {
            "auto": "fp16",
            "float16": "fp16",
            "half": "fp16",
            "fp8": "fp8_e4m3",
            "fp8_e4m3fn": "fp8_e4m3",
            "float8_e4m3": "fp8_e4m3",
            "float8_e4m3fn": "fp8_e4m3",
            "q4": "int4",
            "int4_kv": "int4",
            "kv_int4": "int4",
            "bdr": "int4_bdr",
            "int4_bdr_k": "int4_bdr",
            "bdr_int4": "int4_bdr",
        }
        dtype = aliases.get(raw, raw)
        if dtype not in {"fp16", "fp8_e4m3", "int4", "int4_bdr"}:
            raise ValueError("kv_cache_dtype must be one of: fp16, fp8_e4m3, int4, int4_bdr")
        if dtype == "fp8_e4m3" and not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for fp8_e4m3 KV cache")
        if k_scale <= 0.0 or v_scale <= 0.0:
            raise ValueError("KV cache k_scale/v_scale must be positive")
        order = int(hadamard_order or 128)
        if order < 1 or order & (order - 1):
            raise ValueError("KV BDR hadamard_order must be a positive power of two")
        return cls(
            dtype=dtype,  # type: ignore[arg-type]
            k_scale=float(k_scale),
            v_scale=float(v_scale),
            hadamard_order=order,
            rotate_v=bool(rotate_v),
        )

    @property
    def storage_dtype(self) -> torch.dtype:
        if self.is_int4:
            return torch.uint8
        if self.dtype == "fp8_e4m3":
            return torch.float8_e4m3fn
        return torch.float16

    @property
    def bytes_per_value(self) -> float:
        if self.is_int4:
            return 0.5
        return 1.0 if self.dtype == "fp8_e4m3" else 2.0

    @property
    def is_fp8(self) -> bool:
        return self.dtype == "fp8_e4m3"

    @property
    def is_int4(self) -> bool:
        return self.dtype in {"int4", "int4_bdr"}

    @property
    def uses_bdr(self) -> bool:
        return self.dtype == "int4_bdr"

    def storage_head_dim(self, head_dim: int) -> int:
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        if self.is_int4:
            if head_dim % 2:
                raise ValueError("INT4 KV cache requires an even head_dim")
            if self.uses_bdr and head_dim % self.hadamard_order:
                raise ValueError("BDR hadamard_order must divide head_dim")
            return head_dim // 2
        return head_dim

    def scale_shape(self, *prefix: int) -> tuple[int, ...] | None:
        if not self.is_int4:
            return None
        return tuple(int(v) for v in prefix)

    def summary(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "k_scale": self.k_scale,
            "v_scale": self.v_scale,
            "hadamard_order": self.hadamard_order,
            "rotate_v": self.rotate_v,
        }


@dataclass(frozen=True)
class KVCacheLayout:
    """Model-agnostic physical KV layout.

    Adapters provide only attention layer ids, KV head count, and head_dim. The
    core layout owns packed dimensions, per-token metadata shapes, and byte
    accounting so every model family consumes the same runtime option contract.
    """

    layers: tuple[int, ...]
    num_kv_heads: int
    head_dim: int

    @classmethod
    def from_parts(cls, *, layers: list[int] | tuple[int, ...], num_kv_heads: int, head_dim: int) -> "KVCacheLayout":
        if num_kv_heads < 1:
            raise ValueError("num_kv_heads must be positive")
        if head_dim < 1:
            raise ValueError("head_dim must be positive")
        return cls(layers=tuple(int(v) for v in layers), num_kv_heads=int(num_kv_heads), head_dim=int(head_dim))

    def storage_head_dim(self, spec: KVCacheSpec) -> int:
        return spec.storage_head_dim(self.head_dim)

    def layer_kv_shape(self, spec: KVCacheSpec, seq_len: int, *, leading_shape: tuple[int, ...] = ()) -> tuple[int, ...]:
        return (*leading_shape, self.num_kv_heads, int(seq_len), self.storage_head_dim(spec))

    def layer_meta_shape(self, seq_len: int, *, leading_shape: tuple[int, ...] = ()) -> tuple[int, ...]:
        return (*leading_shape, self.num_kv_heads, int(seq_len))

    def values_bytes(self, spec: KVCacheSpec, seq_len: int, *, leading_count: int = 1) -> float:
        return (
            len(self.layers)
            * 2
            * int(leading_count)
            * self.num_kv_heads
            * int(seq_len)
            * self.head_dim
            * spec.bytes_per_value
        )

    def metadata_bytes(self, spec: KVCacheSpec, seq_len: int, *, leading_count: int = 1) -> int:
        if not spec.is_int4:
            return 0
        # K/V each need scale and zero point. Metadata is fp16 in LangBurst.
        return len(self.layers) * 2 * 2 * int(leading_count) * self.num_kv_heads * int(seq_len) * 2

    def total_bytes(self, spec: KVCacheSpec, seq_len: int, *, leading_count: int = 1) -> int:
        return int(self.values_bytes(spec, seq_len, leading_count=leading_count) + self.metadata_bytes(spec, seq_len, leading_count=leading_count))


@dataclass
class KVCacheTensors:
    k: dict[int, torch.Tensor]
    v: dict[int, torch.Tensor]
    k_scale: dict[int, torch.Tensor] | None = None
    v_scale: dict[int, torch.Tensor] | None = None
    k_zero: dict[int, torch.Tensor] | None = None
    v_zero: dict[int, torch.Tensor] | None = None

    @property
    def has_metadata(self) -> bool:
        return self.k_scale is not None

    def slot_view(self, slot: int) -> "KVCacheTensors":
        return KVCacheTensors(
            k={layer: tensor[slot] for layer, tensor in self.k.items()},
            v={layer: tensor[slot] for layer, tensor in self.v.items()},
            k_scale={layer: tensor[slot] for layer, tensor in self.k_scale.items()} if self.k_scale is not None else None,
            v_scale={layer: tensor[slot] for layer, tensor in self.v_scale.items()} if self.v_scale is not None else None,
            k_zero={layer: tensor[slot] for layer, tensor in self.k_zero.items()} if self.k_zero is not None else None,
            v_zero={layer: tensor[slot] for layer, tensor in self.v_zero.items()} if self.v_zero is not None else None,
        )

    def zero_slot_(self, slot: int) -> None:
        for tensor in self.k.values():
            if tensor.size(-2) > 0:
                tensor[slot].zero_()
        for tensor in self.v.values():
            if tensor.size(-2) > 0:
                tensor[slot].zero_()
        for tensor in (self.k_scale or {}).values():
            tensor[slot].fill_(1.0)
        for tensor in (self.v_scale or {}).values():
            tensor[slot].fill_(1.0)
        for tensor in (self.k_zero or {}).values():
            tensor[slot].zero_()
        for tensor in (self.v_zero or {}).values():
            tensor[slot].zero_()


def allocate_kv_cache_tensors(
    layout: KVCacheLayout,
    spec: KVCacheSpec,
    *,
    seq_len: int,
    device: torch.device,
    leading_shape: tuple[int, ...] = (),
) -> KVCacheTensors:
    shape = layout.layer_kv_shape(spec, seq_len, leading_shape=leading_shape)
    k = {layer: kv_buffer(shape, device=device, dtype=spec.storage_dtype) for layer in layout.layers}
    v = {layer: kv_buffer(shape, device=device, dtype=spec.storage_dtype) for layer in layout.layers}
    if not spec.is_int4:
        return KVCacheTensors(k=k, v=v)
    meta_shape = layout.layer_meta_shape(seq_len, leading_shape=leading_shape)
    k_scale = {layer: torch.ones(meta_shape, device=device, dtype=torch.float16) for layer in layout.layers}
    v_scale = {layer: torch.ones_like(k_scale[layer]) for layer in layout.layers}
    k_zero = {layer: torch.zeros_like(k_scale[layer]) for layer in layout.layers}
    v_zero = {layer: torch.zeros_like(k_scale[layer]) for layer in layout.layers}
    return KVCacheTensors(k=k, v=v, k_scale=k_scale, v_scale=v_scale, k_zero=k_zero, v_zero=v_zero)


def hadamard_transform(x: torch.Tensor, order: int) -> torch.Tensor:
    if order <= 1:
        return x
    if x.size(-1) % order:
        raise ValueError("hadamard_order must divide head_dim")
    original_shape = x.shape
    y = x.reshape(-1, order).to(dtype=torch.float32)
    h = 1
    while h < order:
        y = y.reshape(-1, order // (h * 2), h * 2)
        a = y[..., :h].clone()
        b = y[..., h : 2 * h].clone()
        y[..., :h] = a + b
        y[..., h : 2 * h] = a - b
        h *= 2
        y = y.reshape(-1, order)
    y = y.reshape(original_shape) * (order ** -0.5)
    return y.to(dtype=x.dtype)


def pack_int4_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x32 = x.to(dtype=torch.float32)
    xmin = torch.amin(x32, dim=-1)
    xmax = torch.amax(x32, dim=-1)
    scale = ((xmax - xmin).clamp_min(1e-6) / 15.0).to(dtype=torch.float16)
    zero = (-xmin / scale.to(dtype=torch.float32)).to(dtype=torch.float16)
    q = torch.round(x32 / scale.to(dtype=torch.float32).unsqueeze(-1) + zero.to(dtype=torch.float32).unsqueeze(-1))
    q = q.clamp(0, 15).to(torch.int16)
    half = x.size(-1) // 2
    low = q[..., :half]
    high = q[..., half:]
    packed = (low | (high << 4)).to(torch.uint8)
    return packed, scale, zero


def unpack_int4_rows(packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    vals = torch.empty((*packed.shape[:-1], head_dim), device=packed.device, dtype=torch.float16)
    low = (packed & 0x0F).to(torch.float16)
    high = ((packed >> 4) & 0x0F).to(torch.float16)
    vals[..., : packed.size(-1)] = low
    vals[..., packed.size(-1) :] = high
    return (vals - zero.unsqueeze(-1)) * scale.unsqueeze(-1)
