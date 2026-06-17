from __future__ import annotations

from dataclasses import dataclass, replace
import os
from types import SimpleNamespace
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F

from .config import Qwen36_27B_TextConfig
from ...loader import QuantizedStore, LowBitTensor, LowBitMarlinTensor, FP16Tensor
from ...core.kv_cache import hadamard_transform, pack_int4_rows, unpack_int4_rows
from ...ops import cuda_ops
from ...engines.native.cuda_graph import CudaGraphKey, CudaGraphReplayCache
from .state import DecodeState
from ...speculative_batch import DecodeBatchPlan, resolve_speculative_gpu
from ...tuning import (
    attention_recent_tokens,
    batch_conv_kernels_enabled,
    batch_gdn_kernels_enabled,
    batch_prefill_steps_enabled,
    fast_raw_block_enabled,
    lowbit_rows_per_cta,
    paged_attention_backend,
    paged_attention_kernels_enabled,
    paged_prefill_block_enabled,
    prefill_attention_policy,
    verify_nextn_mode,
)

TensorLike = LowBitTensor | LowBitMarlinTensor | FP16Tensor


def _paged_int4_tiled_layout(arena: object) -> bool:
    return bool(getattr(arena, "paged_int4_tiled_layout", False))


def _copy_paged_int4_rows(
    page: torch.Tensor,
    block: int,
    offset: int,
    rows: torch.Tensor,
    *,
    tiled_layout: bool,
) -> None:
    if tiled_layout:
        page[block, :, :, offset].copy_(rows)
    else:
        page[block, :, offset, :].copy_(rows)


def _read_paged_int4_rows(
    page: torch.Tensor,
    block: int,
    offset: int,
    *,
    tiled_layout: bool,
) -> torch.Tensor:
    if tiled_layout:
        return page[block, :, :, offset]
    return page[block, :, offset, :]


@dataclass
class BlockForwardResult:
    logits: list[torch.Tensor]
    hidden_taps: list[list[torch.Tensor]]
    state: DecodeState
    raw_hiddens: list[torch.Tensor]
    final_hiddens: list[torch.Tensor] | None = None


@dataclass
class VerifyBlockResult:
    target_ids: torch.Tensor
    logits: torch.Tensor
    hidden: torch.Tensor
    state: DecodeState
    state_already_committed: bool = False
    speculative_decision: object | None = None


class WeightResolver:
    """Tolerant resolver for Qwen3.6/Qwen3.5 text checkpoint names."""

    PREFIXES = ("", "language_model.", "model.language_model.", "llm.", "model.llm.")

    def __init__(self, store: QuantizedStore):
        self.store = store
        self.names = set(store.index["tensors"].keys())

    @classmethod
    def expand_candidates(cls, candidates: Sequence[str]) -> list[str]:
        out: list[str] = []
        for c in candidates:
            expanded = [c]
            if c.startswith("model."):
                expanded.append("model.language_model." + c[len("model.") :])
                expanded.append("language_model." + c)
            elif c.startswith("layers."):
                expanded.append("model.layers." + c[len("layers.") :])
                expanded.append("model.language_model.layers." + c[len("layers.") :])
            if c.startswith("lm_head."):
                expanded.append("model.lm_head." + c[len("lm_head.") :])
                expanded.append("language_model.lm_head." + c[len("lm_head.") :])
                expanded.append("model.language_model.lm_head." + c[len("lm_head.") :])
            for p in cls.PREFIXES:
                for e in expanded:
                    out.append(p + e)
        return list(dict.fromkeys(out))

    def candidates_with_prefixes(self, candidates: Sequence[str]) -> list[str]:
        return self.expand_candidates(candidates)

    def get(self, *candidates: str) -> TensorLike:
        for name in self.candidates_with_prefixes(candidates):
            if name in self.names:
                return self.store.tensor(name)
        raise KeyError("missing tensor; tried: " + ", ".join(self.candidates_with_prefixes(candidates)))

    def optional(self, *candidates: str) -> TensorLike | None:
        for name in self.candidates_with_prefixes(candidates):
            if name in self.names:
                return self.store.tensor(name)
        return None

    def linear(self, *candidates: str) -> LowBitTensor:
        t = self.get(*candidates)
        if not isinstance(t, LowBitTensor):
            raise TypeError(f"expected LowBitTensor for {candidates}, got {type(t).__name__}")
        return t

    def any_linear(self, *candidates: str) -> TensorLike:
        t = self.get(*candidates)
        if not isinstance(t, (LowBitTensor, LowBitMarlinTensor, FP16Tensor)):
            raise TypeError(f"expected tensor for {candidates}, got {type(t).__name__}")
        return t

    def fp16(self, *candidates: str) -> torch.Tensor:
        t = self.get(*candidates)
        if not isinstance(t, FP16Tensor):
            raise TypeError(f"expected FP16Tensor for {candidates}, got {type(t).__name__}")
        return t.value

    def optional_fp16(self, *candidates: str) -> torch.Tensor | None:
        t = self.optional(*candidates)
        if t is None:
            return None
        if not isinstance(t, FP16Tensor):
            raise TypeError(f"expected FP16Tensor for {candidates}, got {type(t).__name__}")
        return t.value


def lowbit_linear(w: LowBitTensor | LowBitMarlinTensor, x: torch.Tensor) -> torch.Tensor:
    return w.gemv(x.contiguous())


def lowbit_linear_on_device(w: LowBitTensor | LowBitMarlinTensor, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    device = torch.device(device)
    if w.qweight.device.type == "cuda" and device.type == "cuda":
        return lowbit_linear(w, x)
    if isinstance(w, LowBitMarlinTensor):
        raise RuntimeError("Marlin tensors must be loaded on the target CUDA device")
    q = w.qweight.to(device, non_blocking=True).contiguous()
    s = w.scales.to(device, non_blocking=True).contiguous()
    tmp = LowBitTensor(name=w.name, qweight=q, scales=s, cols=w.cols, group_size=w.group_size, bits=w.bits)
    return lowbit_linear(tmp, x)


def lowbit_linear_pair_on_device(a: LowBitTensor, b: LowBitTensor, x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        a.qweight.device.type == "cuda"
        and b.qweight.device.type == "cuda"
        and a.qweight.device == device
        and b.qweight.device == device
        and a.cols == b.cols
        and a.group_size == b.group_size
        and a.bits == b.bits
        and a.qweight.shape == b.qweight.shape
        and a.scales.shape == b.scales.shape
    ):
        outs = cuda_ops().lowbit_gemv_pair(
            a.qweight,
            a.scales,
            b.qweight,
            b.scales,
            x.contiguous(),
            a.cols,
            a.group_size,
            a.bits,
            lowbit_rows_per_cta(),
        )
        return outs[0], outs[1]
    return lowbit_linear_on_device(a, x, device), lowbit_linear_on_device(b, x, device)


def linear_any(w: TensorLike, x: torch.Tensor) -> torch.Tensor:
    if isinstance(w, (LowBitTensor, LowBitMarlinTensor)):
        if x.ndim == 2:
            return w.gemm(x.to(device=x.device, dtype=torch.float16).contiguous())
        return lowbit_linear_on_device(w, x, x.device)
    if x.ndim == 2:
        return torch.matmul(x.to(device=x.device, dtype=w.value.dtype), w.value.to(device=x.device).t())
    return torch.matmul(w.value.to(device=x.device, dtype=x.dtype), x)


def tensor_rows(w: TensorLike) -> int:
    if isinstance(w, LowBitMarlinTensor):
        return int(w.qweight.size(1) // 2)
    if isinstance(w, LowBitTensor):
        return int(w.qweight.size(0))
    return int(w.value.size(0))


def embed_lookup(w: TensorLike, token: torch.Tensor) -> torch.Tensor:
    if isinstance(w, LowBitTensor):
        return w.row_dequant(token)
    return w.value[token.long()].reshape(-1)


def embed_lookup_batch(w: TensorLike, tokens: torch.Tensor, device: torch.device) -> torch.Tensor:
    tokens = tokens.to(device=device, dtype=torch.long).reshape(-1)
    if isinstance(w, FP16Tensor):
        return w.value.to(device=device)[tokens].contiguous()
    return torch.stack([embed_lookup(w, token).to(device=device, non_blocking=True) for token in tokens], dim=0).contiguous()


def qwen_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return cuda_ops().rmsnorm_qwen(x.contiguous(), weight.to(device=x.device, dtype=x.dtype).contiguous(), eps)


def qwen_rmsnorm_lastdim(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Apply Qwen RMSNorm over the last dimension using the common op path."""
    shape = x.shape
    hidden = shape[-1]
    y = cuda_ops().rmsnorm_qwen(
        x.reshape(-1, hidden).contiguous(),
        weight.to(device=x.device, dtype=x.dtype).contiguous(),
        eps,
    )
    return y.reshape(shape).contiguous()


def qwen_gdn_norm_silu_gate(core: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    weight = weight.to(device=core.device, dtype=core.dtype).contiguous()
    z = z.to(device=core.device, dtype=core.dtype).contiguous()
    if core.ndim == 3:
        if weight.numel() == core.size(-1):
            rows = core.size(0) * core.size(1)
            hidden = core.size(2)
        elif weight.numel() == core.size(1) * core.size(2):
            rows = core.size(0)
            hidden = core.size(1) * core.size(2)
        else:
            raise RuntimeError("weight hidden mismatch")
        return cuda_ops().rmsnorm_silu_gate(
            core.reshape(rows, hidden).contiguous(),
            weight.reshape(-1).contiguous(),
            z.reshape(rows, hidden).contiguous(),
            eps,
        ).reshape_as(core)
    if core.ndim == 2 and weight.numel() == core.numel():
        return cuda_ops().rmsnorm_silu_gate(
            core.reshape(1, -1).contiguous(),
            weight.reshape(-1).contiguous(),
            z.reshape(1, -1).contiguous(),
            eps,
        ).reshape_as(core)
    if weight.numel() == core.numel():
        return cuda_ops().rmsnorm_silu_gate(
            core.reshape(-1).contiguous(),
            weight.reshape(-1).contiguous(),
            z.reshape(-1).contiguous(),
            eps,
        )
    core32 = core.to(torch.float32)
    core_norm = core32 * torch.rsqrt(core32.pow(2).mean(dim=-1, keepdim=True) + eps)
    core_norm = (core_norm * weight.to(device=core.device, dtype=torch.float32)).to(core.dtype)
    return (core_norm.float() * F.silu(z.float())).to(core.dtype).reshape(-1).contiguous()


def gdn_norm_silu_gate_2d(core: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
    """Hot-path GDN norm+gate using the checkpoint weight width as hidden size."""
    hidden = int(weight.numel())
    if core.numel() % hidden != 0 or z.numel() != core.numel():
        raise RuntimeError("GDN norm hidden mismatch")
    return cuda_ops().rmsnorm_silu_gate(
        core.reshape(-1, hidden).contiguous(),
        weight,
        z.reshape(-1, hidden).contiguous(),
        eps,
    ).reshape_as(core)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if gate.device.type == "cuda" and gate.dtype == torch.float16 and up.dtype == torch.float16:
        return cuda_ops().silu_mul(gate.contiguous(), up.contiguous())
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def apply_rope_single_token(q: torch.Tensor, k: torch.Tensor, *, pos: int, rope_dim: int, rope_theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_dim <= 0:
        return q, k
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")
    if rope_dim > q.size(-1) or rope_dim > k.size(-1):
        raise ValueError("rope_dim cannot exceed q/k head_dim")
    device = q.device
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    freqs = float(pos) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(q.dtype)
    sin = emb.sin().to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        half = rope_dim // 2
        rotated = torch.cat((-x_rope[..., half:], x_rope[..., :half]), dim=-1)
        y = x_rope * cos + rotated * sin
        return torch.cat([y, x_pass], dim=-1) if x_pass.numel() else y

    return rotate(q), rotate(k)


def apply_rope_block(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    start_pos: int,
    rope_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_dim <= 0:
        return q, k
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")
    if rope_dim > q.size(-1) or rope_dim > k.size(-1):
        raise ValueError("rope_dim cannot exceed q/k head_dim")
    device = q.device
    positions = torch.arange(start_pos, start_pos + q.size(0), device=device, dtype=torch.float32)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    freqs = positions[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, None, :].to(q.dtype)
    sin = emb.sin()[:, None, :].to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        half = rope_dim // 2
        rotated = torch.cat((-x_rope[..., half:], x_rope[..., :half]), dim=-1)
        y = x_rope * cos + rotated * sin
        return torch.cat([y, x_pass], dim=-1) if x_pass.numel() else y

    return rotate(q).contiguous(), rotate(k).contiguous()


def apply_rope_decode_batch(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    positions: torch.Tensor,
    rope_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rope_dim <= 0:
        return q, k
    if rope_dim % 2 != 0:
        raise ValueError("rope_dim must be even")
    if rope_dim > q.size(-1) or rope_dim > k.size(-1):
        raise ValueError("rope_dim cannot exceed q/k head_dim")
    device = q.device
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    freqs = positions.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[:, None, :].to(q.dtype)
    sin = emb.sin()[:, None, :].to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_rope = x[..., :rope_dim]
        x_pass = x[..., rope_dim:]
        half = rope_dim // 2
        rotated = torch.cat((-x_rope[..., half:], x_rope[..., :half]), dim=-1)
        y = x_rope * cos + rotated * sin
        return torch.cat([y, x_pass], dim=-1) if x_pass.numel() else y

    return rotate(q).contiguous(), rotate(k).contiguous()


def attention_decode_any(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    length: int,
    scale: float,
) -> torch.Tensor:
    if q.device.type == "cuda" and length >= 1024:
        k_live = k_cache[:, :length, :].to(dtype=torch.float16)
        v_live = v_cache[:, :length, :].to(dtype=torch.float16)
        out = F.scaled_dot_product_attention(
            q.unsqueeze(0).unsqueeze(2).contiguous(),
            k_live.unsqueeze(0),
            v_live.unsqueeze(0),
            dropout_p=0.0,
            scale=scale,
            enable_gqa=(q.size(0) != k_live.size(0)),
        )
        return out.squeeze(0).squeeze(1).contiguous()
    if k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
        k_cache = k_cache[:, :length, :].to(dtype=torch.float16).contiguous()
        v_cache = v_cache[:, :length, :].to(dtype=torch.float16).contiguous()
        length = k_cache.size(1)
    return cuda_ops().attention_decode_fp16(q.contiguous(), k_cache, v_cache, length, scale)


def attention_decode_paged_batch(
    arena: object,
    layer: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: Any,
    scale: float,
) -> torch.Tensor:
    k_pages = getattr(arena, "paged_attn_k", None)
    v_pages = getattr(arena, "paged_attn_v", None)
    if k_pages is None or v_pages is None:
        raise RuntimeError("paged KV arena is not allocated")
    block_size = int(getattr(arena, "kv_block_size"))
    if block_size <= 0:
        raise RuntimeError("paged KV block_size is not configured")
    if getattr(plan, "slot_mapping", None) is None or getattr(plan, "block_tables", None) is None:
        raise RuntimeError("paged attention requires slot_mapping and block_tables")
    kv_spec = getattr(arena, "kv_cache_spec")
    slot_mapping = plan.slot_mapping.to(device=q.device, dtype=torch.long).contiguous()
    block_tables = plan.block_tables.to(device=q.device, dtype=torch.int32).contiguous()
    seq_lens = plan.seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    recent_tokens = attention_recent_tokens()
    if recent_tokens > 0 and int(seq_lens.numel()) > 0 and int(block_tables.size(1)) > 0:
        start_blocks = torch.div(
            torch.clamp(seq_lens - int(recent_tokens), min=0),
            int(block_size),
            rounding_mode="floor",
        ).to(dtype=torch.int64)
        cols = torch.arange(block_tables.size(1), device=block_tables.device, dtype=torch.int64).unsqueeze(0)
        gather_idx = cols + start_blocks.to(device=block_tables.device).unsqueeze(1)
        valid = gather_idx < int(block_tables.size(1))
        gather_idx = torch.clamp(gather_idx, max=max(0, int(block_tables.size(1)) - 1))
        shifted = torch.gather(block_tables, 1, gather_idx)
        block_tables = torch.where(valid, shifted, torch.zeros((), device=block_tables.device, dtype=block_tables.dtype)).contiguous()
        seq_lens = (seq_lens - start_blocks.to(device=seq_lens.device, dtype=seq_lens.dtype) * int(block_size)).contiguous()
    if not paged_attention_kernels_enabled():
        return attention_decode_paged_reference(
            arena,
            layer,
            q,
            k,
            v,
            slot_mapping,
            block_tables,
            seq_lens,
            block_size,
            scale,
            getattr(plan, "state_indices", None),
            getattr(plan, "positions", None),
        )
    if kv_spec.is_int4:
        k_scales = getattr(arena, "paged_attn_k_scale", None)
        v_scales = getattr(arena, "paged_attn_v_scale", None)
        k_zeros = getattr(arena, "paged_attn_k_zero", None)
        v_zeros = getattr(arena, "paged_attn_v_zero", None)
        if k_scales is None or v_scales is None or k_zeros is None or v_zeros is None:
            raise RuntimeError("INT4 paged attention requires paged scale/zero tensors")
        ops = cuda_ops()
        backend = paged_attention_backend()
        if backend == "auto":
            if hasattr(ops, "attention_paged_int4_flash"):
                op = ops.attention_paged_int4_flash
            else:
                op = ops.attention_decode_paged_int4
        elif backend == "direct":
            op = ops.attention_decode_paged_int4
        elif backend == "flash":
            if not hasattr(ops, "attention_paged_int4_flash"):
                raise RuntimeError("LANGBURST_PAGED_ATTENTION_BACKEND=flash requires attention_paged_int4_flash")
            op = ops.attention_paged_int4_flash
        else:  # pragma: no cover - paged_attention_backend validates this.
            raise ValueError(f"unknown paged attention backend: {backend}")
        out = op(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            k_pages[layer],
            v_pages[layer],
            k_scales[layer],
            v_scales[layer],
            k_zeros[layer],
            v_zeros[layer],
            slot_mapping,
            block_tables,
            seq_lens,
            block_size,
            scale,
            int(kv_spec.hadamard_order),
            bool(kv_spec.uses_bdr),
            bool(kv_spec.rotate_v),
            _paged_int4_tiled_layout(arena),
        )
    elif kv_spec.is_fp8:
        out = cuda_ops().attention_decode_paged_fp8_e4m3(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            k_pages[layer],
            v_pages[layer],
            slot_mapping,
            block_tables,
            seq_lens,
            block_size,
            scale,
            float(kv_spec.k_scale),
            float(kv_spec.v_scale),
        )
    else:
        out = cuda_ops().attention_decode_paged_fp16(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            k_pages[layer],
            v_pages[layer],
            slot_mapping,
            block_tables,
            seq_lens,
            block_size,
            scale,
        )
    # Keep the canonical arena KV view in sync with the paged hot path.  Paged
    # attention is the serving fast path, but snapshots, forks, fallback paths,
    # and parity checks still read DecodeState.attn_k/v.  A stale canonical view
    # causes continuation drift after an otherwise correct paged decode step.
    arena_k = getattr(arena, "attn_k", {}).get(layer)
    arena_v = getattr(arena, "attn_v", {}).get(layer)
    state_indices = getattr(plan, "state_indices", None)
    positions = getattr(plan, "positions", None)
    if (
        not kv_spec.is_int4
        and arena_k is not None
        and arena_v is not None
        and state_indices is not None
        and positions is not None
    ):
        slots = state_indices.to(device=q.device, dtype=torch.long).contiguous()
        max_seq = int(arena_k.size(2))
        if max_seq > 0:
            write_indices = torch.remainder(positions.to(device=q.device, dtype=torch.long), max_seq).contiguous()
            arena_k[slots, :, write_indices, :] = k.to(device=arena_k.device, dtype=arena_k.dtype)
            arena_v[slots, :, write_indices, :] = v.to(device=arena_v.device, dtype=arena_v.dtype)
    return out


def arena_has_canonical_attention_mirror(arena: object, layer: int) -> bool:
    mirror_k = getattr(arena, "attn_k", {}).get(layer)
    mirror_v = getattr(arena, "attn_v", {}).get(layer)
    return mirror_k is not None and mirror_v is not None and int(mirror_k.size(-2)) > 0


def write_paged_kv_row(
    arena: object,
    layer: int,
    row: int,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: Any | None,
) -> None:
    """Mirror a canonical attention KV write into paged storage.

    This is the state-trajectory parity bridge: the canonical DecodeState path
    remains the source of truth while paged KV pages stay warm for future hot
    kernels, prefix/state cache ownership, and block-table validation.
    """

    if plan is None:
        return
    k_pages = getattr(arena, "paged_attn_k", None)
    v_pages = getattr(arena, "paged_attn_v", None)
    if k_pages is None or v_pages is None:
        return
    slot_mapping = getattr(plan, "slot_mapping", None)
    if slot_mapping is None:
        return
    block_size = int(getattr(arena, "kv_block_size", 0))
    if block_size <= 0:
        return
    slot = int(slot_mapping.to(device=k.device, dtype=torch.long)[row].detach().cpu().item())
    block = slot // block_size
    offset = slot % block_size
    kv_spec = getattr(arena, "kv_cache_spec")
    if kv_spec.is_int4:
        k_store = hadamard_transform(k, kv_spec.hadamard_order) if kv_spec.uses_bdr else k
        v_store = hadamard_transform(v, kv_spec.hadamard_order) if kv_spec.uses_bdr and kv_spec.rotate_v else v
        k_packed, k_scale, k_zero = pack_int4_rows(k_store)
        v_packed, v_scale, v_zero = pack_int4_rows(v_store)
        tiled_layout = _paged_int4_tiled_layout(arena)
        _copy_paged_int4_rows(k_pages[layer], block, offset, k_packed, tiled_layout=tiled_layout)
        _copy_paged_int4_rows(v_pages[layer], block, offset, v_packed, tiled_layout=tiled_layout)
        getattr(arena, "paged_attn_k_scale")[layer][block, :, offset].copy_(k_scale)
        getattr(arena, "paged_attn_v_scale")[layer][block, :, offset].copy_(v_scale)
        getattr(arena, "paged_attn_k_zero")[layer][block, :, offset].copy_(k_zero)
        getattr(arena, "paged_attn_v_zero")[layer][block, :, offset].copy_(v_zero)
        return
    k_pages[layer][block, :, offset, :].copy_(k.to(device=k_pages[layer].device, dtype=k_pages[layer].dtype))
    v_pages[layer][block, :, offset, :].copy_(v.to(device=v_pages[layer].device, dtype=v_pages[layer].dtype))


def append_paged_int4_block(
    arena: object,
    layer: int,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    k_pages = getattr(arena, "paged_attn_k", None)
    v_pages = getattr(arena, "paged_attn_v", None)
    k_scales = getattr(arena, "paged_attn_k_scale", None)
    v_scales = getattr(arena, "paged_attn_v_scale", None)
    k_zeros = getattr(arena, "paged_attn_k_zero", None)
    v_zeros = getattr(arena, "paged_attn_v_zero", None)
    if any(x is None for x in (k_pages, v_pages, k_scales, v_scales, k_zeros, v_zeros)):
        raise RuntimeError("INT4 paged append requires paged KV scale/zero tensors")
    kv_spec = getattr(arena, "kv_cache_spec")
    cuda_ops().attention_append_paged_int4(
        k.contiguous(),
        v.contiguous(),
        k_pages[layer],
        v_pages[layer],
        k_scales[layer],
        v_scales[layer],
        k_zeros[layer],
        v_zeros[layer],
        slot_mapping.to(device=k.device, dtype=torch.long).contiguous(),
        int(getattr(arena, "kv_block_size")),
        int(kv_spec.hadamard_order),
        bool(kv_spec.uses_bdr),
        bool(kv_spec.rotate_v),
        _paged_int4_tiled_layout(arena),
    )


def append_paged_int4_spec_block(
    arena: object,
    layer: int,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    commit_tokens: torch.Tensor,
) -> None:
    k_pages = getattr(arena, "paged_attn_k", None)
    v_pages = getattr(arena, "paged_attn_v", None)
    k_scales = getattr(arena, "paged_attn_k_scale", None)
    v_scales = getattr(arena, "paged_attn_v_scale", None)
    k_zeros = getattr(arena, "paged_attn_k_zero", None)
    v_zeros = getattr(arena, "paged_attn_v_zero", None)
    if any(x is None for x in (k_pages, v_pages, k_scales, v_scales, k_zeros, v_zeros)):
        raise RuntimeError("INT4 paged speculative append requires paged KV scale/zero tensors")
    kv_spec = getattr(arena, "kv_cache_spec")
    cuda_ops().attention_append_paged_int4_spec(
        k.contiguous(),
        v.contiguous(),
        k_pages[layer],
        v_pages[layer],
        k_scales[layer],
        v_scales[layer],
        k_zeros[layer],
        v_zeros[layer],
        slot_mapping.to(device=k.device, dtype=torch.long).contiguous(),
        commit_tokens.to(device=k.device, dtype=torch.int32).contiguous(),
        int(getattr(arena, "kv_block_size")),
        int(kv_spec.hadamard_order),
        bool(kv_spec.uses_bdr),
        bool(kv_spec.rotate_v),
        _paged_int4_tiled_layout(arena),
    )


def _short_prefill_sdpa_has_free_vram(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    min_free_mib = prefill_attention_policy().min_free_mib
    if min_free_mib <= 0:
        return True
    free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    return free_bytes >= min_free_mib * 1024 * 1024


def prefill_sdpa_window(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    live_len: int,
    query_start: int,
    query_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recent = attention_recent_tokens()
    start = max(0, int(query_start) - int(recent) + 1) if recent > 0 else 0
    k_live = k_cache[:, start:int(live_len), :]
    v_live = v_cache[:, start:int(live_len), :]
    key_pos = torch.arange(start, int(live_len), device=k_cache.device, dtype=torch.long)
    query_pos = torch.arange(int(query_start), int(query_start) + int(query_len), device=k_cache.device, dtype=torch.long)
    causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
    if recent > 0:
        causal = causal & (key_pos.unsqueeze(0) >= (query_pos.unsqueeze(1) - int(recent) + 1))
    return k_live, v_live, causal


def attention_decode_paged_reference(
    arena: object,
    layer: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    scale: float,
    state_indices: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Correct paged-KV attention path used as the parity baseline.

    The CUDA paged-attention kernel is a hot path and must be gated by full
    state-trajectory parity. This reference path still uses paged storage and
    block tables, but performs the readback with explicit tensor indexing so
    prefill/decode semantics match the canonical state path.
    """

    kv_spec = getattr(arena, "kv_cache_spec")
    k_pages = getattr(arena, "paged_attn_k")[layer]
    v_pages = getattr(arena, "paged_attn_v")[layer]
    mirror_k = getattr(arena, "attn_k", {}).get(layer)
    mirror_v = getattr(arena, "attn_v", {}).get(layer)
    mirror_enabled = mirror_k is not None and mirror_v is not None and int(mirror_k.size(-2)) > 0
    if state_indices is not None:
        state_indices = state_indices.to(device=q.device, dtype=torch.long).contiguous()
    if positions is not None:
        positions = positions.to(device=q.device, dtype=torch.long).contiguous()
    out_rows: list[torch.Tensor] = []
    for row in range(q.size(0)):
        slot = int(slot_mapping[row].detach().cpu().item())
        block = slot // int(block_size)
        offset = slot % int(block_size)
        logical_pos = int(positions[row].detach().cpu().item()) if positions is not None else int(seq_lens[row].detach().cpu().item()) - 1
        arena_slot = int(state_indices[row].detach().cpu().item()) if state_indices is not None else 0
        if kv_spec.is_int4:
            k_store = hadamard_transform(k[row], kv_spec.hadamard_order) if kv_spec.uses_bdr else k[row]
            v_store = (
                hadamard_transform(v[row], kv_spec.hadamard_order)
                if kv_spec.uses_bdr and kv_spec.rotate_v
                else v[row]
            )
            k_packed, k_scale, k_zero = pack_int4_rows(k_store)
            v_packed, v_scale, v_zero = pack_int4_rows(v_store)
            tiled_layout = _paged_int4_tiled_layout(arena)
            _copy_paged_int4_rows(k_pages, block, offset, k_packed, tiled_layout=tiled_layout)
            _copy_paged_int4_rows(v_pages, block, offset, v_packed, tiled_layout=tiled_layout)
            getattr(arena, "paged_attn_k_scale")[layer][block, :, offset].copy_(k_scale)
            getattr(arena, "paged_attn_v_scale")[layer][block, :, offset].copy_(v_scale)
            getattr(arena, "paged_attn_k_zero")[layer][block, :, offset].copy_(k_zero)
            getattr(arena, "paged_attn_v_zero")[layer][block, :, offset].copy_(v_zero)
        else:
            k_pages[block, :, offset, :].copy_(k[row].to(device=k_pages.device, dtype=k_pages.dtype))
            v_pages[block, :, offset, :].copy_(v[row].to(device=v_pages.device, dtype=v_pages.dtype))

        live = int(seq_lens[row].detach().cpu().item())
        if mirror_enabled and not kv_spec.is_int4:
            max_seq = int(mirror_k.size(-2))
            write_idx = logical_pos % max_seq if max_seq > 0 else 0
            mirror_k[arena_slot, :, write_idx, :].copy_(k[row].to(device=mirror_k.device, dtype=mirror_k.dtype))
            mirror_v[arena_slot, :, write_idx, :].copy_(v[row].to(device=mirror_v.device, dtype=mirror_v.dtype))
            if live <= max_seq or logical_pos < max_seq:
                k_cache = mirror_k[arena_slot, :, :live, :].contiguous()
                v_cache = mirror_v[arena_slot, :, :live, :].contiguous()
            else:
                start = (logical_pos + 1) % max_seq
                if start == 0:
                    k_cache = mirror_k[arena_slot].contiguous()
                    v_cache = mirror_v[arena_slot].contiguous()
                else:
                    k_cache = torch.cat([mirror_k[arena_slot, :, start:, :], mirror_k[arena_slot, :, :start, :]], dim=1).contiguous()
                    v_cache = torch.cat([mirror_v[arena_slot, :, start:, :], mirror_v[arena_slot, :, :start, :]], dim=1).contiguous()
                live = max_seq
            out_rows.append(attention_decode_any(q[row].contiguous(), k_cache, v_cache, live, scale))
            continue

        k_rows: list[torch.Tensor] = []
        v_rows: list[torch.Tensor] = []
        k_scale_rows: list[torch.Tensor] = []
        v_scale_rows: list[torch.Tensor] = []
        k_zero_rows: list[torch.Tensor] = []
        v_zero_rows: list[torch.Tensor] = []
        row_blocks = block_tables[row]
        tiled_layout = _paged_int4_tiled_layout(arena)
        for pos in range(live):
            block_index, block_offset = divmod(pos, int(block_size))
            block_id = int(row_blocks[block_index].detach().cpu().item())
            k_rows.append(_read_paged_int4_rows(k_pages, block_id, block_offset, tiled_layout=tiled_layout))
            v_rows.append(_read_paged_int4_rows(v_pages, block_id, block_offset, tiled_layout=tiled_layout))
            if kv_spec.is_int4:
                k_scale_rows.append(getattr(arena, "paged_attn_k_scale")[layer][block_id, :, block_offset])
                v_scale_rows.append(getattr(arena, "paged_attn_v_scale")[layer][block_id, :, block_offset])
                k_zero_rows.append(getattr(arena, "paged_attn_k_zero")[layer][block_id, :, block_offset])
                v_zero_rows.append(getattr(arena, "paged_attn_v_zero")[layer][block_id, :, block_offset])
        k_cache = torch.stack(k_rows, dim=1).contiguous()
        v_cache = torch.stack(v_rows, dim=1).contiguous()
        if kv_spec.is_int4:
            k_scale = torch.stack(k_scale_rows, dim=1).contiguous()
            v_scale = torch.stack(v_scale_rows, dim=1).contiguous()
            k_zero = torch.stack(k_zero_rows, dim=1).contiguous()
            v_zero = torch.stack(v_zero_rows, dim=1).contiguous()
            k_cache = unpack_int4_rows(k_cache, k_scale, k_zero, head_dim=k[row].size(-1))
            v_cache = unpack_int4_rows(v_cache, v_scale, v_zero, head_dim=v[row].size(-1))
            if kv_spec.uses_bdr:
                k_cache = hadamard_transform(k_cache, kv_spec.hadamard_order)
                if kv_spec.rotate_v:
                    v_cache = hadamard_transform(v_cache, kv_spec.hadamard_order)
        elif k_cache.dtype != torch.float16 or v_cache.dtype != torch.float16:
            k_cache = k_cache.to(dtype=torch.float16)
            v_cache = v_cache.to(dtype=torch.float16)
        out_rows.append(attention_decode_any(q[row].contiguous(), k_cache.contiguous(), v_cache.contiguous(), live, scale))
    return torch.stack(out_rows, dim=0).contiguous()


def sync_state_kv_to_paged(
    state: DecodeState,
    plan: Any,
    row: int,
) -> None:
    """Populate paged KV pages from the canonical per-state ring KV buffers.

    Chunked prefill still uses the sequential/block state path for correctness.
    The decode hot path consumes paged KV, so prefill must publish the canonical
    KV rows into the request's block table before the row enters decode.
    """

    arena = getattr(state, "arena", None)
    if arena is None:
        return
    paged_k = getattr(arena, "paged_attn_k", None)
    paged_v = getattr(arena, "paged_attn_v", None)
    block_tables = getattr(plan, "block_tables", None)
    if paged_k is None or paged_v is None or block_tables is None:
        return
    if not state.attn_k or next(iter(state.attn_k.values())).size(1) == 0:
        return
    block_size = int(getattr(arena, "kv_block_size", 0))
    if block_size <= 0 or state.kv_len <= 0:
        return
    row_blocks = block_tables[row].to(device=state.device, dtype=torch.long)
    logical_start = max(0, int(state.pos) - int(state.kv_len))
    logical_end = int(state.pos)
    max_blocks = int(row_blocks.numel())
    for logical_pos in range(logical_start, logical_end):
        block_index, offset = divmod(logical_pos, block_size)
        if block_index >= max_blocks:
            break
        block_id = int(row_blocks[block_index].detach().cpu().item())
        if block_id < 0:
            continue
        if state.kv_window_policy == "ring":
            src = logical_pos % state.max_seq_len
        elif logical_pos < state.max_seq_len:
            src = logical_pos
        else:
            src = state.max_seq_len - 1
        tiled_layout = _paged_int4_tiled_layout(arena)
        for layer in state.cfg.attention_layers:
            layer_k = state.attn_k.get(layer)
            layer_v = state.attn_v.get(layer)
            if layer_k is None or layer_v is None or layer_k.size(1) == 0 or layer_v.size(1) == 0:
                continue
            if state.kv_cache_spec.is_int4 and tiled_layout:
                paged_k[layer][block_id, :, :, offset].copy_(layer_k[:, src, :])
                paged_v[layer][block_id, :, :, offset].copy_(layer_v[:, src, :])
            else:
                paged_k[layer][block_id, :, offset, :].copy_(layer_k[:, src, :])
                paged_v[layer][block_id, :, offset, :].copy_(layer_v[:, src, :])
            if (
                state.attn_k_scale is not None
                and state.attn_v_scale is not None
                and getattr(arena, "paged_attn_k_scale", None) is not None
                and getattr(arena, "paged_attn_v_scale", None) is not None
                and layer in state.attn_k_scale
                and layer in state.attn_v_scale
                and state.attn_k_scale[layer].size(1) > 0
                and state.attn_v_scale[layer].size(1) > 0
            ):
                arena.paged_attn_k_scale[layer][block_id, :, offset].copy_(state.attn_k_scale[layer][:, src])
                arena.paged_attn_v_scale[layer][block_id, :, offset].copy_(state.attn_v_scale[layer][:, src])
            if (
                state.attn_k_zero is not None
                and state.attn_v_zero is not None
                and getattr(arena, "paged_attn_k_zero", None) is not None
                and getattr(arena, "paged_attn_v_zero", None) is not None
                and layer in state.attn_k_zero
                and layer in state.attn_v_zero
                and state.attn_k_zero[layer].size(1) > 0
                and state.attn_v_zero[layer].size(1) > 0
            ):
                arena.paged_attn_k_zero[layer][block_id, :, offset].copy_(state.attn_k_zero[layer][:, src])
                arena.paged_attn_v_zero[layer][block_id, :, offset].copy_(state.attn_v_zero[layer][:, src])


def has_canonical_attention_kv(state: DecodeState) -> bool:
    attn_k = getattr(state, "attn_k", None)
    if not attn_k:
        return False
    sample = next(iter(attn_k.values()), None)
    return sample is not None and sample.size(1) > 0


def split_gdn_qkv(cfg: Qwen36_27B_TextConfig, mixed_qkv: torch.Tensor):
    """Split Qwen3.5/Qwen3.6 split GDN qkv projection for one token.

    HF layout is flat [all_q, all_k, all_v], not per-head interleaved.
    """
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=0)
    return (
        q.view(cfg.linear_num_key_heads, cfg.linear_key_head_dim).contiguous(),
        k.view(cfg.linear_num_key_heads, cfg.linear_key_head_dim).contiguous(),
        v.view(cfg.linear_num_value_heads, cfg.linear_value_head_dim).contiguous(),
    )


def split_gdn_fused_qkvz(cfg: Qwen36_27B_TextConfig, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor):
    key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
    value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
    q, k, v, z = torch.split(mixed_qkvz, [key_dim, key_dim, value_dim, value_dim], dim=0)
    b, a = torch.split(mixed_ba, [cfg.linear_num_value_heads, cfg.linear_num_value_heads], dim=0)
    return (
        q.view(cfg.linear_num_key_heads, cfg.linear_key_head_dim).contiguous(),
        k.view(cfg.linear_num_key_heads, cfg.linear_key_head_dim).contiguous(),
        v.view(cfg.linear_num_value_heads, cfg.linear_value_head_dim).contiguous(),
        z.view(cfg.linear_num_value_heads, cfg.linear_value_head_dim).contiguous(),
        b.reshape(cfg.linear_num_value_heads).contiguous(),
        a.reshape(cfg.linear_num_value_heads).contiguous(),
    )


def depthwise_conv_update(buf: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    if buf.device.type == "cuda":
        empty_bias = torch.empty((0,), device=buf.device, dtype=buf.dtype)
        return cuda_ops().depthwise_conv_update(
            buf,
            x.to(device=buf.device, dtype=buf.dtype).contiguous(),
            weight.to(device=buf.device, dtype=buf.dtype).contiguous(),
            (bias.to(device=buf.device, dtype=buf.dtype).contiguous() if bias is not None else empty_bias),
        )
    if weight.ndim == 3:
        w = weight[:, 0, :]
    else:
        w = weight
    window = torch.cat([buf, x[:, None]], dim=1)
    y = (window * w.to(device=x.device, dtype=x.dtype)).sum(dim=1)
    if bias is not None:
        y = y + bias.to(device=x.device, dtype=x.dtype)
    if buf.numel() > 0:
        buf[:, :-1] = buf[:, 1:].clone()
        buf[:, -1] = x
    return F.silu(y)


def depthwise_conv_update_block(buf: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("block conv input must be [tokens, channels]")
    if buf.device.type == "cuda":
        empty_bias = torch.empty((0,), device=buf.device, dtype=buf.dtype)
        return cuda_ops().depthwise_conv_update_scan(
            buf,
            x.to(device=buf.device, dtype=buf.dtype).contiguous(),
            weight.to(device=buf.device, dtype=buf.dtype).contiguous(),
            (bias.to(device=buf.device, dtype=buf.dtype).contiguous() if bias is not None else empty_bias),
        )
    out = []
    for row in x:
        out.append(depthwise_conv_update(buf, row.contiguous(), weight, bias))
    return torch.stack(out, dim=0).contiguous()


def _arena_batch(states: Sequence[DecodeState]) -> tuple[object, torch.Tensor] | None:
    if not states:
        return None
    arena = getattr(states[0], "arena", None)
    if arena is None:
        return None
    slots: list[int] = []
    for state in states:
        if getattr(state, "arena", None) is not arena:
            return None
        slot = getattr(state, "arena_slot", None)
        if slot is None:
            return None
        slots.append(int(slot))
    device = next(iter(states[0].gdn_states.values())).device
    return arena, torch.tensor(slots, device=device, dtype=torch.long)


def _arena_positions(arena: object, state_indices: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    pos = getattr(arena, "pos", None)
    if not torch.is_tensor(pos):
        raise RuntimeError("arena-backed hot path requires device-side position counters")
    return pos.index_select(0, state_indices.to(device=pos.device, dtype=torch.long).reshape(-1)).to(
        device=device,
        dtype=torch.long,
    )


def depthwise_conv_update_batch(
    arena: object,
    layer: int,
    state_indices: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    state_arena = getattr(arena, "gdn_conv_states")[layer]
    if state_arena.device.type == "cuda":
        empty_bias = torch.empty((0,), device=state_arena.device, dtype=state_arena.dtype)
        return cuda_ops().depthwise_conv_update_batch(
            state_arena,
            state_indices.to(device=state_arena.device, dtype=torch.long).contiguous(),
            x.to(device=state_arena.device, dtype=state_arena.dtype).contiguous(),
            weight.to(device=state_arena.device, dtype=state_arena.dtype).contiguous(),
            (bias.to(device=state_arena.device, dtype=state_arena.dtype).contiguous() if bias is not None else empty_bias),
        )
    empty_bias = torch.empty((0,), device=state_arena.device, dtype=state_arena.dtype)
    return cuda_ops().depthwise_conv_update_batch(
        state_arena,
        state_indices.to(device=state_arena.device, dtype=torch.long).contiguous(),
        x.to(device=state_arena.device, dtype=state_arena.dtype).contiguous(),
        weight.to(device=state_arena.device, dtype=state_arena.dtype).contiguous(),
        (bias.to(device=state_arena.device, dtype=state_arena.dtype).contiguous() if bias is not None else empty_bias),
    )


def depthwise_conv_update_spec(
    arena: object,
    layer: int,
    state_indices: torch.Tensor,
    commit_tokens: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    state_arena = getattr(arena, "gdn_conv_states")[layer]
    empty_bias = torch.empty((0,), device=state_arena.device, dtype=state_arena.dtype)
    return cuda_ops().depthwise_conv_update_spec(
        state_arena,
        state_indices.to(device=state_arena.device, dtype=torch.long).contiguous(),
        commit_tokens.to(device=state_arena.device, dtype=torch.int32).contiguous(),
        x.to(device=state_arena.device, dtype=state_arena.dtype).contiguous(),
        weight.to(device=state_arena.device, dtype=state_arena.dtype).contiguous(),
        (bias.to(device=state_arena.device, dtype=state_arena.dtype).contiguous() if bias is not None else empty_bias),
    )


def depthwise_conv_update_spec_trajectory(
    arena: object,
    layer: int,
    state_indices: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out: torch.Tensor | None = None,
    trajectory: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    state_arena = getattr(arena, "gdn_conv_states")[layer]
    empty_bias = torch.empty((0,), device=state_arena.device, dtype=state_arena.dtype)
    x_in = x.to(device=state_arena.device, dtype=state_arena.dtype).contiguous()
    weight_in = weight.to(device=state_arena.device, dtype=state_arena.dtype).contiguous()
    bias_in = bias.to(device=state_arena.device, dtype=state_arena.dtype).contiguous() if bias is not None else empty_bias
    if out is None or trajectory is None:
        out, trajectory = cuda_ops().depthwise_conv_update_spec_trajectory(
            state_arena,
            state_indices.to(device=state_arena.device, dtype=torch.long).contiguous(),
            x_in,
            weight_in,
            bias_in,
        )
    else:
        cuda_ops().depthwise_conv_update_spec_trajectory_out(
            state_arena,
            state_indices.to(device=state_arena.device, dtype=torch.long).contiguous(),
            x_in,
            weight_in,
            bias_in,
            out,
            trajectory,
        )
    return out, trajectory


class Qwen36MLP:
    def __init__(self, cfg: Qwen36_27B_TextConfig, w: WeightResolver, layer: int | None = None, *, prefix: str | None = None):
        if prefix is None:
            if layer is None:
                raise ValueError("either layer or prefix is required")
            prefix = f"model.layers.{layer}"
        p = f"{prefix}.mlp"
        self.gate_up = w.optional(f"{p}.gate_up_proj.weight")
        self.gate = None if self.gate_up is not None else w.any_linear(f"{p}.gate_proj.weight")
        self.up = None if self.gate_up is not None else w.any_linear(f"{p}.up_proj.weight")
        self.down = w.any_linear(f"{p}.down_proj.weight")
        self.intermediate_size = tensor_rows(self.gate_up) // 2 if self.gate_up is not None else tensor_rows(self.gate)  # type: ignore[arg-type]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up is not None:
            mixed = linear_any(self.gate_up, x)
            gate, up = torch.split(mixed, [self.intermediate_size, self.intermediate_size], dim=-1)
        elif isinstance(self.gate, LowBitTensor) and isinstance(self.up, LowBitTensor):
            gate, up = lowbit_linear_pair_on_device(self.gate, self.up, x, x.device)
        else:
            gate = linear_any(self.gate, x)
            up = linear_any(self.up, x)
        return linear_any(self.down, silu_mul(gate, up))

    def forward_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up is not None:
            mixed = linear_any(self.gate_up, x)
            gate, up = torch.split(mixed, [self.intermediate_size, self.intermediate_size], dim=-1)
        else:
            gate = linear_any(self.gate, x)
            up = linear_any(self.up, x)
        return linear_any(self.down, silu_mul(gate, up))


class Qwen36GDNLayer:
    def __init__(self, cfg: Qwen36_27B_TextConfig, weights: WeightResolver, layer: int, device: torch.device):
        self.cfg = cfg
        self.layer = layer
        p = f"model.layers.{layer}"
        la_candidates = [f"{p}.linear_attn", f"{p}.linear_attention", f"{p}.self_attn"]
        self.input_norm = weights.fp16(f"{p}.input_layernorm.weight", f"{p}.input_norm.weight")
        self.post_norm = weights.fp16(f"{p}.post_attention_layernorm.weight", f"{p}.post_norm.weight")

        # Two supported checkpoint layouts:
        # 1) Qwen3-Next fused in_proj_qkvz/in_proj_ba.
        # 2) User's Qwen3.6/Qwen3.5 split in_proj_qkv/in_proj_z/in_proj_a/in_proj_b.
        self.in_qkvz = weights.optional(*(f"{la}.in_proj_qkvz.weight" for la in la_candidates))
        self.in_ba = weights.optional(*(f"{la}.in_proj_ba.weight" for la in la_candidates))
        self.in_qkv = weights.optional(*(f"{la}.in_proj_qkv.weight" for la in la_candidates))
        self.in_z = weights.optional(*(f"{la}.in_proj_z.weight" for la in la_candidates))
        self.in_a = weights.optional(*(f"{la}.in_proj_a.weight" for la in la_candidates))
        self.in_b = weights.optional(*(f"{la}.in_proj_b.weight" for la in la_candidates))
        if self.in_qkvz is None and self.in_qkv is None:
            raise KeyError(f"layer {layer}: missing GDN projections")
        if self.in_qkvz is not None and self.in_ba is None and not all(t is not None for t in (self.in_a, self.in_b)):
            raise KeyError(f"layer {layer}: fused qkvz requires in_proj_ba or in_proj_a/in_proj_b")
        if self.in_qkv is not None and not all(t is not None for t in (self.in_z, self.in_a, self.in_b)):
            raise KeyError(f"layer {layer}: split qkv requires in_proj_z/in_proj_a/in_proj_b")

        self.out_proj = weights.any_linear(*(f"{la}.out_proj.weight" for la in la_candidates))
        self.gdn_norm_w = weights.fp16(*(f"{la}.norm.weight" for la in la_candidates))
        self.conv_weight = weights.fp16(*(f"{la}.conv1d.weight" for la in la_candidates), *(f"{la}.conv.weight" for la in la_candidates))
        self.conv_bias = weights.optional_fp16(*(f"{la}.conv1d.bias" for la in la_candidates), *(f"{la}.conv.bias" for la in la_candidates))
        self.A_log = weights.fp16(*(f"{la}.A_log" for la in la_candidates)).to(device=device, dtype=torch.float32).contiguous()
        self.dt_bias = weights.fp16(*(f"{la}.dt_bias" for la in la_candidates)).to(device=device, dtype=torch.float32).contiguous()
        self.mlp = Qwen36MLP(cfg, weights, layer)

    def project(self, x: torch.Tensor):
        if self.in_qkvz is not None:
            mixed_qkvz = linear_any(self.in_qkvz, x)
            if self.in_ba is not None:
                mixed_ba = linear_any(self.in_ba, x)
            elif isinstance(self.in_a, LowBitTensor) and isinstance(self.in_b, LowBitTensor):
                a, b = lowbit_linear_pair_on_device(self.in_a, self.in_b, x, x.device)
                mixed_ba = torch.cat([b.reshape(-1), a.reshape(-1)], dim=0)
            else:
                b = linear_any(self.in_b, x)  # type: ignore[arg-type]
                a = linear_any(self.in_a, x)  # type: ignore[arg-type]
                mixed_ba = torch.cat([b.reshape(-1), a.reshape(-1)], dim=0)
            return split_gdn_fused_qkvz(self.cfg, mixed_qkvz, mixed_ba)
        q, k, v = split_gdn_qkv(self.cfg, linear_any(self.in_qkv, x))  # type: ignore[arg-type]
        z = linear_any(self.in_z, x).view(self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim)  # type: ignore[arg-type]
        if isinstance(self.in_a, LowBitTensor) and isinstance(self.in_b, LowBitTensor):
            a, b = lowbit_linear_pair_on_device(self.in_a, self.in_b, x, x.device)
        else:
            a = linear_any(self.in_a, x)  # type: ignore[arg-type]
            b = linear_any(self.in_b, x)  # type: ignore[arg-type]
        a = a.reshape(self.cfg.linear_num_value_heads)
        b = b.reshape(self.cfg.linear_num_value_heads)
        return q.contiguous(), k.contiguous(), v.contiguous(), z.contiguous(), b.contiguous(), a.contiguous()

    def project_block(self, x: torch.Tensor):
        t = x.size(0)
        key_dim = self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads
        value_dim = self.cfg.linear_value_head_dim * self.cfg.linear_num_value_heads
        if self.in_qkvz is not None:
            mixed_qkvz = linear_any(self.in_qkvz, x)
            if self.in_ba is not None:
                mixed_ba = linear_any(self.in_ba, x)
            elif x.size(0) == 1 and isinstance(self.in_a, LowBitTensor) and isinstance(self.in_b, LowBitTensor):
                a1, b1 = lowbit_linear_pair_on_device(self.in_a, self.in_b, x.reshape(-1), x.device)
                mixed_ba = torch.cat([b1.reshape(1, -1), a1.reshape(1, -1)], dim=-1)
            else:
                b = linear_any(self.in_b, x)  # type: ignore[arg-type]
                a = linear_any(self.in_a, x)  # type: ignore[arg-type]
                mixed_ba = torch.cat([b.reshape(x.size(0), -1), a.reshape(x.size(0), -1)], dim=-1)
            q, k, v, z = torch.split(mixed_qkvz, [key_dim, key_dim, value_dim, value_dim], dim=-1)
            b, a = torch.split(mixed_ba, [self.cfg.linear_num_value_heads, self.cfg.linear_num_value_heads], dim=-1)
        else:
            qkv = linear_any(self.in_qkv, x)  # type: ignore[arg-type]
            q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
            z = linear_any(self.in_z, x)  # type: ignore[arg-type]
            if x.size(0) == 1 and isinstance(self.in_a, LowBitTensor) and isinstance(self.in_b, LowBitTensor):
                a1, b1 = lowbit_linear_pair_on_device(self.in_a, self.in_b, x.reshape(-1), x.device)
                a = a1.reshape(1, -1)
                b = b1.reshape(1, -1)
            else:
                a = linear_any(self.in_a, x)  # type: ignore[arg-type]
                b = linear_any(self.in_b, x)  # type: ignore[arg-type]
        return (
            q.view(t, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous(),
            k.view(t, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous(),
            v.view(t, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous(),
            z.view(t, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous(),
            b.view(t, self.cfg.linear_num_value_heads).contiguous(),
            a.view(t, self.cfg.linear_num_value_heads).contiguous(),
        )

    def __call__(self, x: torch.Tensor, state: DecodeState) -> torch.Tensor:
        ops = cuda_ops()
        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        q, k, v, z, b, a = self.project(x)

        conv_in = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)], dim=0).contiguous()
        conv_out = depthwise_conv_update(state.gdn_conv_states[self.layer], conv_in, self.conv_weight, self.conv_bias)
        q2, k2, v2 = torch.split(
            conv_out,
            [
                self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads,
                self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads,
                self.cfg.linear_value_head_dim * self.cfg.linear_num_value_heads,
            ],
        )
        q = q2.view(self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous()
        k = k2.view_as(q).contiguous()
        v = v2.view(self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous()

        core = ops.gdn_recurrent_ab(
            q,
            k,
            v,
            a.to(device=q.device, dtype=torch.float16).contiguous(),
            b.to(device=q.device, dtype=torch.float16).contiguous(),
            self.A_log,
            self.dt_bias,
            state.gdn_states[self.layer],
        )
        core_norm = gdn_norm_silu_gate_2d(
            core.reshape(1, -1),
            self.gdn_norm_w,
            z.reshape(1, -1),
            self.cfg.rms_norm_eps,
        ).reshape(-1)
        x = residual + linear_any(self.out_proj, core_norm)

        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(x)

    def forward_block(self, x: torch.Tensor, state: DecodeState) -> torch.Tensor:
        ops = cuda_ops()
        residual = x.contiguous()
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        q, k, v, z, b, a = self.project_block(h)
        conv_in = torch.cat(
            [
                q.reshape(x.size(0), -1),
                k.reshape(x.size(0), -1),
                v.reshape(x.size(0), -1),
            ],
            dim=-1,
        ).contiguous()
        conv_out = depthwise_conv_update_block(
            state.gdn_conv_states[self.layer],
            conv_in,
            self.conv_weight,
            self.conv_bias,
        )
        key_dim = self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads
        value_dim = self.cfg.linear_value_head_dim * self.cfg.linear_num_value_heads
        q2, k2, v2 = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)
        q = q2.view(x.size(0), self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous()
        k = k2.view_as(q).contiguous()
        v = v2.view(x.size(0), self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous()
        core = ops.gdn_recurrent_ab_scan(
            q,
            k,
            v,
            a.to(device=x.device, dtype=torch.float16).contiguous(),
            b.to(device=x.device, dtype=torch.float16).contiguous(),
            self.A_log,
            self.dt_bias,
            state.gdn_states[self.layer],
        )
        core_norm = gdn_norm_silu_gate_2d(
            core.reshape(x.size(0), -1),
            self.gdn_norm_w,
            z.reshape(x.size(0), -1),
            self.cfg.rms_norm_eps,
        )
        h = residual + linear_any(self.out_proj, core_norm)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp.forward_block(h)

    def forward_decode_batch(self, x: torch.Tensor, states: Sequence[DecodeState], plan: Any | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.size(0) != len(states):
            raise ValueError("decode batch input must be [batch, hidden] and match states")
        ops = cuda_ops()
        residual = x.contiguous()
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        q, k, v, z, b, a = self.project_block(h)
        key_dim = self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads
        value_dim = self.cfg.linear_value_head_dim * self.cfg.linear_num_value_heads
        conv_in = torch.cat(
            [
                q.reshape(x.size(0), -1),
                k.reshape(x.size(0), -1),
                v.reshape(x.size(0), -1),
            ],
            dim=-1,
        ).contiguous()
        arena_ctx = _arena_batch(states)
        use_batch_conv_kernel = arena_ctx is not None and batch_conv_kernels_enabled()
        use_batch_gdn_kernel = arena_ctx is not None and batch_gdn_kernels_enabled()
        if use_batch_conv_kernel:
            arena, state_indices = arena_ctx
            conv_out = depthwise_conv_update_batch(
                arena,
                self.layer,
                state_indices,
                conv_in,
                self.conv_weight,
                self.conv_bias,
            )
        else:
            conv_rows = []
            for row, state in enumerate(states):
                conv_rows.append(depthwise_conv_update(state.gdn_conv_states[self.layer], conv_in[row], self.conv_weight, self.conv_bias))
            conv_out = torch.stack(conv_rows, dim=0).contiguous()
        q2, k2, v2 = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)
        q = q2.view(x.size(0), self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous()
        k = k2.view_as(q).contiguous()
        v = v2.view(x.size(0), self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous()
        a_h = a.to(device=x.device, dtype=torch.float16).contiguous()
        b_h = b.to(device=x.device, dtype=torch.float16).contiguous()
        if use_batch_gdn_kernel:
            arena, state_indices = arena_ctx
            core = ops.gdn_recurrent_ab_batch(
                q,
                k,
                v,
                a_h,
                b_h,
                self.A_log,
                self.dt_bias,
                getattr(arena, "gdn_states")[self.layer],
                state_indices.to(device=x.device, dtype=torch.long).contiguous(),
            )
        else:
            core_rows = []
            for row, state in enumerate(states):
                core_rows.append(
                    ops.gdn_recurrent_ab(
                        q[row],
                        k[row],
                        v[row],
                        a_h[row],
                        b_h[row],
                        self.A_log,
                        self.dt_bias,
                        state.gdn_states[self.layer],
                    )
                )
            core = torch.stack(core_rows, dim=0).contiguous()
        core_norm = gdn_norm_silu_gate_2d(
            core.reshape(x.size(0), -1),
            self.gdn_norm_w,
            z.reshape(x.size(0), -1),
            self.cfg.rms_norm_eps,
        )
        h = residual + linear_any(self.out_proj, core_norm)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp.forward_block(h)

    def forward_speculative_batch(
        self,
        x: torch.Tensor,
        states: Sequence[DecodeState],
        commit_tokens: torch.Tensor,
        plan: Any | None = None,
        trajectory: dict[str, object] | None = None,
    ) -> torch.Tensor:
        """Run this GDN layer over [B,T,D] with commit-token state semantics."""

        if x.ndim != 3 or x.size(0) != len(states):
            raise ValueError("speculative GDN input must be [batch,tokens,hidden] and match states")
        arena_ctx = _arena_batch(states)
        if arena_ctx is None:
            raise RuntimeError("speculative GDN batch requires arena-backed states")
        arena, state_indices = arena_ctx
        batch, tokens, hidden = x.shape
        flat = x.reshape(batch * tokens, hidden).contiguous()
        residual = flat
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        q, k, v, z, b, a = self.project_block(h)
        key_dim = self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads
        value_dim = self.cfg.linear_value_head_dim * self.cfg.linear_num_value_heads
        conv_in = torch.cat(
            [
                q.reshape(batch * tokens, -1),
                k.reshape(batch * tokens, -1),
                v.reshape(batch * tokens, -1),
            ],
            dim=-1,
        ).reshape(batch, tokens, -1).contiguous()
        if trajectory is not None:
            workspace = trajectory.get("workspace") if isinstance(trajectory, dict) else None
            layer_ws = workspace.get(self.layer, {}) if isinstance(workspace, dict) else {}
            conv_out_raw, conv_traj = depthwise_conv_update_spec_trajectory(
                arena,
                self.layer,
                state_indices,
                conv_in,
                self.conv_weight,
                self.conv_bias,
                out=layer_ws.get("conv_out"),
                trajectory=layer_ws.get("conv_traj"),
            )
            trajectory.setdefault("conv", []).append((self.layer, conv_traj))
            conv_out = conv_out_raw.reshape(batch * tokens, -1)
        else:
            conv_out = depthwise_conv_update_spec(
                arena,
                self.layer,
                state_indices,
                commit_tokens,
                conv_in,
                self.conv_weight,
                self.conv_bias,
            ).reshape(batch * tokens, -1)
        q2, k2, v2 = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)
        q = q2.view(batch, tokens, self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous()
        k = k2.view_as(q).contiguous()
        v = v2.view(batch, tokens, self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous()
        ops = cuda_ops()
        a_spec = a.to(device=x.device, dtype=torch.float16).view(batch, tokens, self.cfg.linear_num_value_heads).contiguous()
        b_spec = b.to(device=x.device, dtype=torch.float16).view(batch, tokens, self.cfg.linear_num_value_heads).contiguous()
        if trajectory is not None:
            workspace = trajectory.get("workspace") if isinstance(trajectory, dict) else None
            layer_ws = workspace.get(self.layer, {}) if isinstance(workspace, dict) else {}
            core = layer_ws.get("gdn_out")
            gdn_traj = layer_ws.get("gdn_traj")
            if core is None or gdn_traj is None:
                core, gdn_traj = ops.gdn_recurrent_ab_spec_trajectory(
                    q,
                    k,
                    v,
                    a_spec,
                    b_spec,
                    self.A_log,
                    self.dt_bias,
                    getattr(arena, "gdn_states")[self.layer],
                    state_indices.to(device=x.device, dtype=torch.long).contiguous(),
                )
            else:
                ops.gdn_recurrent_ab_spec_trajectory_out(
                    q,
                    k,
                    v,
                    a_spec,
                    b_spec,
                    self.A_log,
                    self.dt_bias,
                    getattr(arena, "gdn_states")[self.layer],
                    state_indices.to(device=x.device, dtype=torch.long).contiguous(),
                    core,
                    gdn_traj,
                )
            trajectory.setdefault("gdn", []).append((self.layer, gdn_traj))
        else:
            core = ops.gdn_recurrent_ab_spec(
                q,
                k,
                v,
                a_spec,
                b_spec,
                self.A_log,
                self.dt_bias,
                getattr(arena, "gdn_states")[self.layer],
                state_indices.to(device=x.device, dtype=torch.long).contiguous(),
                commit_tokens.to(device=x.device, dtype=torch.int32).contiguous(),
            )
        core_norm = gdn_norm_silu_gate_2d(
            core.reshape(batch * tokens, -1),
            self.gdn_norm_w,
            z.reshape(batch * tokens, -1),
            self.cfg.rms_norm_eps,
        )
        h = residual + linear_any(self.out_proj, core_norm)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return (residual + self.mlp.forward_block(h)).reshape(batch, tokens, hidden).contiguous()


class Qwen36AttentionLayer:
    def __init__(self, cfg: Qwen36_27B_TextConfig, weights: WeightResolver, layer: int):
        self.cfg = cfg
        self.layer = layer
        p = f"model.layers.{layer}"
        sa = f"{p}.self_attn"
        self.input_norm = weights.fp16(f"{p}.input_layernorm.weight", f"{p}.input_norm.weight")
        self.post_norm = weights.fp16(f"{p}.post_attention_layernorm.weight", f"{p}.post_norm.weight")
        self.qkv_proj = weights.optional(f"{sa}.qkv_proj.weight")
        self.q_proj = None if self.qkv_proj is not None else weights.any_linear(f"{sa}.q_proj.weight")
        self.k_proj = None if self.qkv_proj is not None else weights.any_linear(f"{sa}.k_proj.weight")
        self.v_proj = None if self.qkv_proj is not None else weights.any_linear(f"{sa}.v_proj.weight")
        kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
        self.qkv_q_rows = tensor_rows(self.qkv_proj) - 2 * kv_rows if self.qkv_proj is not None else tensor_rows(self.q_proj)  # type: ignore[arg-type]
        self.o_proj = weights.any_linear(f"{sa}.o_proj.weight")
        self.q_norm = weights.optional_fp16(f"{sa}.q_norm.weight")
        self.k_norm = weights.optional_fp16(f"{sa}.k_norm.weight")
        self.mlp = Qwen36MLP(cfg, weights, layer)

    def __call__(self, x: torch.Tensor, state: DecodeState) -> torch.Tensor:
        ops = cuda_ops()
        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        if self.qkv_proj is not None:
            qkv_all = linear_any(self.qkv_proj, x)
            kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
            q_all, k_all, v_all = torch.split(qkv_all, [self.qkv_q_rows, kv_rows, kv_rows], dim=0)
        else:
            q_all = linear_any(self.q_proj, x)  # type: ignore[arg-type]
            k_all = linear_any(self.k_proj, x)  # type: ignore[arg-type]
            v_all = linear_any(self.v_proj, x)  # type: ignore[arg-type]
        q_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim
        if q_all.numel() == q_dim * 2:
            q_heads = q_all.view(self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
            q, gate = torch.chunk(q_heads, 2, dim=-1)
            q = q.contiguous()
            gate_flat = gate.reshape(-1).contiguous()
        else:
            q = q_all.view(self.cfg.num_attention_heads, self.cfg.attention_head_dim)
            gate_flat = None
        k = k_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        v = v_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)

        if self.q_norm is not None:
            q = qwen_rmsnorm_lastdim(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = qwen_rmsnorm_lastdim(k, self.k_norm, self.cfg.rms_norm_eps)
        q, k = apply_rope_single_token(q, k, pos=state.pos, rope_dim=self.cfg.rope_dim, rope_theta=self.cfg.rope_theta)

        state.append_attention_kv(self.layer, k.contiguous(), v.contiguous())
        k_cache, v_cache, length = state.attention_kv_view(self.layer)
        att = attention_decode_any(q.contiguous(), k_cache, v_cache, length, self.cfg.attention_head_dim ** -0.5)
        att_flat = att.reshape(-1).contiguous()
        if gate_flat is not None:
            att_flat = att_flat * torch.sigmoid(gate_flat.to(att_flat.dtype))
        x = residual + linear_any(self.o_proj, att_flat)

        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(x)

    def forward_block(self, x: torch.Tensor, state: DecodeState, *, base_pos: int, base_kv_len: int) -> torch.Tensor:
        residual = x.contiguous()
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        if self.qkv_proj is not None:
            qkv_all = linear_any(self.qkv_proj, h)
            kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
            q_all, k_all, v_all = torch.split(qkv_all, [self.qkv_q_rows, kv_rows, kv_rows], dim=-1)
        else:
            q_all = linear_any(self.q_proj, h)  # type: ignore[arg-type]
            k_all = linear_any(self.k_proj, h)  # type: ignore[arg-type]
            v_all = linear_any(self.v_proj, h)  # type: ignore[arg-type]

        q_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim
        gate_flat: torch.Tensor | None
        if q_all.size(-1) == q_dim * 2:
            q_heads = q_all.view(x.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
            q, gate = torch.chunk(q_heads, 2, dim=-1)
            q = q.contiguous()
            gate_flat = gate.reshape(x.size(0), -1).contiguous()
        else:
            q = q_all.view(x.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim).contiguous()
            gate_flat = None
        k = k_all.view(x.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        v = v_all.view(x.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        if self.q_norm is not None:
            q = qwen_rmsnorm_lastdim(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = qwen_rmsnorm_lastdim(k, self.k_norm, self.cfg.rms_norm_eps)

        can_sdpa_prefill = (
            x.device.type == "cuda"
            and not state.kv_cache_spec.is_int4
            and base_pos >= 0
            and base_pos + x.size(0) <= state.max_seq_len
            and base_kv_len < state.max_seq_len
        )
        if can_sdpa_prefill:
            q_rope, k_rope = apply_rope_block(
                q,
                k,
                start_pos=base_pos,
                rope_dim=self.cfg.rope_dim,
                rope_theta=self.cfg.rope_theta,
            )
            state.attn_k[self.layer][:, base_pos : base_pos + x.size(0), :].copy_(k_rope.permute(1, 0, 2).contiguous())
            state.attn_v[self.layer][:, base_pos : base_pos + x.size(0), :].copy_(v.permute(1, 0, 2).contiguous())
            live_len = min(base_kv_len + x.size(0), state.max_seq_len)
            k_live, v_live, causal = prefill_sdpa_window(
                state.attn_k[self.layer][:, :live_len, :].to(dtype=torch.float16).contiguous(),
                state.attn_v[self.layer][:, :live_len, :].to(dtype=torch.float16).contiguous(),
                live_len=int(live_len),
                query_start=int(base_pos),
                query_len=int(x.size(0)),
            )
            att = F.scaled_dot_product_attention(
                q_rope.permute(1, 0, 2).unsqueeze(0).contiguous(),
                k_live.unsqueeze(0),
                v_live.unsqueeze(0),
                attn_mask=causal,
                dropout_p=0.0,
                scale=self.cfg.attention_head_dim ** -0.5,
                enable_gqa=(self.cfg.num_attention_heads != self.cfg.num_key_value_heads),
            )
            att_block = att.squeeze(0).permute(1, 0, 2).reshape(x.size(0), -1).contiguous()
            if gate_flat is not None:
                att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
            h = residual + linear_any(self.o_proj, att_block)
            residual = h
            h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
            return residual + self.mlp.forward_block(h)

        can_staged_int4_sdpa_prefill = (
            x.device.type == "cuda"
            and state.kv_cache_spec.is_int4
            and base_pos >= 0
            and base_pos + x.size(0) <= state.max_seq_len
            and base_kv_len < state.max_seq_len
        )
        if can_staged_int4_sdpa_prefill:
            q_rope, k_rope = apply_rope_block(
                q,
                k,
                start_pos=base_pos,
                rope_dim=self.cfg.rope_dim,
                rope_theta=self.cfg.rope_theta,
            )
            write_start = int(base_pos)
            write_end = int(base_pos) + int(x.size(0))
            live_len = min(base_kv_len + x.size(0), state.max_seq_len)
            policy = prefill_attention_policy()
            if (
                int(base_kv_len) == 0
                and policy.allows_fresh_sdpa(tokens=int(x.size(0)))
                and _short_prefill_sdpa_has_free_vram(x.device)
            ):
                k_scratch = k_rope.permute(1, 0, 2).contiguous()
                v_scratch = v.permute(1, 0, 2).contiguous()
                causal = torch.tril(torch.ones((x.size(0), x.size(0)), device=x.device, dtype=torch.bool))
                att = F.scaled_dot_product_attention(
                    q_rope.permute(1, 0, 2).unsqueeze(0).contiguous(),
                    k_scratch.unsqueeze(0),
                    v_scratch.unsqueeze(0),
                    attn_mask=causal,
                    dropout_p=0.0,
                    scale=self.cfg.attention_head_dim ** -0.5,
                    enable_gqa=(self.cfg.num_attention_heads != self.cfg.num_key_value_heads),
                )
                state.append_attention_kv_block_at(
                    self.layer,
                    k_rope.contiguous(),
                    v.contiguous(),
                    start_logical_pos=write_start,
                )
                att_block = att.squeeze(0).permute(1, 0, 2).reshape(x.size(0), -1).contiguous()
                if gate_flat is not None:
                    att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
                h = residual + linear_any(self.o_proj, att_block)
                residual = h
                h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
                return residual + self.mlp.forward_block(h)
            if (
                int(base_kv_len) > 0
                and policy.allows_extend_sdpa(live_tokens=int(live_len))
                and _short_prefill_sdpa_has_free_vram(x.device)
            ):
                state.append_attention_kv_block_at(
                    self.layer,
                    k_rope.contiguous(),
                    v.contiguous(),
                    start_logical_pos=write_start,
                )
                k_cache, v_cache, live_len = state.attention_kv_view_at(
                    self.layer,
                    logical_pos=write_end - 1,
                    live_len=int(live_len),
                )
                k_live, v_live, causal = prefill_sdpa_window(
                    k_cache.to(dtype=torch.float16).contiguous(),
                    v_cache.to(dtype=torch.float16).contiguous(),
                    live_len=int(live_len),
                    query_start=int(base_pos),
                    query_len=int(x.size(0)),
                )
                att = F.scaled_dot_product_attention(
                    q_rope.permute(1, 0, 2).unsqueeze(0).contiguous(),
                    k_live.unsqueeze(0),
                    v_live.unsqueeze(0),
                    attn_mask=causal,
                    dropout_p=0.0,
                    scale=self.cfg.attention_head_dim ** -0.5,
                    enable_gqa=(self.cfg.num_attention_heads != self.cfg.num_key_value_heads),
                )
                att_block = att.squeeze(0).permute(1, 0, 2).reshape(x.size(0), -1).contiguous()
                if gate_flat is not None:
                    att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
                h = residual + linear_any(self.o_proj, att_block)
                residual = h
                h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
                return residual + self.mlp.forward_block(h)

        att_rows = []
        for offset in range(x.size(0)):
            logical_pos = base_pos + offset
            live_len = min(base_kv_len + offset + 1, state.max_seq_len)
            q_row, k_row = apply_rope_single_token(
                q[offset],
                k[offset],
                pos=logical_pos,
                rope_dim=self.cfg.rope_dim,
                rope_theta=self.cfg.rope_theta,
            )
            state.append_attention_kv_at(self.layer, k_row.contiguous(), v[offset], logical_pos=logical_pos)
            k_cache, v_cache, length = state.attention_kv_view_at(self.layer, logical_pos=logical_pos, live_len=live_len)
            att = attention_decode_any(q_row.contiguous(), k_cache, v_cache, length, self.cfg.attention_head_dim ** -0.5)
            att_flat = att.reshape(-1).contiguous()
            if gate_flat is not None:
                att_flat = att_flat * torch.sigmoid(gate_flat[offset].to(att_flat.dtype))
            att_rows.append(att_flat)
        att_block = torch.stack(att_rows, dim=0).contiguous()
        h = residual + linear_any(self.o_proj, att_block)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp.forward_block(h)

    def forward_paged_prefill_block(
        self,
        x: torch.Tensor,
        state: DecodeState,
        plan: DecodeBatchPlan,
        *,
        row: int,
        start: int,
    ) -> torch.Tensor:
        """Block prefill for paged KV states.

        Qwen3.6 attention remains state-trajectory sequential, but the large
        qkv/o-proj/MLP projections should run over [T, D] instead of re-running
        the whole layer stack one token at a time.
        """

        residual = x.contiguous()
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        if self.qkv_proj is not None:
            qkv_all = linear_any(self.qkv_proj, h)
            kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
            q_all, k_all, v_all = torch.split(qkv_all, [self.qkv_q_rows, kv_rows, kv_rows], dim=-1)
        else:
            q_all = linear_any(self.q_proj, h)  # type: ignore[arg-type]
            k_all = linear_any(self.k_proj, h)  # type: ignore[arg-type]
            v_all = linear_any(self.v_proj, h)  # type: ignore[arg-type]

        q_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim
        gate_flat: torch.Tensor | None
        if q_all.size(-1) == q_dim * 2:
            q_heads = q_all.view(x.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
            q, gate = torch.chunk(q_heads, 2, dim=-1)
            q = q.contiguous()
            gate_flat = gate.reshape(x.size(0), -1).contiguous()
        else:
            q = q_all.view(x.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim).contiguous()
            gate_flat = None
        k = k_all.view(x.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        v = v_all.view(x.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        if self.q_norm is not None:
            q = qwen_rmsnorm_lastdim(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = qwen_rmsnorm_lastdim(k, self.k_norm, self.cfg.rms_norm_eps)

        arena_ctx = _arena_batch([state])
        if arena_ctx is None:
            raise RuntimeError("paged prefill block requires arena-backed state")
        arena, state_indices = arena_ctx
        device = x.device
        block_tables = plan.block_tables
        slot_mapping = plan.slot_mapping
        if block_tables is None or slot_mapping is None:
            raise RuntimeError("paged prefill block requires block_tables and slot_mapping")
        state_indices = plan.state_indices.to(device=device, dtype=torch.int32)[row : row + 1].contiguous()
        row_block_tables = block_tables.to(device=device, dtype=torch.int32)[row : row + 1].contiguous()
        slot_mapping = slot_mapping.to(device=device, dtype=torch.long)
        positions = plan.positions.to(device=device, dtype=torch.long)
        block_positions = positions[int(start) : int(start) + x.size(0)].contiguous()
        q_rope, k_rope = apply_rope_decode_batch(
            q,
            k,
            positions=block_positions,
            rope_dim=self.cfg.rope_dim,
            rope_theta=self.cfg.rope_theta,
        )
        kv_spec = getattr(arena, "kv_cache_spec")
        block_slots = slot_mapping[int(start) : int(start) + x.size(0)].contiguous()
        if kv_spec.is_int4 and x.device.type == "cuda":
            write_start = int(getattr(state, "pos", 0))
            write_end = write_start + x.size(0)
            policy = prefill_attention_policy()
            if (
                write_start == 0
                and policy.allows_fresh_sdpa(tokens=int(x.size(0)))
                and _short_prefill_sdpa_has_free_vram(x.device)
            ):
                k_scratch = k_rope.permute(1, 0, 2).contiguous()
                v_scratch = v.permute(1, 0, 2).contiguous()
                causal = torch.tril(torch.ones((x.size(0), x.size(0)), device=x.device, dtype=torch.bool))
                att = F.scaled_dot_product_attention(
                    q_rope.permute(1, 0, 2).unsqueeze(0).contiguous(),
                    k_scratch.unsqueeze(0),
                    v_scratch.unsqueeze(0),
                    attn_mask=causal,
                    dropout_p=0.0,
                    scale=self.cfg.attention_head_dim ** -0.5,
                    enable_gqa=(self.cfg.num_attention_heads != self.cfg.num_key_value_heads),
                )
                append_paged_int4_block(arena, self.layer, k_rope, v, block_slots)
                att_block = att.squeeze(0).permute(1, 0, 2).reshape(x.size(0), -1).contiguous()
                if gate_flat is not None:
                    att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
                h = residual + linear_any(self.o_proj, att_block)
                residual = h
                h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
                return residual + self.mlp.forward_block(h)
            append_paged_int4_block(arena, self.layer, k_rope, v, block_slots)
            if (
                write_start > 0
                and policy.allows_extend_sdpa(live_tokens=int(write_end))
                and _short_prefill_sdpa_has_free_vram(x.device)
                and arena_has_canonical_attention_mirror(arena, self.layer)
            ):
                slot = int(getattr(state, "arena_slot", 0))
                k_cache = getattr(arena, "attn_k")[self.layer][slot]
                v_cache = getattr(arena, "attn_v")[self.layer][slot]
                k_cache[:, write_start:write_end, :].copy_(k_rope.permute(1, 0, 2).contiguous())
                v_cache[:, write_start:write_end, :].copy_(v.permute(1, 0, 2).contiguous())
                k_live, v_live, causal = prefill_sdpa_window(
                    k_cache.to(dtype=torch.float16).contiguous(),
                    v_cache.to(dtype=torch.float16).contiguous(),
                    live_len=int(write_end),
                    query_start=int(write_start),
                    query_len=int(x.size(0)),
                )
                att = F.scaled_dot_product_attention(
                    q_rope.permute(1, 0, 2).unsqueeze(0).contiguous(),
                    k_live.unsqueeze(0),
                    v_live.unsqueeze(0),
                    attn_mask=causal,
                    dropout_p=0.0,
                    scale=self.cfg.attention_head_dim ** -0.5,
                    enable_gqa=(self.cfg.num_attention_heads != self.cfg.num_key_value_heads),
                )
                att_block = att.squeeze(0).permute(1, 0, 2).reshape(x.size(0), -1).contiguous()
                if gate_flat is not None:
                    att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
                h = residual + linear_any(self.o_proj, att_block)
                residual = h
                h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
                return residual + self.mlp.forward_block(h)
        seq_lens = (block_positions + 1).to(dtype=torch.int32).contiguous()
        block_size = x.size(0)
        subplan = SimpleNamespace(
            request_ids=[str(plan.request_ids[row])],
            state_indices=state_indices.expand(block_size).contiguous(),
            input_ids=torch.zeros((block_size,), dtype=torch.long, device=device),
            positions=block_positions[:1],
            query_start_loc=torch.arange(0, block_size + 1, dtype=torch.int32, device=device),
            seq_lens=seq_lens,
            logits_indices=torch.arange(0, block_size, dtype=torch.long, device=device),
            cu_num_logits=torch.arange(0, block_size + 1, dtype=torch.int32, device=device),
            row_spans=tuple((i, i + 1) for i in range(block_size)),
            num_scheduled_tokens=[1] * block_size,
            num_draft_tokens_per_request=[0] * block_size,
            is_prefill=[False] * block_size,
            block_tables=row_block_tables.expand(block_size, -1).contiguous(),
            slot_mapping=slot_mapping[int(start) : int(start) + 1].contiguous(),
        )
        subplan.positions = block_positions
        subplan.slot_mapping = slot_mapping[int(start) : int(start) + block_size].contiguous()
        att_block = attention_decode_paged_batch(
            arena,
            self.layer,
            q_rope,
            k_rope,
            v,
            subplan,
            self.cfg.attention_head_dim ** -0.5,
        ).reshape(block_size, -1).contiguous()
        if gate_flat is not None:
            att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
        h = residual + linear_any(self.o_proj, att_block)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp.forward_block(h)

    def forward_decode_batch(self, x: torch.Tensor, states: Sequence[DecodeState], plan: Any | None = None) -> torch.Tensor:
        if x.ndim != 2 or x.size(0) != len(states):
            raise ValueError("decode batch input must be [batch, hidden] and match states")
        residual = x.contiguous()
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        if self.qkv_proj is not None:
            qkv_all = linear_any(self.qkv_proj, h)
            kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
            q_all, k_all, v_all = torch.split(qkv_all, [self.qkv_q_rows, kv_rows, kv_rows], dim=-1)
        else:
            q_all = linear_any(self.q_proj, h)  # type: ignore[arg-type]
            k_all = linear_any(self.k_proj, h)  # type: ignore[arg-type]
            v_all = linear_any(self.v_proj, h)  # type: ignore[arg-type]

        q_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim
        gate_flat: torch.Tensor | None
        if q_all.size(-1) == q_dim * 2:
            q_heads = q_all.view(x.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
            q, gate = torch.chunk(q_heads, 2, dim=-1)
            q = q.contiguous()
            gate_flat = gate.reshape(x.size(0), -1).contiguous()
        else:
            q = q_all.view(x.size(0), self.cfg.num_attention_heads, self.cfg.attention_head_dim).contiguous()
            gate_flat = None
        k = k_all.view(x.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        v = v_all.view(x.size(0), self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        if self.q_norm is not None:
            q = qwen_rmsnorm_lastdim(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = qwen_rmsnorm_lastdim(k, self.k_norm, self.cfg.rms_norm_eps)

        arena_ctx = _arena_batch(states)
        use_paged_attention = False
        if (
            arena_ctx is not None
            and plan is not None
            and getattr(plan, "slot_mapping", None) is not None
            and getattr(plan, "block_tables", None) is not None
            and getattr(arena_ctx[0], "paged_attn_k", None) is not None
        ):
            arena, _state_indices = arena_ctx
            use_paged_attention = paged_attention_kernels_enabled() or not arena_has_canonical_attention_mirror(arena, self.layer)
        if use_paged_attention:
            arena, _state_indices = arena_ctx  # type: ignore[misc]
            positions = plan.positions.to(device=x.device, dtype=torch.long).contiguous()
            q_rope, k_rope = apply_rope_decode_batch(
                q,
                k,
                positions=positions,
                rope_dim=self.cfg.rope_dim,
                rope_theta=self.cfg.rope_theta,
            )
            att_block = attention_decode_paged_batch(
                arena,
                self.layer,
                q_rope,
                k_rope,
                v,
                plan,
                self.cfg.attention_head_dim ** -0.5,
            ).reshape(x.size(0), -1).contiguous()
            if gate_flat is not None:
                att_block = att_block * torch.sigmoid(gate_flat.to(att_block.dtype))
        else:
            att_rows = []
            for row, state in enumerate(states):
                q_row, k_row = apply_rope_single_token(
                    q[row],
                    k[row],
                    pos=state.pos,
                    rope_dim=self.cfg.rope_dim,
                    rope_theta=self.cfg.rope_theta,
                )
                state.append_attention_kv(self.layer, k_row.contiguous(), v[row].contiguous())
                k_cache, v_cache, length = state.attention_kv_view(self.layer)
                att = attention_decode_any(q_row.contiguous(), k_cache, v_cache, length, self.cfg.attention_head_dim ** -0.5)
                att_flat = att.reshape(-1).contiguous()
                if gate_flat is not None:
                    att_flat = att_flat * torch.sigmoid(gate_flat[row].to(att_flat.dtype))
                att_rows.append(att_flat)
                arena_ctx_row = _arena_batch(states)
                if arena_ctx_row is not None:
                    arena, _state_indices = arena_ctx_row
                    write_paged_kv_row(arena, self.layer, row, k_row.contiguous(), v[row].contiguous(), plan)
            att_block = torch.stack(att_rows, dim=0).contiguous()
        h = residual + linear_any(self.o_proj, att_block)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp.forward_block(h)

    def forward_speculative_batch(
        self,
        x: torch.Tensor,
        states: Sequence[DecodeState],
        commit_tokens: torch.Tensor,
        plan: Any | None = None,
        trajectory: dict[str, list] | None = None,
    ) -> torch.Tensor:
        """Run this attention layer over [B,T,D] without verifier live-KV pollution."""

        if x.ndim != 3 or x.size(0) != len(states):
            raise ValueError("speculative attention input must be [batch,tokens,hidden] and match states")
        arena_ctx = _arena_batch(states)
        if arena_ctx is None or plan is None:
            raise RuntimeError("speculative attention batch requires arena-backed states and a decode plan")
        arena, state_indices = arena_ctx
        kv_spec = getattr(arena, "kv_cache_spec")
        if not kv_spec.is_int4:
            raise RuntimeError("speculative attention hot path currently requires INT4/BDR paged KV")
        if getattr(plan, "block_tables", None) is None or getattr(plan, "slot_mapping", None) is None:
            raise RuntimeError("speculative attention requires block_tables and slot_mapping")

        batch, tokens, hidden = x.shape
        flat = x.reshape(batch * tokens, hidden).contiguous()
        residual = flat
        h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
        if self.qkv_proj is not None:
            qkv_all = linear_any(self.qkv_proj, h)
            kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
            q_all, k_all, v_all = torch.split(qkv_all, [self.qkv_q_rows, kv_rows, kv_rows], dim=-1)
        else:
            q_all = linear_any(self.q_proj, h)  # type: ignore[arg-type]
            k_all = linear_any(self.k_proj, h)  # type: ignore[arg-type]
            v_all = linear_any(self.v_proj, h)  # type: ignore[arg-type]

        q_dim = self.cfg.num_attention_heads * self.cfg.attention_head_dim
        gate_flat: torch.Tensor | None
        if q_all.size(-1) == q_dim * 2:
            q_heads = q_all.view(batch * tokens, self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
            q, gate = torch.chunk(q_heads, 2, dim=-1)
            q = q.contiguous()
            gate_flat = gate.reshape(batch * tokens, -1).contiguous()
        else:
            q = q_all.view(batch * tokens, self.cfg.num_attention_heads, self.cfg.attention_head_dim).contiguous()
            gate_flat = None
        k = k_all.view(batch * tokens, self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        v = v_all.view(batch * tokens, self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        if self.q_norm is not None:
            q = qwen_rmsnorm_lastdim(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = qwen_rmsnorm_lastdim(k, self.k_norm, self.cfg.rms_norm_eps)

        base_positions = _arena_positions(arena, state_indices, device=x.device)
        positions = (base_positions.unsqueeze(1) + torch.arange(tokens, device=x.device, dtype=torch.long).unsqueeze(0)).reshape(batch * tokens).contiguous()
        q_rope, k_rope = apply_rope_decode_batch(
            q,
            k,
            positions=positions,
            rope_dim=self.cfg.rope_dim,
            rope_theta=self.cfg.rope_theta,
        )
        q_btd = q_rope.reshape(batch, tokens, self.cfg.num_attention_heads, self.cfg.attention_head_dim).contiguous()
        k_btd = k_rope.reshape(batch, tokens, self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        v_btd = v.reshape(batch, tokens, self.cfg.num_key_value_heads, self.cfg.attention_head_dim).contiguous()
        block_tables = plan.block_tables.to(device=x.device, dtype=torch.int32).contiguous()
        base_seq_lens = base_positions.to(dtype=torch.int32).contiguous()
        att = cuda_ops().attention_spec_decode_paged_int4(
            q_btd,
            k_btd,
            v_btd,
            getattr(arena, "paged_attn_k")[self.layer],
            getattr(arena, "paged_attn_v")[self.layer],
            getattr(arena, "paged_attn_k_scale")[self.layer],
            getattr(arena, "paged_attn_v_scale")[self.layer],
            getattr(arena, "paged_attn_k_zero")[self.layer],
            getattr(arena, "paged_attn_v_zero")[self.layer],
            block_tables,
            base_seq_lens,
            int(getattr(arena, "kv_block_size")),
            self.cfg.attention_head_dim ** -0.5,
            int(kv_spec.hadamard_order),
            bool(kv_spec.uses_bdr),
            bool(kv_spec.rotate_v),
            _paged_int4_tiled_layout(arena),
        ).reshape(batch * tokens, -1).contiguous()
        if gate_flat is not None:
            att = att * torch.sigmoid(gate_flat.to(att.dtype))

        slot_mapping = plan.slot_mapping.to(device=x.device, dtype=torch.long)
        if slot_mapping.ndim == 1:
            row_spans = getattr(plan, "row_spans", None)
            if row_spans is None:
                raise RuntimeError("flat speculative slot_mapping requires row_spans")
            rows = []
            for row in range(batch):
                start, end = row_spans[row]
                slots = slot_mapping[int(start) : int(end)]
                if slots.numel() < tokens:
                    raise RuntimeError("slot_mapping row has fewer slots than speculative tokens")
                rows.append(slots[:tokens])
            slot_mapping = torch.stack(rows, dim=0).contiguous()
        if trajectory is not None:
            trajectory.setdefault("attn", []).append((self.layer, k_btd, v_btd, slot_mapping))
        else:
            append_paged_int4_spec_block(arena, self.layer, k_btd, v_btd, slot_mapping, commit_tokens)

        h = residual + linear_any(self.o_proj, att)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return (residual + self.mlp.forward_block(h)).reshape(batch, tokens, hidden).contiguous()


class Qwen36Model:
    def __init__(
        self,
        store: QuantizedStore,
        cfg: Qwen36_27B_TextConfig | None = None,
        device: str | torch.device = "cuda",
        embed_store: QuantizedStore | None = None,
        head_store: QuantizedStore | None = None,
    ):
        self.cfg = cfg or Qwen36_27B_TextConfig()
        self.device = torch.device(device)
        self.store = store
        wr = WeightResolver(store)
        embed_wr = WeightResolver(embed_store or store)
        head_wr = WeightResolver(head_store or store)
        self.embed = embed_wr.get("model.embed_tokens.weight", "model.tok_embeddings.weight")
        self.final_norm = wr.fp16("model.norm.weight", "model.final_layernorm.weight")
        lm = head_wr.optional("lm_head.weight", "model.lm_head.weight", "model.output.weight")
        if lm is None:
            # Rare tied-output fallback.  For low-bit embed this is intentionally
            # not used because full vocab projection from embedding rows would be slow.
            if isinstance(self.embed, FP16Tensor):
                lm = FP16Tensor("tied_lm_head", self.embed.value)
            else:
                raise KeyError("lm_head.weight is required when embeddings are low-bit")
        self.lm_head = lm
        self._verify_graph_cache = CudaGraphReplayCache(
            enabled=os.environ.get("LANGBURST_CUDA_GRAPH", "").strip().lower() in {"1", "true", "on", "yes"},
            warmup_steps=0,
        )
        self._verify_graph_outputs: dict[CudaGraphKey, list[VerifyBlockResult]] = {}
        self._spec_workspace_cache: dict[tuple[str, int, int, torch.dtype, int], dict[int, dict[str, torch.Tensor]]] = {}
        self._spec_index_cache: dict[tuple[str, int, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}
        self.layers = []
        for i in range(self.cfg.num_layers):
            if self.cfg.layer_type(i) == "gdn":
                self.layers.append(Qwen36GDNLayer(self.cfg, wr, i, self.device))
            else:
                self.layers.append(Qwen36AttentionLayer(self.cfg, wr, i))

    def reset(self) -> None:
        return None

    def clear_runtime_caches(self) -> None:
        self._verify_graph_outputs.clear()
        self._spec_workspace_cache.clear()
        self._spec_index_cache.clear()

    @torch.no_grad()
    def forward_one(
        self,
        token: torch.Tensor | int,
        state: DecodeState,
        *,
        use_mtp: bool = False,
        return_hidden: bool = False,
        return_raw_hidden: bool = False,
        return_logits: bool = True,
        hidden_tap_layers: Sequence[int] | None = None,
    ):
        if not torch.is_tensor(token):
            token = torch.tensor(int(token), device=self.device, dtype=torch.long)
        token = token.to(device=self.device, dtype=torch.long).reshape(())
        x = embed_lookup(self.embed, token).to(self.device, non_blocking=True).reshape(-1).contiguous()
        tap_set = set(hidden_tap_layers or ())
        hidden_taps: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, state)
            if layer_idx in tap_set:
                hidden_taps.append(x)
        state.finish_token()
        if hidden_tap_layers is not None and len(hidden_taps) != len(tap_set):
            missing = sorted(tap_set - set(range(len(self.layers))))
            raise ValueError(f"hidden tap layer out of range: {missing}")
        if not return_logits:
            if return_hidden:
                if not return_raw_hidden:
                    x = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
                if hidden_tap_layers is not None:
                    return None, x, hidden_taps
                return None, x
            if use_mtp:
                return None, []
            if hidden_tap_layers is not None:
                return None, hidden_taps
            return None
        raw_hidden = x
        x = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
        logits = lowbit_linear_on_device(self.lm_head, x, self.device) if isinstance(self.lm_head, LowBitTensor) else linear_any(self.lm_head, x)
        if return_hidden and use_mtp:
            hidden = raw_hidden if return_raw_hidden else x
            if hidden_tap_layers is not None:
                return logits, [], hidden, hidden_taps
            return logits, [], hidden
        if return_hidden:
            hidden = raw_hidden if return_raw_hidden else x
            if hidden_tap_layers is not None:
                return logits, hidden, hidden_taps
            return logits, hidden
        if use_mtp:
            if hidden_tap_layers is not None:
                return logits, [], hidden_taps
            return logits, []
        state.last_raw_hidden = raw_hidden.contiguous().clone()
        if hidden_tap_layers is not None:
            return logits, hidden_taps
        return logits

    @torch.no_grad()
    def forward_block(
        self,
        tokens: Sequence[int],
        state: DecodeState,
        *,
        hidden_tap_layers: Sequence[int] | None = None,
        return_logits: bool = True,
        logits_mode: Literal["all", "last"] = "all",
        commit: bool = False,
    ) -> BlockForwardResult:
        """Run a target block and commit only the requested state branch."""
        if fast_raw_block_enabled():
            branch = state if commit else state.fork()
            result = self._forward_block_raw(
                tokens,
                branch,
                hidden_tap_layers=hidden_tap_layers,
                return_logits=return_logits,
                logits_mode=logits_mode,
            )
            if commit and branch is not state:
                state.copy_from_(branch)
            return result
        branch = state if commit else state.fork()
        logits_out: list[torch.Tensor] = []
        raw_hiddens: list[torch.Tensor] = []
        final_hiddens: list[torch.Tensor] = []
        taps_by_token: list[list[torch.Tensor]] = []
        token_list = [int(t) for t in tokens]
        for i, token in enumerate(token_list):
            want_logits = return_logits and (logits_mode == "all" or i == len(token_list) - 1)
            result_one = self.forward_one(
                token,
                branch,
                return_logits=want_logits,
                return_hidden=True,
                return_raw_hidden=True,
                hidden_tap_layers=hidden_tap_layers,
            )
            if hidden_tap_layers is None:
                logits, raw_hidden = result_one
                taps: list[torch.Tensor] = []
            else:
                logits, raw_hidden, taps = result_one
            if want_logits:
                logits_out.append(logits.contiguous().clone())
            raw_hiddens.append(raw_hidden.contiguous().clone())
            final_hiddens.append(qwen_rmsnorm(raw_hidden.contiguous(), self.final_norm, self.cfg.rms_norm_eps).contiguous().clone())
            taps_by_token.append(taps)
        result = BlockForwardResult(
            logits=logits_out,
            hidden_taps=taps_by_token,
            state=branch,
            raw_hiddens=raw_hiddens,
            final_hiddens=final_hiddens,
        )
        if commit and branch is not state:
            state.copy_from_(branch)
        return result

    @torch.no_grad()
    def forward_batch(
        self,
        plan: object,
        states: Sequence[DecodeState],
        *,
        return_logits: bool = True,
    ) -> list[torch.Tensor | None]:
        """Execute a multi-request decode batch plan.

        The public contract mirrors the reference runtime's model runner: request scheduling is
        represented by one batch plan, while each row owns independent state.
        Rows use the block path so chunked prefill and spec-verify spans avoid
        token-by-token Python loops inside RuntimeEngine.
        """

        num_requests = int(getattr(plan, "num_requests"))
        if len(states) != num_requests:
            raise ValueError("states length must match plan.num_requests")
        row_spans = getattr(plan, "row_spans", None)
        draft_counts = getattr(plan, "num_draft_tokens_per_request", [0] * num_requests)
        prefill_flags = getattr(plan, "is_prefill", [False] * num_requests)
        canonical_prefill_kv = all(has_canonical_attention_kv(state) for state in states)
        if (
            return_logits
            and row_spans is not None
            and all((end - start) == 1 for start, end in row_spans)
            and all(int(v) == 0 for v in draft_counts)
            and all(not bool(v) for v in prefill_flags)
        ):
            return self._forward_single_token_batch(getattr(plan, "input_ids"), row_spans, states, plan=plan)
        if (
            row_spans is not None
            and batch_prefill_steps_enabled()
            and all(bool(v) for v in prefill_flags)
            and all(int(v) == 0 for v in draft_counts)
            and any((end - start) > 1 for start, end in row_spans)
            and num_requests == 1
            and paged_prefill_block_enabled()
        ):
            return self._forward_prefill_paged_block_single(
                getattr(plan, "input_ids"),
                row_spans,
                states,
                plan=plan,
                return_logits=return_logits,
            )
        if (
            row_spans is not None
            and batch_prefill_steps_enabled()
            and all(bool(v) for v in prefill_flags)
            and all(int(v) == 0 for v in draft_counts)
            and (num_requests > 1 or not canonical_prefill_kv)
        ):
            return self._forward_prefill_timestep_batch(
                getattr(plan, "input_ids"),
                row_spans,
                states,
                plan=plan,
                return_logits=return_logits,
            )
        input_ids = getattr(plan, "input_ids")
        outputs: list[torch.Tensor | None] = []
        for row, state in enumerate(states):
            if row_spans is not None:
                start, end = row_spans[row]
            else:
                query_start_loc = getattr(plan, "query_start_loc")
                start = int(query_start_loc[row].detach().cpu().item())
                end = int(query_start_loc[row + 1].detach().cpu().item())
            token_list = [int(t) for t in input_ids[start:end].detach().cpu().tolist()]
            if not token_list:
                outputs.append(None)
                continue
            if len(token_list) == 1:
                logits = self.forward_one(token_list[0], state, return_logits=return_logits)
                if bool(prefill_flags[row]):
                    sync_state_kv_to_paged(state, plan, row)
                outputs.append(logits if return_logits else None)
                continue
            result = self.forward_block(
                token_list,
                state,
                return_logits=return_logits,
                logits_mode="last",
                commit=True,
            )
            if bool(prefill_flags[row]):
                sync_state_kv_to_paged(state, plan, row)
            outputs.append(result.logits[-1] if return_logits and result.logits else None)
        return outputs

    @torch.no_grad()
    def forward_batch_logits(
        self,
        plan: object,
        states: Sequence[DecodeState],
    ) -> list[list[torch.Tensor]]:
        num_requests = int(getattr(plan, "num_requests"))
        if len(states) != num_requests:
            raise ValueError("states length must match plan.num_requests")
        row_spans = getattr(plan, "row_spans", None)
        draft_counts = getattr(plan, "num_draft_tokens_per_request")
        prefill_flags = getattr(plan, "is_prefill", [False] * num_requests)
        canonical_prefill_kv = all(has_canonical_attention_kv(state) for state in states)
        if (
            row_spans is not None
            and all((end - start) == 1 for start, end in row_spans)
            and all(int(v) == 0 for v in draft_counts)
            and all(not bool(v) for v in prefill_flags)
        ):
            return [[logit] for logit in self._forward_single_token_batch(getattr(plan, "input_ids"), row_spans, states, plan=plan)]
        if (
            row_spans is not None
            and batch_prefill_steps_enabled()
            and all(bool(v) for v in prefill_flags)
            and all(int(v) == 0 for v in draft_counts)
            and any((end - start) > 1 for start, end in row_spans)
            and num_requests == 1
            and not canonical_prefill_kv
            and paged_prefill_block_enabled()
        ):
            outputs = self._forward_prefill_paged_block_single(
                getattr(plan, "input_ids"),
                row_spans,
                states,
                plan=plan,
                return_logits=True,
            )
            return [[logit] if logit is not None else [] for logit in outputs]
        if (
            row_spans is not None
            and batch_prefill_steps_enabled()
            and all(bool(v) for v in prefill_flags)
            and all(int(v) == 0 for v in draft_counts)
            and (num_requests > 1 or not canonical_prefill_kv)
        ):
            outputs = self._forward_prefill_timestep_batch(
                getattr(plan, "input_ids"),
                row_spans,
                states,
                plan=plan,
                return_logits=True,
            )
            return [[logit] if logit is not None else [] for logit in outputs]
        input_ids = getattr(plan, "input_ids")
        outputs: list[list[torch.Tensor]] = []
        for row, state in enumerate(states):
            if row_spans is not None:
                start, end = row_spans[row]
            else:
                query_start_loc = getattr(plan, "query_start_loc")
                start = int(query_start_loc[row].detach().cpu().item())
                end = int(query_start_loc[row + 1].detach().cpu().item())
            token_list = [int(t) for t in input_ids[start:end].detach().cpu().tolist()]
            if not token_list:
                outputs.append([])
                continue
            logits_mode: Literal["all", "last"] = "all" if int(draft_counts[row]) > 0 else "last"
            if len(token_list) == 1:
                logits = self.forward_one(token_list[0], state, return_logits=True)
                if bool(prefill_flags[row]):
                    sync_state_kv_to_paged(state, plan, row)
                outputs.append([logits])
                continue
            result = self.forward_block(
                token_list,
                state,
                return_logits=True,
                logits_mode=logits_mode,
                commit=True,
            )
            if bool(prefill_flags[row]):
                sync_state_kv_to_paged(state, plan, row)
            outputs.append([logit for logit in result.logits])
        return outputs

    @torch.no_grad()
    def _forward_prefill_paged_block_single(
        self,
        input_ids: torch.Tensor,
        row_spans: Sequence[tuple[int, int]],
        states: Sequence[DecodeState],
        *,
        plan: DecodeBatchPlan,
        return_logits: bool,
    ) -> list[torch.Tensor | None]:
        if len(row_spans) != 1 or len(states) != 1:
            raise ValueError("paged block prefill currently supports one request row")
        start, end = row_spans[0]
        if end <= start:
            return [None]
        state = states[0]
        token_tensor = input_ids[int(start) : int(end)].to(device=self.device, dtype=torch.long)
        x = embed_lookup_batch(self.embed, token_tensor, self.device)
        for layer in self.layers:
            forward_paged_prefill_block = getattr(layer, "forward_paged_prefill_block", None)
            if callable(forward_paged_prefill_block):
                x = forward_paged_prefill_block(x, state, plan, row=0, start=int(start))
            else:
                x = layer.forward_block(x, state)
        for _ in range(int(end) - int(start)):
            state.finish_token()
        state.last_raw_hidden = x[-1].contiguous().clone()
        if not return_logits:
            return [None]
        h = qwen_rmsnorm(x[-1].contiguous(), self.final_norm, self.cfg.rms_norm_eps)
        return [linear_any(self.lm_head, h).contiguous()]

    @torch.no_grad()
    def _forward_single_token_batch(
        self,
        input_ids: torch.Tensor,
        row_spans: Sequence[tuple[int, int]],
        states: Sequence[DecodeState],
        *,
        plan: Any | None = None,
    ) -> list[torch.Tensor]:
        if len(row_spans) != len(states):
            raise ValueError("row_spans length must match states")
        token_indices = [start for start, end in row_spans if end - start == 1]
        if len(token_indices) != len(states):
            raise ValueError("single-token batch path requires exactly one token per row")
        token_tensor = input_ids[token_indices].to(device=self.device, dtype=torch.long)
        x = embed_lookup_batch(self.embed, token_tensor, self.device)
        for layer in self.layers:
            forward_decode_batch = getattr(layer, "forward_decode_batch", None)
            if not callable(forward_decode_batch):
                raise RuntimeError(f"layer {type(layer).__name__} does not support decode batching")
            x = forward_decode_batch(x, states, plan)
        for state in states:
            state.finish_token()
        for state, row_hidden in zip(states, x, strict=True):
            state.last_raw_hidden = row_hidden.contiguous().clone()
        h = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
        logits = linear_any(self.lm_head, h)
        return [row.contiguous() for row in logits]

    @torch.no_grad()
    def _forward_prefill_timestep_batch(
        self,
        input_ids: torch.Tensor,
        row_spans: Sequence[tuple[int, int]],
        states: Sequence[DecodeState],
        *,
        plan: Any,
        return_logits: bool,
    ) -> list[torch.Tensor | None]:
        """Run chunked prefill rows through the batch-state decode kernels.

        This deliberately keeps the state trajectory token-sequential while
        removing cross-request row serialization.  Each timestep batches the
        active request rows as [B, D], and lm_head runs only for rows that finish
        their scheduled prefill span.
        """

        if len(row_spans) != len(states):
            raise ValueError("row_spans length must match states")
        lengths = [int(end) - int(start) for start, end in row_spans]
        if any(length <= 0 for length in lengths):
            raise ValueError("prefill timestep batch requires non-empty rows")

        final_logits: list[torch.Tensor | None] = [None] * len(states)
        max_prefill_steps = max(lengths)
        for step in range(max_prefill_steps):
            active_rows = [row for row, length in enumerate(lengths) if step < length]
            if not active_rows:
                continue
            token_indices = [int(row_spans[row][0]) + step for row in active_rows]
            step_tokens = input_ids[token_indices].to(device=self.device, dtype=torch.long)
            x = embed_lookup_batch(self.embed, step_tokens, self.device)
            active_states = [states[row] for row in active_rows]
            subplan = self._verify_subplan(plan, states, active_rows, step)
            if all(has_canonical_attention_kv(state) for state in active_states):
                # Non-paged states build canonical KV during prefill and publish
                # it once the chunk is complete. Paged-only arena states have
                # zero-length canonical KV by design, so they must stay on the
                # paged hot path instead of falling through to append_attention_kv.
                subplan = replace(subplan, block_tables=None, slot_mapping=None)
            for layer in self.layers:
                forward_decode_batch = getattr(layer, "forward_decode_batch", None)
                if not callable(forward_decode_batch):
                    raise RuntimeError(f"layer {type(layer).__name__} does not support prefill batching")
                x = forward_decode_batch(x, active_states, subplan)
            for row in active_rows:
                states[row].finish_token()
            for local_row, global_row in enumerate(active_rows):
                states[global_row].last_raw_hidden = x[local_row].contiguous().clone()

            if not return_logits:
                continue

            finished_local_rows = [
                local_row
                for local_row, global_row in enumerate(active_rows)
                if step == lengths[global_row] - 1
            ]
            if not finished_local_rows:
                continue
            hidden = x[finished_local_rows].contiguous()
            h = qwen_rmsnorm(hidden, self.final_norm, self.cfg.rms_norm_eps)
            logits = linear_any(self.lm_head, h)
            for out_row, local_row in zip(logits, finished_local_rows, strict=True):
                final_logits[active_rows[local_row]] = out_row.contiguous()

        for row in range(len(states)):
            sync_state_kv_to_paged(states[row], plan, row)
        return final_logits

    @torch.no_grad()
    def _forward_block_raw(
        self,
        tokens: Sequence[int],
        state: DecodeState,
        *,
        hidden_tap_layers: Sequence[int] | None,
        return_logits: bool,
        logits_mode: Literal["all", "last"],
    ) -> BlockForwardResult:
        token_list = [int(t) for t in tokens]
        if not token_list:
            return BlockForwardResult(logits=[], hidden_taps=[], state=state, raw_hiddens=[], final_hiddens=[])
        token_tensor = torch.tensor(token_list, device=self.device, dtype=torch.long)
        x = torch.stack([embed_lookup(self.embed, tid).to(self.device, non_blocking=True) for tid in token_tensor], dim=0).contiguous()
        tap_set = set(hidden_tap_layers or ())
        taps_by_token: list[list[torch.Tensor]] = [[] for _ in token_list]
        base_pos = state.pos
        base_kv_len = state.kv_len
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, Qwen36AttentionLayer):
                x = layer.forward_block(x, state, base_pos=base_pos, base_kv_len=base_kv_len)
            else:
                x = layer.forward_block(x, state)
            if layer_idx in tap_set:
                for i in range(x.size(0)):
                    taps_by_token[i].append(x[i].detach())
        for _ in token_list:
            state.finish_token()
        needs_all_hiddens = logits_mode == "all" or bool(tap_set)
        if needs_all_hiddens:
            raw_hiddens = [row.contiguous().clone() for row in x]
        else:
            raw_hiddens = [x[-1].contiguous().clone()]
        final_hiddens: list[torch.Tensor] = []
        logits_out: list[torch.Tensor] = []
        if return_logits:
            if logits_mode == "last":
                h = qwen_rmsnorm(x[-1].contiguous(), self.final_norm, self.cfg.rms_norm_eps)
                final_hiddens = [h.contiguous().clone()]
                logits_out = [linear_any(self.lm_head, h).contiguous().clone()]
            else:
                h = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
                final_hiddens = [row.contiguous().clone() for row in h]
                logits = linear_any(self.lm_head, h)
                logits_out = [row.contiguous().clone() for row in logits]
        else:
            if needs_all_hiddens:
                h = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
                final_hiddens = [row.contiguous().clone() for row in h]
            else:
                h = qwen_rmsnorm(x[-1].contiguous(), self.final_norm, self.cfg.rms_norm_eps)
                final_hiddens = [h.contiguous().clone()]
        return BlockForwardResult(
            logits=logits_out,
            hidden_taps=taps_by_token,
            state=state,
            raw_hiddens=raw_hiddens,
            final_hiddens=final_hiddens,
        )

    @torch.no_grad()
    def forward_verify_batch(
        self,
        plan: object,
        states: Sequence[DecodeState],
    ) -> list[VerifyBlockResult]:
        """Verify speculative rows through the target-verify state hot path.

        Arena-backed serving rows must not fall back to the ordinary single-row
        block verifier: that bypasses the batch GDN/conv state kernels and the
        paged-attention contract used by normal continuous decode.
        """

        num_requests = int(getattr(plan, "num_requests"))
        if len(states) != num_requests:
            raise ValueError("states length must match plan.num_requests")
        if self._verify_batch_uses_state_hot_path(plan, states):
            return self._forward_verify_batch_uniform_hot(plan, states)
        raise RuntimeError("Qwen36 forward_verify_batch requires arena-backed paged speculative hot path")

    def _build_uniform_spec_plan(self, plan: object, states: Sequence[DecodeState]) -> tuple[DecodeBatchPlan, torch.Tensor, torch.Tensor]:
        row_spans = getattr(plan, "row_spans")
        input_ids = getattr(plan, "input_ids")
        batch = len(states)
        lengths = [int(end) - int(start) for start, end in row_spans]
        if not lengths or any(length <= 0 for length in lengths):
            raise RuntimeError("verify batch requires non-empty rows")
        tokens = max(lengths)
        device = torch.device(self.device)
        token_matrix = torch.empty((batch, tokens), device=device, dtype=torch.long)
        for row, (start, end) in enumerate(row_spans):
            row_tokens = input_ids[int(start) : int(end)].to(device=device, dtype=torch.long)
            token_matrix[row, : row_tokens.numel()] = row_tokens
            if int(row_tokens.numel()) < tokens:
                token_matrix[row, int(row_tokens.numel()) :] = row_tokens[-1]

        slot_mapping = getattr(plan, "slot_mapping", None)
        if slot_mapping is None:
            raise RuntimeError("uniform speculative verifier requires slot_mapping")
        slot_mapping = slot_mapping.to(device=device, dtype=torch.long)
        if slot_mapping.ndim == 2:
            slot_matrix = slot_mapping[:, :tokens].contiguous()
        else:
            slot_matrix = torch.empty((batch, tokens), device=device, dtype=torch.long)
            for row, (start, end) in enumerate(row_spans):
                row_slots = slot_mapping[int(start) : int(end)]
                if int(row_slots.numel()) <= 0:
                    raise RuntimeError("verify row has no KV slots")
                slot_matrix[row, : row_slots.numel()] = row_slots
                if int(row_slots.numel()) < tokens:
                    slot_matrix[row, int(row_slots.numel()) :] = row_slots[-1]
        arena = getattr(states[0], "arena", None) if states else None
        state_indices = getattr(plan, "state_indices", None)
        if arena is None or state_indices is None or not torch.is_tensor(getattr(arena, "pos", None)):
            raise RuntimeError("uniform speculative verifier requires arena device-side position counters")
        base_positions = _arena_positions(arena, state_indices, device=device)
        positions = (base_positions.unsqueeze(1) + torch.arange(tokens, dtype=torch.long, device=device).unsqueeze(0)).reshape(-1).contiguous()
        uniform = replace(
            plan,
            input_ids=token_matrix.reshape(-1).contiguous(),
            positions=positions,
            query_start_loc=torch.arange(0, (batch + 1) * tokens, tokens, dtype=torch.int32, device=device),
            seq_lens=(base_positions + tokens).to(dtype=torch.int32),
            logits_indices=torch.arange(0, batch * tokens, dtype=torch.long, device=device),
            cu_num_logits=torch.arange(0, (batch + 1) * tokens, tokens, dtype=torch.int32, device=device),
            row_spans=tuple((row * tokens, (row + 1) * tokens) for row in range(batch)),
            num_scheduled_tokens=[tokens] * batch,
            slot_mapping=slot_matrix.contiguous(),
        )
        return uniform, token_matrix, torch.tensor(lengths, device=device, dtype=torch.int32)

    def _spec_workspace_for(
        self,
        *,
        arena: object,
        batch: int,
        tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, dict[str, torch.Tensor]]:
        key = (str(device), int(batch), int(tokens), dtype, int(id(arena)))
        cached = self._spec_workspace_cache.get(key)
        if cached is not None:
            return cached
        workspace: dict[int, dict[str, torch.Tensor]] = {}
        conv_states = getattr(arena, "gdn_conv_states", {})
        gdn_states = getattr(arena, "gdn_states", {})
        for layer in self.layers:
            if not isinstance(layer, Qwen36GDNLayer):
                continue
            conv_state = conv_states[layer.layer]
            gdn_state = gdn_states[layer.layer]
            conv_channels = int(conv_state.size(1))
            conv_history = int(conv_state.size(2))
            v_heads = int(gdn_state.size(1))
            head_dim = int(gdn_state.size(2))
            workspace[layer.layer] = {
                "conv_out": torch.empty((batch, tokens, conv_channels), device=device, dtype=dtype),
                "conv_traj": torch.empty((batch, tokens, conv_channels, conv_history), device=device, dtype=dtype),
                "gdn_out": torch.empty((batch, tokens, v_heads, head_dim), device=device, dtype=dtype),
                "gdn_traj": torch.empty((batch, tokens, v_heads, head_dim, head_dim), device=device, dtype=dtype),
            }
        self._spec_workspace_cache[key] = workspace
        return workspace

    def _spec_logit_positions(
        self,
        *,
        draft_counts: Sequence[int],
        tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = tuple(int(v) for v in draft_counts)
        key = (str(device), int(tokens), counts)
        cached = self._spec_index_cache.get(key)
        if cached is not None:
            return cached
        target_positions: list[int] = []
        bonus_positions: list[int] = []
        for row, draft_n in enumerate(counts):
            row_start = row * int(tokens)
            target_positions.extend(row_start + offset for offset in range(draft_n))
            bonus_positions.append(row_start + draft_n)
        target_tensor = torch.tensor(target_positions, dtype=torch.long, device=device)
        bonus_tensor = torch.tensor(bonus_positions, dtype=torch.long, device=device)
        cached = (target_tensor, bonus_tensor)
        self._spec_index_cache[key] = cached
        return cached

    def _forward_speculative_uniform_layers(
        self,
        token_matrix: torch.Tensor,
        states: Sequence[DecodeState],
        plan: DecodeBatchPlan,
        commit_tokens: torch.Tensor,
        trajectory: dict[str, list] | None = None,
    ) -> torch.Tensor:
        x = embed_lookup_batch(self.embed, token_matrix.reshape(-1).contiguous(), self.device)
        x = x.reshape(token_matrix.size(0), token_matrix.size(1), -1).contiguous()
        for layer in self.layers:
            forward_speculative_batch = getattr(layer, "forward_speculative_batch", None)
            if not callable(forward_speculative_batch):
                raise RuntimeError(f"layer {type(layer).__name__} does not support uniform speculative batching")
            x = forward_speculative_batch(x, states, commit_tokens, plan, trajectory=trajectory)
        return x.contiguous()

    def _commit_speculative_trajectory(
        self,
        *,
        trajectory: dict[str, list],
        arena: object,
        state_indices: torch.Tensor,
        commit_tokens: torch.Tensor,
    ) -> None:
        device = commit_tokens.device
        state_idx = state_indices.to(device=device, dtype=torch.long).reshape(-1).contiguous()
        row_index = torch.arange(int(commit_tokens.numel()), device=device, dtype=torch.long)
        last_indices = torch.clamp(commit_tokens.to(device=device, dtype=torch.long) - 1, min=0)
        for layer, conv_traj in trajectory.get("conv", []):
            selected = conv_traj[row_index, last_indices].contiguous()
            getattr(arena, "gdn_conv_states")[layer].index_copy_(0, state_idx, selected)
        for layer, gdn_traj in trajectory.get("gdn", []):
            selected = gdn_traj[row_index, last_indices].contiguous()
            getattr(arena, "gdn_states")[layer].index_copy_(0, state_idx, selected)
        for layer, k_btd, v_btd, slot_mapping in trajectory.get("attn", []):
            append_paged_int4_spec_block(arena, layer, k_btd, v_btd, slot_mapping, commit_tokens)

    def _forward_verify_batch_uniform_hot(
        self,
        plan: object,
        states: Sequence[DecodeState],
    ) -> list[VerifyBlockResult]:
        if (
            self._verify_graph_cache.enabled
            and torch.cuda.is_available()
            and torch.is_tensor(getattr(plan, "input_ids", None))
            and getattr(plan, "input_ids").device.type == "cuda"
        ):
            return self._forward_verify_batch_uniform_hot_graph(plan, states)
        return self._forward_verify_batch_uniform_hot_eager(plan, states)

    def _verify_graph_key(self, plan: object, token_matrix: torch.Tensor) -> CudaGraphKey:
        block_tables = getattr(plan, "block_tables", None)
        slot_mapping = getattr(plan, "slot_mapping", None)
        state_indices = getattr(plan, "state_indices", None)
        seq_lens = getattr(plan, "seq_lens", None)
        context_bucket = 0
        if torch.is_tensor(block_tables):
            context_bucket = int(block_tables.size(1))
        ptrs = tuple(
            int(t.data_ptr())
            for t in (getattr(plan, "input_ids", None), block_tables, slot_mapping, state_indices, seq_lens)
            if torch.is_tensor(t)
        )
        return CudaGraphKey(
            batch_size=int(token_matrix.size(0)),
            query_len=int(token_matrix.size(1)),
            speculative_tokens=max(0, int(token_matrix.size(1)) - 1),
            context_bucket=context_bucket,
            buffer_signature=ptrs,
        )

    def _forward_verify_batch_uniform_hot_graph(
        self,
        plan: object,
        states: Sequence[DecodeState],
    ) -> list[VerifyBlockResult]:
        _uniform_plan, token_matrix, _row_lengths = self._build_uniform_spec_plan(plan, states)
        key = self._verify_graph_key(plan, token_matrix)
        holder: dict[str, list[VerifyBlockResult]] = {}

        def run() -> None:
            holder["results"] = self._forward_verify_batch_uniform_hot_eager(plan, states)

        executable = self._verify_graph_cache.get(key)
        if executable is None:
            try:
                executable = self._verify_graph_cache.capture(key, run)
            except Exception:
                self._verify_graph_cache.disable()
                self._verify_graph_outputs.clear()
                return self._forward_verify_batch_uniform_hot_eager(plan, states)
            self._verify_graph_outputs[key] = holder["results"]
            return self._verify_graph_outputs[key]
        executable.replay()
        return self._verify_graph_outputs[key]

    def _forward_verify_batch_uniform_hot_eager(
        self,
        plan: object,
        states: Sequence[DecodeState],
    ) -> list[VerifyBlockResult]:
        metadata = getattr(plan, "spec_decode_metadata", None)
        if metadata is None:
            raise RuntimeError("uniform speculative verifier requires spec_decode_metadata")
        draft_counts = [int(v) for v in getattr(plan, "num_draft_tokens_per_request")]
        uniform_plan, token_matrix, row_lengths = self._build_uniform_spec_plan(plan, states)
        batch, tokens = token_matrix.shape
        arena = getattr(states[0], "arena", None) if states else None
        state_indices = getattr(plan, "state_indices", None)
        if arena is None or state_indices is None or not hasattr(arena, "advance_slots"):
            raise RuntimeError("uniform speculative verifier requires arena device-side state counters")
        zero_commit = torch.zeros((batch,), dtype=torch.int32, device=token_matrix.device)
        trajectory: dict[str, object] = {
            "conv": [],
            "gdn": [],
            "attn": [],
            "workspace": self._spec_workspace_for(
                arena=arena,
                batch=batch,
                tokens=tokens,
                device=token_matrix.device,
                dtype=torch.float16,
            ),
        }

        # Single target pass: compute verifier logits and save per-token state
        # trajectory.  The live arena is untouched until reducer-selected
        # commit_tokens are known below.
        probe_hidden = self._forward_speculative_uniform_layers(
            token_matrix,
            states,
            uniform_plan,
            zero_commit,
            trajectory=trajectory,
        )
        probe_norm = qwen_rmsnorm(probe_hidden.reshape(batch * tokens, -1).contiguous(), self.final_norm, self.cfg.rms_norm_eps)
        flat_probe_logits = linear_any(self.lm_head, probe_norm).contiguous()
        probe_logits = flat_probe_logits.reshape(batch, tokens, -1)

        target_position_tensor, bonus_position_tensor = self._spec_logit_positions(
            draft_counts=draft_counts,
            tokens=tokens,
            device=token_matrix.device,
        )
        if int(target_position_tensor.numel()) > 0:
            target_logits_tensor = flat_probe_logits.index_select(0, target_position_tensor).contiguous()
            if target_logits_tensor.device.type == "cuda":
                target_ids_flat = cuda_ops().argmax_many(target_logits_tensor).to(device=token_matrix.device, dtype=torch.long)
            else:
                target_ids_flat = torch.argmax(target_logits_tensor, dim=-1).to(device=token_matrix.device, dtype=torch.long)
        else:
            target_ids_flat = torch.empty((0,), device=token_matrix.device, dtype=torch.long)
        bonus_logits_tensor = flat_probe_logits.index_select(0, bonus_position_tensor).contiguous()
        if bonus_logits_tensor.device.type == "cuda":
            bonus_ids = cuda_ops().argmax_many(bonus_logits_tensor).to(device=token_matrix.device, dtype=torch.long)
        else:
            bonus_ids = torch.argmax(bonus_logits_tensor, dim=-1).to(device=token_matrix.device, dtype=torch.long)
        resolved = resolve_speculative_gpu(
            metadata,
            target_token_ids=target_ids_flat,
            bonus_token_ids=bonus_ids,
            scheduled_token_counts=row_lengths,
        )
        commit_tokens = resolved.commit_tokens.to(device=token_matrix.device, dtype=torch.int32).contiguous()

        state_indices_tensor = state_indices.to(device=token_matrix.device, dtype=torch.long).reshape(-1).contiguous()
        self._commit_speculative_trajectory(
            trajectory=trajectory,
            arena=arena,
            state_indices=state_indices_tensor,
            commit_tokens=commit_tokens,
        )
        arena.advance_slots(state_indices_tensor, commit_tokens)
        row_index = torch.arange(batch, device=token_matrix.device, dtype=torch.long)
        last_indices = torch.clamp(commit_tokens.to(device=token_matrix.device, dtype=torch.long) - 1, min=0)
        selected_hidden = probe_norm.reshape(batch, tokens, -1).contiguous()[row_index, last_indices].contiguous()
        selected_logits = probe_logits[row_index, last_indices].contiguous()
        selected_raw_hidden = probe_hidden[row_index, last_indices].contiguous()
        results: list[VerifyBlockResult] = []
        target_offset = 0
        for row, state in enumerate(states):
            draft_n = draft_counts[row]
            row_target_ids = target_ids_flat[target_offset : target_offset + draft_n].contiguous()
            target_offset += draft_n
            state.last_raw_hidden = selected_raw_hidden[row].contiguous().clone()
            results.append(
                VerifyBlockResult(
                    target_ids=row_target_ids,
                    logits=selected_logits[row].contiguous(),
                    hidden=selected_hidden[row].contiguous(),
                    state=state,
                    state_already_committed=True,
                    speculative_decision=resolved,
                )
            )
        return results

    def _verify_batch_uses_state_hot_path(self, plan: object, states: Sequence[DecodeState]) -> bool:
        if not states or _arena_batch(states) is None:
            return False
        if getattr(plan, "state_indices", None) is None:
            return False
        # Paged-KV tensors are the production verifier contract.  Without them
        # attention would silently fall back to per-row logical views, which is
        # useful for tests but not the vLLM-style hot path.
        return getattr(plan, "block_tables", None) is not None and getattr(plan, "slot_mapping", None) is not None

    def _verify_subplan(
        self,
        parent_plan: object,
        states: Sequence[DecodeState],
        active_rows: Sequence[int],
        step: int,
    ) -> DecodeBatchPlan:
        n = len(active_rows)
        device = torch.device(self.device)
        positions = torch.tensor([int(states[row].pos) for row in active_rows], dtype=torch.long, device=device)
        state_indices = getattr(parent_plan, "state_indices", None)
        if state_indices is not None:
            sub_state_indices = state_indices.to(device=device, dtype=torch.int32)[list(active_rows)].contiguous()
        else:
            sub_state_indices = torch.tensor([int(getattr(states[row], "arena_slot", row) or row) for row in active_rows], dtype=torch.int32, device=device)
        block_tables = getattr(parent_plan, "block_tables", None)
        sub_block_tables = block_tables.to(device=device, dtype=torch.int32)[list(active_rows)].contiguous() if block_tables is not None else None
        slot_mapping = getattr(parent_plan, "slot_mapping", None)
        sub_slot_mapping = None
        if slot_mapping is not None:
            flat_indices = [int(parent_plan.row_spans[row][0]) + int(step) for row in active_rows]
            sub_slot_mapping = slot_mapping.to(device=device, dtype=torch.long)[flat_indices].contiguous()
        return DecodeBatchPlan(
            request_ids=[str(getattr(parent_plan, "request_ids", [row])[row]) for row in active_rows],
            state_indices=sub_state_indices,
            input_ids=torch.zeros((n,), dtype=torch.long, device=device),
            positions=positions,
            query_start_loc=torch.arange(0, n + 1, dtype=torch.int32, device=device),
            seq_lens=(positions + 1).to(dtype=torch.int32),
            logits_indices=torch.arange(0, n, dtype=torch.long, device=device),
            cu_num_logits=torch.arange(0, n + 1, dtype=torch.int32, device=device),
            row_spans=tuple((i, i + 1) for i in range(n)),
            num_scheduled_tokens=[1] * n,
            num_draft_tokens_per_request=[0] * n,
            is_prefill=[False] * n,
            block_tables=sub_block_tables,
            slot_mapping=sub_slot_mapping,
        )
