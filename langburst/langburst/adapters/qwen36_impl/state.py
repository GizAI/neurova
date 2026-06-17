from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal
import os
import time

import torch

from .config import Qwen36_27B_TextConfig
from langburst.core.kv_cache import (
    KVCacheLayout,
    KVCacheSpec,
    KVCacheTensors,
    allocate_kv_cache_tensors,
    kv_buffer,
    hadamard_transform,
    pack_int4_rows,
    unpack_int4_rows,
)
from langburst.tuning import int4_kv_layout

KVWindowPolicy = Literal["error", "shift", "ring"]


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "off", "no"}


@dataclass
class StateSnapshotInfo:
    """Small JSON-like metadata stored next to tensor state.

    The snapshot intentionally records the engine/runtime contract rather than
    model weights. A snapshot is only valid for the same model family, quantized
    checkpoint, layer layout, and tokenizer position policy.
    """

    schema_version: int
    created_unix: float
    model_family: str
    pos: int
    max_seq_len: int
    kv_len: int
    kv_window_policy: str
    kv_cache_dtype: str
    dtype: str
    include_attention_kv: bool
    include_conv_state: bool


@dataclass
class DecodeStateWriteSnapshot:
    gdn_states: dict[int, torch.Tensor]
    gdn_conv_states: dict[int, torch.Tensor]
    attn_k_rows: dict[int, tuple[list[int], torch.Tensor]]
    attn_v_rows: dict[int, tuple[list[int], torch.Tensor]]
    attn_k_scale_rows: dict[int, tuple[list[int], torch.Tensor]]
    attn_v_scale_rows: dict[int, tuple[list[int], torch.Tensor]]
    attn_k_zero_rows: dict[int, tuple[list[int], torch.Tensor]]
    attn_v_zero_rows: dict[int, tuple[list[int], torch.Tensor]]
    pos: int
    kv_len: int

    def restore_(self, state: "DecodeState") -> None:
        for layer, tensor in self.gdn_states.items():
            state.gdn_states[layer].copy_(tensor)
        for layer, tensor in self.gdn_conv_states.items():
            state.gdn_conv_states[layer].copy_(tensor)
        for layer, (indices, values) in self.attn_k_rows.items():
            for row, idx in enumerate(indices):
                state.attn_k[layer][:, idx, :].copy_(values[:, row, :])
        for layer, (indices, values) in self.attn_v_rows.items():
            for row, idx in enumerate(indices):
                state.attn_v[layer][:, idx, :].copy_(values[:, row, :])
        if state.attn_k_scale is not None:
            for layer, (indices, values) in self.attn_k_scale_rows.items():
                for row, idx in enumerate(indices):
                    state.attn_k_scale[layer][:, idx].copy_(values[:, row])
        if state.attn_v_scale is not None:
            for layer, (indices, values) in self.attn_v_scale_rows.items():
                for row, idx in enumerate(indices):
                    state.attn_v_scale[layer][:, idx].copy_(values[:, row])
        if state.attn_k_zero is not None:
            for layer, (indices, values) in self.attn_k_zero_rows.items():
                for row, idx in enumerate(indices):
                    state.attn_k_zero[layer][:, idx].copy_(values[:, row])
        if state.attn_v_zero is not None:
            for layer, (indices, values) in self.attn_v_zero_rows.items():
                for row, idx in enumerate(indices):
                    state.attn_v_zero[layer][:, idx].copy_(values[:, row])
        state.pos = self.pos
        state.kv_len = self.kv_len
        pos_tensor = getattr(state, "pos_tensor", None)
        if torch.is_tensor(pos_tensor):
            pos_tensor.fill_(int(self.pos))
        kv_len_tensor = getattr(state, "kv_len_tensor", None)
        if torch.is_tensor(kv_len_tensor):
            kv_len_tensor.fill_(int(self.kv_len))


@dataclass
class DecodeState:
    """Runtime state for stateful / streaming text generation.

    For Qwen3.6-27B the state has three pieces:
      1. GDN recurrent matrices: fixed-size compressed long memory.
      2. GDN depthwise-conv buffers: required for exact recurrent streaming.
      3. Full-attention KV cache: recent exact memory, finite and windowed.

    Keeping conv buffers here, instead of inside layer objects, is what makes
    snapshots, state forking, branch search, and infinite streaming possible.
    """

    cfg: Qwen36_27B_TextConfig
    gdn_states: dict[int, torch.Tensor]
    gdn_conv_states: dict[int, torch.Tensor]
    attn_k: dict[int, torch.Tensor]
    attn_v: dict[int, torch.Tensor]
    attn_k_scale: dict[int, torch.Tensor] | None = None
    attn_v_scale: dict[int, torch.Tensor] | None = None
    attn_k_zero: dict[int, torch.Tensor] | None = None
    attn_v_zero: dict[int, torch.Tensor] | None = None
    pos: int = 0
    max_seq_len: int = 0
    kv_len: int = 0
    kv_window_policy: KVWindowPolicy = "error"
    kv_cache_spec: KVCacheSpec = field(default_factory=KVCacheSpec)

    @staticmethod
    def kv_layout(cfg: Qwen36_27B_TextConfig) -> KVCacheLayout:
        return KVCacheLayout.from_parts(
            layers=cfg.attention_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.attention_head_dim,
        )

    @classmethod
    def allocate(
        cls,
        cfg: Qwen36_27B_TextConfig,
        max_seq_len: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        kv_window_policy: KVWindowPolicy = "error",
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> "DecodeState":
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        device = torch.device(device)
        kv_cache_spec = kv_cache_spec or KVCacheSpec()
        conv_dim = (
            cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        )
        gdn_states = {
            layer: torch.zeros(
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
                device=device,
                dtype=dtype,
            )
            for layer in cfg.gdn_layers
        }
        gdn_conv_states = {
            layer: torch.zeros(
                conv_dim,
                cfg.linear_conv_kernel_dim - 1,
                device=device,
                dtype=dtype,
            )
            for layer in cfg.gdn_layers
        }
        kv = allocate_kv_cache_tensors(cls.kv_layout(cfg), kv_cache_spec, seq_len=max_seq_len, device=device)
        return cls(
            cfg=cfg,
            gdn_states=gdn_states,
            gdn_conv_states=gdn_conv_states,
            attn_k=kv.k,
            attn_v=kv.v,
            attn_k_scale=kv.k_scale,
            attn_v_scale=kv.v_scale,
            attn_k_zero=kv.k_zero,
            attn_v_zero=kv.v_zero,
            pos=0,
            max_seq_len=max_seq_len,
            kv_len=0,
            kv_window_policy=kv_window_policy,
            kv_cache_spec=kv_cache_spec,
        )

    @property
    def device(self) -> torch.device:
        # There is always at least one GDN layer in Qwen3.6 text config.
        return next(iter(self.gdn_states.values())).device

    @property
    def dtype(self) -> torch.dtype:
        return next(iter(self.gdn_states.values())).dtype

    @property
    def gdn_state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.gdn_states.values())

    @property
    def conv_state_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.gdn_conv_states.values())

    @property
    def attention_kv_bytes(self) -> int:
        total = sum(t.numel() * t.element_size() for t in self.attn_k.values()) + sum(
            t.numel() * t.element_size() for t in self.attn_v.values()
        )
        total += sum(t.numel() * t.element_size() for t in (self.attn_k_scale or {}).values())
        total += sum(t.numel() * t.element_size() for t in (self.attn_v_scale or {}).values())
        total += sum(t.numel() * t.element_size() for t in (self.attn_k_zero or {}).values())
        total += sum(t.numel() * t.element_size() for t in (self.attn_v_zero or {}).values())
        return total

    @property
    def total_bytes(self) -> int:
        return self.gdn_state_bytes + self.conv_state_bytes + self.attention_kv_bytes

    def reset(self, *, reset_attention: bool = True) -> None:
        for tensor in self.gdn_states.values():
            tensor.zero_()
        for tensor in self.gdn_conv_states.values():
            tensor.zero_()
        if reset_attention:
            self.kv_len = 0
            kv_len_tensor = getattr(self, "kv_len_tensor", None)
            if torch.is_tensor(kv_len_tensor):
                kv_len_tensor.zero_()
        self.pos = 0
        pos_tensor = getattr(self, "pos_tensor", None)
        if torch.is_tensor(pos_tensor):
            pos_tensor.zero_()

    def decay_gdn_(self, factor: float) -> None:
        """In-place soft reset of compressed long memory.

        factor=1 keeps memory, factor=0 erases it. This is an engine-level
        hygiene control for document/task boundaries; it does not change weights.
        """
        if not 0.0 <= factor <= 1.0:
            raise ValueError("factor must be in [0, 1]")
        for tensor in self.gdn_states.values():
            tensor.mul_(factor)
        # Conv buffers are short-range exact state; decay them too at boundaries.
        for tensor in self.gdn_conv_states.values():
            tensor.mul_(factor)

    def fork(self, *, clone_attention: bool = True) -> "DecodeState":
        """Create an independent branch state for block or beam-style evaluation."""
        return DecodeState(
            cfg=self.cfg,
            gdn_states={k: v.clone() for k, v in self.gdn_states.items()},
            gdn_conv_states={k: v.clone() for k, v in self.gdn_conv_states.items()},
            attn_k={k: v.clone() for k, v in self.attn_k.items()} if clone_attention else self.attn_k.copy(),
            attn_v={k: v.clone() for k, v in self.attn_v.items()} if clone_attention else self.attn_v.copy(),
            attn_k_scale=(
                {k: v.clone() for k, v in self.attn_k_scale.items()}
                if clone_attention and self.attn_k_scale is not None
                else (self.attn_k_scale.copy() if self.attn_k_scale is not None else None)
            ),
            attn_v_scale=(
                {k: v.clone() for k, v in self.attn_v_scale.items()}
                if clone_attention and self.attn_v_scale is not None
                else (self.attn_v_scale.copy() if self.attn_v_scale is not None else None)
            ),
            attn_k_zero=(
                {k: v.clone() for k, v in self.attn_k_zero.items()}
                if clone_attention and self.attn_k_zero is not None
                else (self.attn_k_zero.copy() if self.attn_k_zero is not None else None)
            ),
            attn_v_zero=(
                {k: v.clone() for k, v in self.attn_v_zero.items()}
                if clone_attention and self.attn_v_zero is not None
                else (self.attn_v_zero.copy() if self.attn_v_zero is not None else None)
            ),
            pos=self.pos,
            max_seq_len=self.max_seq_len,
            kv_len=self.kv_len,
            kv_window_policy=self.kv_window_policy,
            kv_cache_spec=self.kv_cache_spec,
        )

    def copy_from_(self, other: "DecodeState", *, copy_attention: bool = True) -> None:
        """Replace this state with another state under the same allocation contract."""
        if (
            self.max_seq_len != other.max_seq_len
            or self.kv_window_policy != other.kv_window_policy
            or self.kv_cache_spec != other.kv_cache_spec
        ):
            raise ValueError("cannot copy DecodeState with different KV allocation contract")
        for k, v in other.gdn_states.items():
            self.gdn_states[k].copy_(v)
        for k, v in other.gdn_conv_states.items():
            self.gdn_conv_states[k].copy_(v)
        if copy_attention:
            for k, v in other.attn_k.items():
                self.attn_k[k].copy_(v)
            for k, v in other.attn_v.items():
                self.attn_v[k].copy_(v)
            if self.attn_k_scale is not None and other.attn_k_scale is not None:
                for k, v in other.attn_k_scale.items():
                    self.attn_k_scale[k].copy_(v)
            if self.attn_v_scale is not None and other.attn_v_scale is not None:
                for k, v in other.attn_v_scale.items():
                    self.attn_v_scale[k].copy_(v)
            if self.attn_k_zero is not None and other.attn_k_zero is not None:
                for k, v in other.attn_k_zero.items():
                    self.attn_k_zero[k].copy_(v)
            if self.attn_v_zero is not None and other.attn_v_zero is not None:
                for k, v in other.attn_v_zero.items():
                    self.attn_v_zero[k].copy_(v)
        self.pos = other.pos
        self.kv_len = other.kv_len
        pos_tensor = getattr(self, "pos_tensor", None)
        other_pos_tensor = getattr(other, "pos_tensor", None)
        if torch.is_tensor(pos_tensor):
            if torch.is_tensor(other_pos_tensor):
                pos_tensor.copy_(other_pos_tensor.to(device=pos_tensor.device, dtype=pos_tensor.dtype))
            else:
                pos_tensor.fill_(int(other.pos))
        kv_len_tensor = getattr(self, "kv_len_tensor", None)
        other_kv_len_tensor = getattr(other, "kv_len_tensor", None)
        if torch.is_tensor(kv_len_tensor):
            if torch.is_tensor(other_kv_len_tensor):
                kv_len_tensor.copy_(other_kv_len_tensor.to(device=kv_len_tensor.device, dtype=kv_len_tensor.dtype))
            else:
                kv_len_tensor.fill_(int(other.kv_len))

    def speculative_write_snapshot(self, num_tokens: int) -> DecodeStateWriteSnapshot:
        """Snapshot only state locations a speculative block may overwrite.

        This is the langburst analogue of the reference runtime's lookahead slot safety: a block
        verifier may write candidate state into live buffers, then either keep it
        on full accept or restore this snapshot on rejection.
        """
        if num_tokens < 0:
            raise ValueError("num_tokens must be >= 0")
        indices: list[int] = []
        for offset in range(num_tokens):
            logical_pos = self.pos + offset
            if self.kv_window_policy == "ring":
                idx = logical_pos % self.max_seq_len
            elif logical_pos < self.max_seq_len:
                idx = logical_pos
            elif self.kv_window_policy == "error":
                raise RuntimeError(f"attention KV window is full at {self.max_seq_len} tokens")
            else:
                # The shift policy mutates the whole cache when full; it is a
                # correctness fallback, not a safe transactional verifier mode.
                raise RuntimeError("shift KV policy does not support speculative transaction snapshots")
            indices.append(idx)
        unique_indices = list(dict.fromkeys(indices))
        return DecodeStateWriteSnapshot(
            gdn_states={k: v.clone() for k, v in self.gdn_states.items()},
            gdn_conv_states={k: v.clone() for k, v in self.gdn_conv_states.items()},
            attn_k_rows={
                layer: (unique_indices, tensor[:, unique_indices, :].detach().clone())
                for layer, tensor in self.attn_k.items()
            },
            attn_v_rows={
                layer: (unique_indices, tensor[:, unique_indices, :].detach().clone())
                for layer, tensor in self.attn_v.items()
            },
            attn_k_scale_rows={
                layer: (unique_indices, tensor[:, unique_indices].detach().clone())
                for layer, tensor in (self.attn_k_scale or {}).items()
            },
            attn_v_scale_rows={
                layer: (unique_indices, tensor[:, unique_indices].detach().clone())
                for layer, tensor in (self.attn_v_scale or {}).items()
            },
            attn_k_zero_rows={
                layer: (unique_indices, tensor[:, unique_indices].detach().clone())
                for layer, tensor in (self.attn_k_zero or {}).items()
            },
            attn_v_zero_rows={
                layer: (unique_indices, tensor[:, unique_indices].detach().clone())
                for layer, tensor in (self.attn_v_zero or {}).items()
            },
            pos=self.pos,
            kv_len=self.kv_len,
        )

    def attention_write_index(self) -> int:
        """Return physical KV index for the next token.

        `shift` is a correctness fallback that physically moves memory. `ring` is
        the final runtime contract: write by logical position modulo window size
        and let the attention kernel consume logical order.  The current Python
        `attention_kv_view()` materializes ordered cache for baseline kernels;
        the CUDA hot path should avoid that copy.
        """
        if self.kv_len < self.max_seq_len:
            return self.kv_len
        if self.kv_window_policy == "error":
            raise RuntimeError(
                f"attention KV window is full at {self.max_seq_len} tokens; "
                "use kv_window_policy='ring' for streaming or 'shift' for fallback tests"
            )
        if self.kv_window_policy == "ring":
            return self.pos % self.max_seq_len
        return self.max_seq_len - 1

    def append_attention_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> int:
        idx = self.attention_write_index()
        if self.kv_len >= self.max_seq_len and self.kv_window_policy == "shift":
            # Correct but not fast. Prefer ring for infinite streaming.
            self.attn_k[layer][:, :-1, :].copy_(self.attn_k[layer][:, 1:, :])
            self.attn_v[layer][:, :-1, :].copy_(self.attn_v[layer][:, 1:, :])
            if self.attn_k_scale is not None and self.attn_v_scale is not None:
                self.attn_k_scale[layer][:, :-1].copy_(self.attn_k_scale[layer][:, 1:])
                self.attn_v_scale[layer][:, :-1].copy_(self.attn_v_scale[layer][:, 1:])
            if self.attn_k_zero is not None and self.attn_v_zero is not None:
                self.attn_k_zero[layer][:, :-1].copy_(self.attn_k_zero[layer][:, 1:])
                self.attn_v_zero[layer][:, :-1].copy_(self.attn_v_zero[layer][:, 1:])
        if self.kv_cache_spec.is_int4:
            k_store = hadamard_transform(k, self.kv_cache_spec.hadamard_order) if self.kv_cache_spec.uses_bdr else k
            v_store = (
                hadamard_transform(v, self.kv_cache_spec.hadamard_order)
                if self.kv_cache_spec.uses_bdr and self.kv_cache_spec.rotate_v
                else v
            )
            k_packed, k_scale, k_zero = pack_int4_rows(k_store)
            v_packed, v_scale, v_zero = pack_int4_rows(v_store)
            self.attn_k[layer][:, idx, :] = k_packed
            self.attn_v[layer][:, idx, :] = v_packed
            if self.attn_k_scale is not None and self.attn_v_scale is not None:
                self.attn_k_scale[layer][:, idx] = k_scale
                self.attn_v_scale[layer][:, idx] = v_scale
            if self.attn_k_zero is not None and self.attn_v_zero is not None:
                self.attn_k_zero[layer][:, idx] = k_zero
                self.attn_v_zero[layer][:, idx] = v_zero
        else:
            self.attn_k[layer][:, idx, :] = k
            self.attn_v[layer][:, idx, :] = v
        return idx

    def append_attention_kv_at(self, layer: int, k: torch.Tensor, v: torch.Tensor, *, logical_pos: int) -> int:
        """Write KV for block verification without mutating global position."""
        if self.kv_window_policy == "ring":
            idx = logical_pos % self.max_seq_len
        elif logical_pos < self.max_seq_len:
            idx = logical_pos
        elif self.kv_window_policy == "error":
            raise RuntimeError(f"attention KV window is full at {self.max_seq_len} tokens")
        else:
            idx = self.max_seq_len - 1
        if self.kv_cache_spec.is_int4:
            k_store = hadamard_transform(k, self.kv_cache_spec.hadamard_order) if self.kv_cache_spec.uses_bdr else k
            v_store = (
                hadamard_transform(v, self.kv_cache_spec.hadamard_order)
                if self.kv_cache_spec.uses_bdr and self.kv_cache_spec.rotate_v
                else v
            )
            k_packed, k_scale, k_zero = pack_int4_rows(k_store)
            v_packed, v_scale, v_zero = pack_int4_rows(v_store)
            self.attn_k[layer][:, idx, :] = k_packed
            self.attn_v[layer][:, idx, :] = v_packed
            if self.attn_k_scale is not None and self.attn_v_scale is not None:
                self.attn_k_scale[layer][:, idx] = k_scale
                self.attn_v_scale[layer][:, idx] = v_scale
            if self.attn_k_zero is not None and self.attn_v_zero is not None:
                self.attn_k_zero[layer][:, idx] = k_zero
                self.attn_v_zero[layer][:, idx] = v_zero
        else:
            self.attn_k[layer][:, idx, :] = k
            self.attn_v[layer][:, idx, :] = v
        return idx

    def append_attention_kv_block_at(
        self,
        layer: int,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        start_logical_pos: int,
    ) -> None:
        """Write a contiguous prefill KV block without mutating global counters."""

        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("block KV tensors must be [tokens, heads, head_dim]")
        if k.size(0) != v.size(0):
            raise ValueError("K and V block token counts must match")
        num_tokens = int(k.size(0))
        if num_tokens == 0:
            return
        logical = torch.arange(
            int(start_logical_pos),
            int(start_logical_pos) + num_tokens,
            device=k.device,
            dtype=torch.long,
        )
        if self.kv_window_policy == "ring":
            indices = torch.remainder(logical, self.max_seq_len)
        elif int(start_logical_pos) + num_tokens <= self.max_seq_len:
            indices = logical
        elif self.kv_window_policy == "error":
            raise RuntimeError(f"attention KV window is full at {self.max_seq_len} tokens")
        else:
            indices = torch.arange(
                self.max_seq_len - num_tokens,
                self.max_seq_len,
                device=k.device,
                dtype=torch.long,
            )

        if self.kv_cache_spec.is_int4:
            k_store = hadamard_transform(k, self.kv_cache_spec.hadamard_order) if self.kv_cache_spec.uses_bdr else k
            v_store = (
                hadamard_transform(v, self.kv_cache_spec.hadamard_order)
                if self.kv_cache_spec.uses_bdr and self.kv_cache_spec.rotate_v
                else v
            )
            tokens, k_heads, k_dim = k_store.shape
            _, v_heads, v_dim = v_store.shape
            k_packed, k_scale, k_zero = pack_int4_rows(k_store.reshape(tokens * k_heads, k_dim).contiguous())
            v_packed, v_scale, v_zero = pack_int4_rows(v_store.reshape(tokens * v_heads, v_dim).contiguous())
            self.attn_k[layer][:, indices, :] = k_packed.view(tokens, k_heads, -1).permute(1, 0, 2).contiguous()
            self.attn_v[layer][:, indices, :] = v_packed.view(tokens, v_heads, -1).permute(1, 0, 2).contiguous()
            if self.attn_k_scale is not None and self.attn_v_scale is not None:
                self.attn_k_scale[layer][:, indices] = k_scale.view(tokens, k_heads).transpose(0, 1).contiguous()
                self.attn_v_scale[layer][:, indices] = v_scale.view(tokens, v_heads).transpose(0, 1).contiguous()
            if self.attn_k_zero is not None and self.attn_v_zero is not None:
                self.attn_k_zero[layer][:, indices] = k_zero.view(tokens, k_heads).transpose(0, 1).contiguous()
                self.attn_v_zero[layer][:, indices] = v_zero.view(tokens, v_heads).transpose(0, 1).contiguous()
        else:
            self.attn_k[layer][:, indices, :] = k.permute(1, 0, 2).contiguous()
            self.attn_v[layer][:, indices, :] = v.permute(1, 0, 2).contiguous()

    def _dequant_attention_view(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.kv_cache_spec.is_int4:
            return k, v
        if self.attn_k_scale is None or self.attn_v_scale is None or self.attn_k_zero is None or self.attn_v_zero is None:
            raise RuntimeError("INT4 KV cache is missing token-wise scale/zero tensors")
        k_scale = self.attn_k_scale[layer]
        v_scale = self.attn_v_scale[layer]
        k_zero = self.attn_k_zero[layer]
        v_zero = self.attn_v_zero[layer]
        start = (self.pos + 1) % self.max_seq_len
        if start != 0 and self.kv_window_policy == "ring" and self.kv_len >= self.max_seq_len:
            k_scale = torch.cat([k_scale[:, start:], k_scale[:, :start]], dim=1)
            v_scale = torch.cat([v_scale[:, start:], v_scale[:, :start]], dim=1)
            k_zero = torch.cat([k_zero[:, start:], k_zero[:, :start]], dim=1)
            v_zero = torch.cat([v_zero[:, start:], v_zero[:, :start]], dim=1)
        k_fp = unpack_int4_rows(k, k_scale[:, : k.size(1)], k_zero[:, : k.size(1)], head_dim=self.cfg.attention_head_dim)
        v_fp = unpack_int4_rows(v, v_scale[:, : v.size(1)], v_zero[:, : v.size(1)], head_dim=self.cfg.attention_head_dim)
        if self.kv_cache_spec.uses_bdr:
            k_fp = hadamard_transform(k_fp, self.kv_cache_spec.hadamard_order)
            if self.kv_cache_spec.rotate_v:
                v_fp = hadamard_transform(v_fp, self.kv_cache_spec.hadamard_order)
        return k_fp, v_fp

    def attention_kv_view(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return KV in logical oldest→newest order for baseline attention.

        For `ring` this may allocate a concatenated view.  That is acceptable for
        correctness and first-real-chat; production CUDA should consume physical
        ring indices directly.
        """
        live = min(self.kv_len + 1, self.max_seq_len)
        if self.kv_window_policy != "ring" or self.kv_len < self.max_seq_len:
            k, v = self.attn_k[layer], self.attn_v[layer]
            k, v = self._dequant_attention_view(layer, k, v)
            return k, v, live
        start = (self.pos + 1) % self.max_seq_len
        if start == 0:
            k, v = self.attn_k[layer], self.attn_v[layer]
            k, v = self._dequant_attention_view(layer, k, v)
            return k, v, live
        k = torch.cat([self.attn_k[layer][:, start:, :], self.attn_k[layer][:, :start, :]], dim=1)
        v = torch.cat([self.attn_v[layer][:, start:, :], self.attn_v[layer][:, :start, :]], dim=1)
        k, v = self._dequant_attention_view(layer, k, v)
        return k, v, live

    def attention_kv_view_at(self, layer: int, *, logical_pos: int, live_len: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return KV view after a block-local write at `logical_pos`."""
        live = min(live_len, self.max_seq_len)
        if self.kv_window_policy != "ring" or live < self.max_seq_len:
            k, v = self.attn_k[layer], self.attn_v[layer]
            k, v = self._dequant_attention_view(layer, k, v)
            return k, v, live
        start = (logical_pos + 1) % self.max_seq_len
        if start == 0:
            k, v = self.attn_k[layer], self.attn_v[layer]
            k, v = self._dequant_attention_view(layer, k, v)
            return k, v, live
        k = torch.cat([self.attn_k[layer][:, start:, :], self.attn_k[layer][:, :start, :]], dim=1)
        v = torch.cat([self.attn_v[layer][:, start:, :], self.attn_v[layer][:, :start, :]], dim=1)
        k, v = self._dequant_attention_view(layer, k, v)
        return k, v, live

    def finish_token(self) -> None:
        self.finish_tokens(1)

    def finish_tokens(self, count: int) -> None:
        count = int(count)
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return
        self.pos += count
        self.kv_len = min(self.kv_len + count, self.max_seq_len)
        pos_tensor = getattr(self, "pos_tensor", None)
        if torch.is_tensor(pos_tensor):
            pos_tensor.add_(count)
        kv_len_tensor = getattr(self, "kv_len_tensor", None)
        if torch.is_tensor(kv_len_tensor):
            kv_len_tensor.add_(count).clamp_(max=self.max_seq_len)

    def sync_metadata_from_device_(self) -> None:
        """Refresh Python metadata from arena device counters.

        This is intentionally a cold-path helper for diagnostics, snapshots,
        and request cleanup. CUDA-graphable decode/verify paths must consume
        `pos_tensor`/`kv_len_tensor` directly instead of calling this.
        """

        pos_tensor = getattr(self, "pos_tensor", None)
        if torch.is_tensor(pos_tensor):
            self.pos = int(pos_tensor.detach().cpu().item())
        kv_len_tensor = getattr(self, "kv_len_tensor", None)
        if torch.is_tensor(kv_len_tensor):
            self.kv_len = int(kv_len_tensor.detach().cpu().item())

    def snapshot_dict(self, *, include_attention_kv: bool = True) -> dict[str, Any]:
        info = StateSnapshotInfo(
            schema_version=3,
            created_unix=time.time(),
            model_family="qwen3.6-27b-text",
            pos=self.pos,
            max_seq_len=self.max_seq_len,
            kv_len=self.kv_len,
            kv_window_policy=self.kv_window_policy,
            kv_cache_dtype=self.kv_cache_spec.dtype,
            dtype=str(self.dtype).replace("torch.", ""),
            include_attention_kv=include_attention_kv,
            include_conv_state=True,
        )
        payload: dict[str, Any] = {
            "info": asdict(info),
            "gdn_states": {k: v.detach().cpu() for k, v in self.gdn_states.items()},
            "gdn_conv_states": {k: v.detach().cpu() for k, v in self.gdn_conv_states.items()},
        }
        if include_attention_kv:
            # Store the physical ring buffers, not only a prefix. Once the ring
            # has wrapped, the live logical window is split across the end and
            # beginning of the allocation; saving a prefix would corrupt
            # warm-boot continuation.
            payload["attn_k"] = {k: v.detach().cpu() for k, v in self.attn_k.items()}
            payload["attn_v"] = {k: v.detach().cpu() for k, v in self.attn_v.items()}
            if self.attn_k_scale is not None:
                payload["attn_k_scale"] = {k: v.detach().cpu() for k, v in self.attn_k_scale.items()}
            if self.attn_v_scale is not None:
                payload["attn_v_scale"] = {k: v.detach().cpu() for k, v in self.attn_v_scale.items()}
            if self.attn_k_zero is not None:
                payload["attn_k_zero"] = {k: v.detach().cpu() for k, v in self.attn_k_zero.items()}
            if self.attn_v_zero is not None:
                payload["attn_v_zero"] = {k: v.detach().cpu() for k, v in self.attn_v_zero.items()}
        return payload

    def save_snapshot(self, path: str | Path, *, include_attention_kv: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.snapshot_dict(include_attention_kv=include_attention_kv), path)

    @classmethod
    def load_snapshot(
        cls,
        path: str | Path,
        cfg: Qwen36_27B_TextConfig | None = None,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        max_seq_len: int | None = None,
    ) -> "DecodeState":
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        info = payload["info"]
        if int(info.get("schema_version", 0)) != 3:
            raise ValueError(f"unsupported snapshot schema: {info.get('schema_version')}")
        cfg = cfg or Qwen36_27B_TextConfig()
        dtype = dtype or getattr(torch, str(info.get("dtype", "float16")))
        max_seq_len = int(max_seq_len or info["max_seq_len"])
        state = cls.allocate(
            cfg,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            kv_window_policy=info.get("kv_window_policy", "error"),
            kv_cache_spec=KVCacheSpec.resolve(info.get("kv_cache_dtype", "fp16")),
        )
        state.pos = int(info["pos"])
        state.kv_len = min(int(info["kv_len"]), max_seq_len)
        for k, v in payload["gdn_states"].items():
            state.gdn_states[int(k)].copy_(v.to(device=device, dtype=dtype))
        for k, v in payload["gdn_conv_states"].items():
            state.gdn_conv_states[int(k)].copy_(v.to(device=device, dtype=dtype))
        if info.get("include_attention_kv") and "attn_k" in payload:
            for k, v in payload["attn_k"].items():
                live = min(v.size(1), state.max_seq_len)
                state.attn_k[int(k)][:, :live, :].copy_(v[:, :live, :].to(device=device, dtype=state.kv_cache_spec.storage_dtype))
            for k, v in payload["attn_v"].items():
                live = min(v.size(1), state.max_seq_len)
                state.attn_v[int(k)][:, :live, :].copy_(v[:, :live, :].to(device=device, dtype=state.kv_cache_spec.storage_dtype))
            if state.attn_k_scale is not None and "attn_k_scale" in payload:
                for k, v in payload["attn_k_scale"].items():
                    live = min(v.size(1), state.max_seq_len)
                    state.attn_k_scale[int(k)][:, :live].copy_(v[:, :live].to(device=device, dtype=torch.float16))
            if state.attn_v_scale is not None and "attn_v_scale" in payload:
                for k, v in payload["attn_v_scale"].items():
                    live = min(v.size(1), state.max_seq_len)
                    state.attn_v_scale[int(k)][:, :live].copy_(v[:, :live].to(device=device, dtype=torch.float16))
            if state.attn_k_zero is not None and "attn_k_zero" in payload:
                for k, v in payload["attn_k_zero"].items():
                    live = min(v.size(1), state.max_seq_len)
                    state.attn_k_zero[int(k)][:, :live].copy_(v[:, :live].to(device=device, dtype=torch.float16))
            if state.attn_v_zero is not None and "attn_v_zero" in payload:
                for k, v in payload["attn_v_zero"].items():
                    live = min(v.size(1), state.max_seq_len)
                    state.attn_v_zero[int(k)][:, :live].copy_(v[:, :live].to(device=device, dtype=torch.float16))
        return state


class DecodeStateArena:
    """Slot-indexed state pool for continuous-serving batched serving.

    Each request gets a lightweight DecodeState view into contiguous
    [slot, ...] buffers. Releasing a request resets its slot and returns it to
    the free list; model code can still consume the standard DecodeState API.
    """

    def __init__(
        self,
        *,
        cfg: Qwen36_27B_TextConfig,
        max_seq_len: int,
        num_slots: int,
        kv_num_blocks: int | None = None,
        kv_block_size: int | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        kv_window_policy: KVWindowPolicy = "error",
        kv_cache_spec: KVCacheSpec | None = None,
    ) -> None:
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        self.cfg = cfg
        self.max_seq_len = int(max_seq_len)
        self.num_slots = int(num_slots)
        self.kv_num_blocks = int(kv_num_blocks or 0)
        self.kv_block_size = int(kv_block_size or 0)
        self.device = torch.device(device)
        self.dtype = dtype
        self.kv_window_policy = kv_window_policy
        self.kv_cache_spec = kv_cache_spec or KVCacheSpec()
        self.paged_kv_enabled = self.kv_num_blocks > 0 and self.kv_block_size > 0
        self.kv_layout = DecodeState.kv_layout(cfg)
        conv_dim = (
            cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        )
        self.gdn_states = {
            layer: torch.zeros(
                self.num_slots,
                cfg.linear_num_value_heads,
                cfg.linear_key_head_dim,
                cfg.linear_value_head_dim,
                device=self.device,
                dtype=dtype,
            )
            for layer in cfg.gdn_layers
        }
        self.gdn_conv_states = {
            layer: torch.zeros(
                self.num_slots,
                conv_dim,
                cfg.linear_conv_kernel_dim - 1,
                device=self.device,
                dtype=dtype,
            )
            for layer in cfg.gdn_layers
        }
        self.pos = torch.zeros((self.num_slots,), device=self.device, dtype=torch.int64)
        self.kv_len = torch.zeros((self.num_slots,), device=self.device, dtype=torch.int64)
        mirror_paged_kv = _env_flag("LANGBURST_PAGED_KV_MIRROR", self.kv_cache_spec.dtype == "fp16")
        paged_attention_kernels = _env_flag("LANGBURST_PAGED_ATTENTION_KERNELS", False)
        shadow_default = (not mirror_paged_kv) or paged_attention_kernels
        allocate_paged_shadow = self.paged_kv_enabled and _env_flag("LANGBURST_PAGED_KV_SHADOW", shadow_default)
        arena_kv = allocate_kv_cache_tensors(
            self.kv_layout,
            self.kv_cache_spec,
            seq_len=self.max_seq_len if (not self.paged_kv_enabled or mirror_paged_kv) else 0,
            device=self.device,
            leading_shape=(self.num_slots,),
        )
        self.attn_k = arena_kv.k
        self.attn_v = arena_kv.v
        self.attn_k_scale = arena_kv.k_scale
        self.attn_v_scale = arena_kv.v_scale
        self.attn_k_zero = arena_kv.k_zero
        self.attn_v_zero = arena_kv.v_zero
        self.paged_attn_k = None
        self.paged_attn_v = None
        self.paged_attn_k_scale = None
        self.paged_attn_v_scale = None
        self.paged_attn_k_zero = None
        self.paged_attn_v_zero = None
        self.paged_int4_tiled_layout = False
        self.paged_kv_pages_allocated = False
        if allocate_paged_shadow:
            if self.kv_cache_spec.is_int4 and int4_kv_layout() == "tiled":
                packed_dim = self.kv_cache_spec.storage_head_dim(self.cfg.attention_head_dim)
                page_shape = (
                    self.kv_num_blocks,
                    self.cfg.num_key_value_heads,
                    packed_dim,
                    self.kv_block_size,
                )
                meta_shape = (self.kv_num_blocks, self.cfg.num_key_value_heads, self.kv_block_size)
                paged_kv = KVCacheTensors(
                    k={
                        layer: kv_buffer(page_shape, device=self.device, dtype=self.kv_cache_spec.storage_dtype)
                        for layer in self.kv_layout.layers
                    },
                    v={
                        layer: kv_buffer(page_shape, device=self.device, dtype=self.kv_cache_spec.storage_dtype)
                        for layer in self.kv_layout.layers
                    },
                    k_scale={layer: torch.ones(meta_shape, device=self.device, dtype=torch.float16) for layer in self.kv_layout.layers},
                    v_scale={layer: torch.ones(meta_shape, device=self.device, dtype=torch.float16) for layer in self.kv_layout.layers},
                    k_zero={layer: torch.zeros(meta_shape, device=self.device, dtype=torch.float16) for layer in self.kv_layout.layers},
                    v_zero={layer: torch.zeros(meta_shape, device=self.device, dtype=torch.float16) for layer in self.kv_layout.layers},
                )
                self.paged_int4_tiled_layout = True
            else:
                paged_kv = allocate_kv_cache_tensors(
                    self.kv_layout,
                    self.kv_cache_spec,
                    seq_len=self.kv_block_size,
                    device=self.device,
                    leading_shape=(self.kv_num_blocks,),
                )
            self.paged_attn_k = paged_kv.k
            self.paged_attn_v = paged_kv.v
            self.paged_attn_k_scale = paged_kv.k_scale
            self.paged_attn_v_scale = paged_kv.v_scale
            self.paged_attn_k_zero = paged_kv.k_zero
            self.paged_attn_v_zero = paged_kv.v_zero
            self.paged_kv_pages_allocated = True
        self._free_slots = list(range(self.num_slots - 1, -1, -1))
        self._active_slots: set[int] = set()

    @property
    def free_slot_count(self) -> int:
        return len(self._free_slots)

    @property
    def active_slot_count(self) -> int:
        return len(self._active_slots)

    def allocate(self) -> tuple[int, DecodeState]:
        if not self._free_slots:
            raise MemoryError("DecodeStateArena exhausted")
        slot = self._free_slots.pop()
        self._active_slots.add(slot)
        self.reset_slot(slot)
        return slot, self.view(slot)

    def release(self, slot: int) -> None:
        slot = int(slot)
        if slot not in self._active_slots:
            return
        self.reset_slot(slot)
        self._active_slots.remove(slot)
        self._free_slots.append(slot)

    def reset_slot(self, slot: int) -> None:
        slot = int(slot)
        for tensor in self.gdn_states.values():
            tensor[slot].zero_()
        for tensor in self.gdn_conv_states.values():
            tensor[slot].zero_()
        for tensor in self.attn_k.values():
            if tensor.size(-2) > 0:
                tensor[slot].zero_()
        for tensor in self.attn_v.values():
            if tensor.size(-2) > 0:
                tensor[slot].zero_()
        for tensor in (self.attn_k_scale or {}).values():
            tensor[slot].fill_(1.0)
        for tensor in (self.attn_v_scale or {}).values():
            tensor[slot].fill_(1.0)
        for tensor in (self.attn_k_zero or {}).values():
            tensor[slot].zero_()
        for tensor in (self.attn_v_zero or {}).values():
            tensor[slot].zero_()
        self.pos[slot].zero_()
        self.kv_len[slot].zero_()

    def advance_slots(self, state_indices: torch.Tensor, token_counts: torch.Tensor) -> None:
        """Advance slot counters on device for fixed-shape decode/verify paths."""

        idx = state_indices.to(device=self.device, dtype=torch.long).reshape(-1).contiguous()
        counts = token_counts.to(device=self.device, dtype=torch.int64).reshape(-1).contiguous()
        if idx.numel() != counts.numel():
            raise ValueError("state_indices and token_counts must have the same length")
        if idx.numel() == 0:
            return
        self.pos.index_add_(0, idx, counts)
        current = self.kv_len.index_select(0, idx)
        updated = torch.clamp(current + counts, max=self.max_seq_len)
        self.kv_len.index_copy_(0, idx, updated)

    def view(self, slot: int) -> DecodeState:
        slot = int(slot)
        if slot < 0 or slot >= self.num_slots:
            raise IndexError("DecodeStateArena slot out of range")
        state = DecodeState(
            cfg=self.cfg,
            gdn_states={layer: tensor[slot] for layer, tensor in self.gdn_states.items()},
            gdn_conv_states={layer: tensor[slot] for layer, tensor in self.gdn_conv_states.items()},
            attn_k={layer: tensor[slot] for layer, tensor in self.attn_k.items()},
            attn_v={layer: tensor[slot] for layer, tensor in self.attn_v.items()},
            attn_k_scale={layer: tensor[slot] for layer, tensor in self.attn_k_scale.items()} if self.attn_k_scale is not None else None,
            attn_v_scale={layer: tensor[slot] for layer, tensor in self.attn_v_scale.items()} if self.attn_v_scale is not None else None,
            attn_k_zero={layer: tensor[slot] for layer, tensor in self.attn_k_zero.items()} if self.attn_k_zero is not None else None,
            attn_v_zero={layer: tensor[slot] for layer, tensor in self.attn_v_zero.items()} if self.attn_v_zero is not None else None,
            pos=0,
            max_seq_len=self.max_seq_len,
            kv_len=0,
            kv_window_policy=self.kv_window_policy,
            kv_cache_spec=self.kv_cache_spec,
        )
        # Dynamic metadata keeps the public DecodeState shape stable while
        # letting hot CUDA paths address slot-indexed arena buffers directly.
        state.arena = self  # type: ignore[attr-defined]
        state.arena_slot = slot  # type: ignore[attr-defined]
        state.pos_tensor = self.pos[slot]  # type: ignore[attr-defined]
        state.kv_len_tensor = self.kv_len[slot]  # type: ignore[attr-defined]
        return state

    @staticmethod
    def _tensor_bytes(tensors: dict[int, torch.Tensor] | None) -> int:
        if tensors is None:
            return 0
        return sum(int(t.numel() * t.element_size()) for t in tensors.values())

    @staticmethod
    def _mib(num_bytes: int) -> float:
        return round(float(num_bytes) / 1024.0 / 1024.0, 2)

    def memory_summary(self) -> dict[str, object]:
        gdn_bytes = self._tensor_bytes(self.gdn_states)
        conv_bytes = self._tensor_bytes(self.gdn_conv_states)
        mirror_kv_bytes = (
            self._tensor_bytes(self.attn_k)
            + self._tensor_bytes(self.attn_v)
            + self._tensor_bytes(self.attn_k_scale)
            + self._tensor_bytes(self.attn_v_scale)
            + self._tensor_bytes(self.attn_k_zero)
            + self._tensor_bytes(self.attn_v_zero)
        )
        paged_kv_bytes = (
            self._tensor_bytes(self.paged_attn_k)
            + self._tensor_bytes(self.paged_attn_v)
            + self._tensor_bytes(self.paged_attn_k_scale)
            + self._tensor_bytes(self.paged_attn_v_scale)
            + self._tensor_bytes(self.paged_attn_k_zero)
            + self._tensor_bytes(self.paged_attn_v_zero)
        )
        total = gdn_bytes + conv_bytes + mirror_kv_bytes + paged_kv_bytes
        return {
            "total_mib": self._mib(total),
            "mib_by_group": {
                "gdn_recurrent": self._mib(gdn_bytes),
                "gdn_conv": self._mib(conv_bytes),
                "dense_or_mirror_kv": self._mib(mirror_kv_bytes),
                "paged_kv": self._mib(paged_kv_bytes),
            },
        }

    def summary(self) -> dict[str, object]:
        out = {
            "num_slots": self.num_slots,
            "active_slots": self.active_slot_count,
            "free_slots": self.free_slot_count,
            "max_seq_len": self.max_seq_len,
            "kv_num_blocks": self.kv_num_blocks,
            "kv_block_size": self.kv_block_size,
            "kv_cache_dtype": self.kv_cache_spec.dtype,
            "kv_storage_head_dim": self.kv_layout.storage_head_dim(self.kv_cache_spec),
            "paged_kv_enabled": self.paged_kv_enabled,
            "device_state_counters": True,
            "memory": self.memory_summary(),
        }
        if self.paged_kv_enabled:
            out["paged_kv_mirror"] = bool(next(iter(self.attn_k.values())).size(-2) > 0) if self.attn_k else False
            out["paged_kv_pages_allocated"] = self.paged_kv_pages_allocated
        return out
