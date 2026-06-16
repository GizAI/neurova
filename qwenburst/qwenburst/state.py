from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal
import time

import torch

from .config import Qwen36_27B_TextConfig

KVWindowPolicy = Literal["error", "shift", "ring"]


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
    dtype: str
    include_attention_kv: bool
    include_conv_state: bool


@dataclass
class DecodeStateWriteSnapshot:
    gdn_states: dict[int, torch.Tensor]
    gdn_conv_states: dict[int, torch.Tensor]
    attn_k_rows: dict[int, tuple[list[int], torch.Tensor]]
    attn_v_rows: dict[int, tuple[list[int], torch.Tensor]]
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
        state.pos = self.pos
        state.kv_len = self.kv_len


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
    pos: int = 0
    max_seq_len: int = 0
    kv_len: int = 0
    kv_window_policy: KVWindowPolicy = "error"

    @classmethod
    def allocate(
        cls,
        cfg: Qwen36_27B_TextConfig,
        max_seq_len: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        kv_window_policy: KVWindowPolicy = "error",
    ) -> "DecodeState":
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        device = torch.device(device)
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
        attn_k = {
            layer: torch.zeros(
                cfg.num_key_value_heads,
                max_seq_len,
                cfg.attention_head_dim,
                device=device,
                dtype=dtype,
            )
            for layer in cfg.attention_layers
        }
        attn_v = {layer: torch.zeros_like(attn_k[layer]) for layer in cfg.attention_layers}
        return cls(
            cfg=cfg,
            gdn_states=gdn_states,
            gdn_conv_states=gdn_conv_states,
            attn_k=attn_k,
            attn_v=attn_v,
            pos=0,
            max_seq_len=max_seq_len,
            kv_len=0,
            kv_window_policy=kv_window_policy,
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
        return sum(t.numel() * t.element_size() for t in self.attn_k.values()) + sum(
            t.numel() * t.element_size() for t in self.attn_v.values()
        )

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
        self.pos = 0

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
            pos=self.pos,
            max_seq_len=self.max_seq_len,
            kv_len=self.kv_len,
            kv_window_policy=self.kv_window_policy,
        )

    def copy_from_(self, other: "DecodeState", *, copy_attention: bool = True) -> None:
        """Replace this state with another state under the same allocation contract."""
        if self.max_seq_len != other.max_seq_len or self.kv_window_policy != other.kv_window_policy:
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
        self.pos = other.pos
        self.kv_len = other.kv_len

    def speculative_write_snapshot(self, num_tokens: int) -> DecodeStateWriteSnapshot:
        """Snapshot only state locations a speculative block may overwrite.

        This is the qwenburst analogue of vLLM's lookahead slot safety: a block
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
        self.attn_k[layer][:, idx, :] = k
        self.attn_v[layer][:, idx, :] = v
        return idx

    def attention_kv_view(self, layer: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return KV in logical oldest→newest order for baseline attention.

        For `ring` this may allocate a concatenated view.  That is acceptable for
        correctness and first-real-chat; production CUDA should consume physical
        ring indices directly.
        """
        live = min(self.kv_len + 1, self.max_seq_len)
        if self.kv_window_policy != "ring" or self.kv_len < self.max_seq_len:
            return self.attn_k[layer], self.attn_v[layer], live
        start = (self.pos + 1) % self.max_seq_len
        if start == 0:
            return self.attn_k[layer], self.attn_v[layer], live
        k = torch.cat([self.attn_k[layer][:, start:, :], self.attn_k[layer][:, :start, :]], dim=1)
        v = torch.cat([self.attn_v[layer][:, start:, :], self.attn_v[layer][:, :start, :]], dim=1)
        return k, v, live

    def attention_kv_view_at(self, layer: int, *, logical_pos: int, live_len: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return KV view after a block-local write at `logical_pos`."""
        live = min(live_len, self.max_seq_len)
        if self.kv_window_policy != "ring" or live < self.max_seq_len:
            return self.attn_k[layer], self.attn_v[layer], live
        start = (logical_pos + 1) % self.max_seq_len
        if start == 0:
            return self.attn_k[layer], self.attn_v[layer], live
        k = torch.cat([self.attn_k[layer][:, start:, :], self.attn_k[layer][:, :start, :]], dim=1)
        v = torch.cat([self.attn_v[layer][:, start:, :], self.attn_v[layer][:, :start, :]], dim=1)
        return k, v, live

    def finish_token(self) -> None:
        self.pos += 1
        self.kv_len = min(self.kv_len + 1, self.max_seq_len)

    def snapshot_dict(self, *, include_attention_kv: bool = True) -> dict[str, Any]:
        info = StateSnapshotInfo(
            schema_version=3,
            created_unix=time.time(),
            model_family="qwen3.6-27b-text",
            pos=self.pos,
            max_seq_len=self.max_seq_len,
            kv_len=self.kv_len,
            kv_window_policy=self.kv_window_policy,
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
                state.attn_k[int(k)][:, :live, :].copy_(v[:, :live, :].to(device=device, dtype=dtype))
            for k, v in payload["attn_v"].items():
                live = min(v.size(1), state.max_seq_len)
                state.attn_v[int(k)][:, :live, :].copy_(v[:, :live, :].to(device=device, dtype=dtype))
        return state


class DecodeStateArena:
    """Slot-indexed state pool for vLLM-style batched serving.

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
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        kv_window_policy: KVWindowPolicy = "error",
    ) -> None:
        if num_slots < 1:
            raise ValueError("num_slots must be >= 1")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")
        self.cfg = cfg
        self.max_seq_len = int(max_seq_len)
        self.num_slots = int(num_slots)
        self.device = torch.device(device)
        self.dtype = dtype
        self.kv_window_policy = kv_window_policy
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
        self.attn_k = {
            layer: torch.zeros(
                self.num_slots,
                cfg.num_key_value_heads,
                self.max_seq_len,
                cfg.attention_head_dim,
                device=self.device,
                dtype=dtype,
            )
            for layer in cfg.attention_layers
        }
        self.attn_v = {layer: torch.zeros_like(self.attn_k[layer]) for layer in cfg.attention_layers}
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
            tensor[slot].zero_()
        for tensor in self.attn_v.values():
            tensor[slot].zero_()

    def view(self, slot: int) -> DecodeState:
        slot = int(slot)
        if slot < 0 or slot >= self.num_slots:
            raise IndexError("DecodeStateArena slot out of range")
        return DecodeState(
            cfg=self.cfg,
            gdn_states={layer: tensor[slot] for layer, tensor in self.gdn_states.items()},
            gdn_conv_states={layer: tensor[slot] for layer, tensor in self.gdn_conv_states.items()},
            attn_k={layer: tensor[slot] for layer, tensor in self.attn_k.items()},
            attn_v={layer: tensor[slot] for layer, tensor in self.attn_v.items()},
            pos=0,
            max_seq_len=self.max_seq_len,
            kv_len=0,
            kv_window_policy=self.kv_window_policy,
        )

    def summary(self) -> dict[str, int]:
        return {
            "num_slots": self.num_slots,
            "active_slots": self.active_slot_count,
            "free_slots": self.free_slot_count,
            "max_seq_len": self.max_seq_len,
        }
