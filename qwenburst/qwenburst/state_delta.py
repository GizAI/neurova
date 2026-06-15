from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import DecodeState


@dataclass
class DecodeStateDelta:
    """Tensor delta between two DecodeState snapshots with the same allocation.

    This is intentionally small and exact for current state tensors. It is a
    building block for future reusable chunk deltas; it does not try to solve
    non-commutative recurrent-state merging.
    """

    gdn_states: dict[int, torch.Tensor]
    gdn_conv_states: dict[int, torch.Tensor]
    attn_k: dict[int, torch.Tensor]
    attn_v: dict[int, torch.Tensor]
    pos_delta: int
    kv_len: int

    @classmethod
    def between(cls, before: DecodeState, after: DecodeState) -> "DecodeStateDelta":
        if before.max_seq_len != after.max_seq_len or before.kv_window_policy != after.kv_window_policy:
            raise ValueError("state delta requires matching allocation contract")
        return cls(
            gdn_states={k: after.gdn_states[k] - before.gdn_states[k] for k in before.gdn_states},
            gdn_conv_states={k: after.gdn_conv_states[k] - before.gdn_conv_states[k] for k in before.gdn_conv_states},
            attn_k={k: after.attn_k[k] - before.attn_k[k] for k in before.attn_k},
            attn_v={k: after.attn_v[k] - before.attn_v[k] for k in before.attn_v},
            pos_delta=int(after.pos - before.pos),
            kv_len=int(after.kv_len),
        )

    def apply_to(self, state: DecodeState) -> None:
        for k, delta in self.gdn_states.items():
            state.gdn_states[k].add_(delta.to(device=state.device, dtype=state.dtype))
        for k, delta in self.gdn_conv_states.items():
            state.gdn_conv_states[k].add_(delta.to(device=state.device, dtype=state.dtype))
        for k, delta in self.attn_k.items():
            state.attn_k[k].add_(delta.to(device=state.device, dtype=state.dtype))
        for k, delta in self.attn_v.items():
            state.attn_v[k].add_(delta.to(device=state.device, dtype=state.dtype))
        state.pos += self.pos_delta
        state.kv_len = min(self.kv_len, state.max_seq_len)
