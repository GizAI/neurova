from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

import torch

from ..adapters.qwen36_impl.config import Qwen36_27B_TextConfig
from ..adapters.qwen36_impl.state import DecodeState


class OneTokenModel(Protocol):
    cfg: Qwen36_27B_TextConfig

    def forward_one(self, token: torch.Tensor | int, state: DecodeState, *, use_mtp: bool = False) -> torch.Tensor:
        """Advance model/state by one token and return logits."""


@dataclass
class StreamStats:
    tokens_ingested: int = 0
    chunks_ingested: int = 0
    snapshots_written: int = 0
    state_decays: int = 0


@dataclass
class InfiniteStreamPolicy:
    """Controls how LangBurst turns finite exact KV into infinite streaming.

    recent_window_tokens is the exact-attention budget. GDN state carries the
    compressed long memory beyond that window. A future paged/ring KV kernel can
    replace the current correctness fallback without changing this API.
    """

    recent_window_tokens: int = 8192
    kv_window_policy: str = "ring"
    boundary_decay: float = 1.0
    snapshot_every_tokens: int = 0
    snapshot_dir: Path | None = None


@dataclass
class InfiniteStreamingRuntime:
    """High-level stateful ingestion runtime.

    This class is deliberately model-agnostic. It gives the engine three product
    capabilities that ordinary prompt/KV serving lacks:
      * unbounded token ingestion with fixed recurrent-state memory,
      * cheap snapshots after huge documents/codebases,
      * branchable decode states for search or what-if generation.
    """

    model: OneTokenModel
    state: DecodeState
    policy: InfiniteStreamPolicy = field(default_factory=InfiniteStreamPolicy)
    stats: StreamStats = field(default_factory=StreamStats)

    @classmethod
    def create(
        cls,
        model: OneTokenModel,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
        policy: InfiniteStreamPolicy | None = None,
    ) -> "InfiniteStreamingRuntime":
        policy = policy or InfiniteStreamPolicy()
        state = DecodeState.allocate(
            model.cfg,
            max_seq_len=policy.recent_window_tokens,
            device=device,
            dtype=dtype,
            kv_window_policy=policy.kv_window_policy,  # type: ignore[arg-type]
        )
        return cls(model=model, state=state, policy=policy)

    @torch.no_grad()
    def ingest_tokens(self, tokens: Iterable[int], *, boundary: bool = False) -> None:
        if boundary and self.policy.boundary_decay < 1.0:
            self.state.decay_gdn_(self.policy.boundary_decay)
            self.stats.state_decays += 1
        device = self.state.device
        for tok in tokens:
            token = torch.tensor(int(tok), device=device, dtype=torch.long)
            _ = self.model.forward_one(token, self.state, use_mtp=False)
            self.stats.tokens_ingested += 1
            if self.policy.snapshot_every_tokens and self.stats.tokens_ingested % self.policy.snapshot_every_tokens == 0:
                self.write_snapshot()
        self.stats.chunks_ingested += 1

    def fork(self) -> DecodeState:
        return self.state.fork(clone_attention=True)

    def write_snapshot(self, name: str | None = None, *, include_attention_kv: bool = True) -> Path:
        if self.policy.snapshot_dir is None:
            raise RuntimeError("snapshot_dir is not configured")
        self.policy.snapshot_dir.mkdir(parents=True, exist_ok=True)
        name = name or f"stream_pos_{self.state.pos:012d}.qbstate.pt"
        path = self.policy.snapshot_dir / name
        self.state.save_snapshot(path, include_attention_kv=include_attention_kv)
        self.stats.snapshots_written += 1
        return path
