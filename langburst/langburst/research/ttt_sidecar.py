from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TTTSidecarConfig:
    hidden_size: int = 5120
    memory_rank: int = 256
    lr: float = 0.05
    decay: float = 0.995
    read_scale: float = 0.05
    dtype: torch.dtype = torch.float16


class TTTSidecarMemory:
    """A safe TTT-style sidecar, not a replacement for Qwen layers.

    It learns a tiny fast-weight memory from hidden vectors during prefill or
    chunk ingestion. Decode can read from it as a residual hint. This keeps the
    base Qwen3.6 weights and GDN dynamics intact, avoiding the quality collapse
    risk of swapping internal layers without retraining.
    """

    def __init__(self, cfg: TTTSidecarConfig, device: str | torch.device = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        # Low-rank projection is fixed random by default. Production can train it.
        g = torch.Generator(device="cpu").manual_seed(36)
        proj = torch.randn(cfg.hidden_size, cfg.memory_rank, generator=g, dtype=torch.float32)
        proj = F.normalize(proj, dim=0).to(device=self.device, dtype=cfg.dtype)
        self.proj = proj
        self.memory = torch.zeros(cfg.memory_rank, cfg.memory_rank, device=self.device, dtype=torch.float32)
        self.updates = 0

    @torch.no_grad()
    def reset(self) -> None:
        self.memory.zero_()
        self.updates = 0

    @torch.no_grad()
    def update(self, hidden: torch.Tensor) -> None:
        """Online self-supervised fast-weight update.

        hidden may be [H] or [T,H]. We use adjacent projected states as a small
        next-state associative memory. This is intentionally cheap and stable.
        """
        h = hidden.reshape(-1, self.cfg.hidden_size).to(device=self.device, dtype=self.cfg.dtype)
        if h.size(0) < 2:
            key = F.normalize((h @ self.proj).float(), dim=-1)
            val = key
        else:
            key = F.normalize((h[:-1] @ self.proj).float(), dim=-1)
            val = F.normalize((h[1:] @ self.proj).float(), dim=-1)
        delta = key.transpose(0, 1) @ val / max(1, key.size(0))
        self.memory.mul_(self.cfg.decay).add_(delta, alpha=self.cfg.lr)
        self.updates += int(key.size(0))

    @torch.no_grad()
    def read(self, query_hidden: torch.Tensor) -> torch.Tensor:
        q = query_hidden.reshape(-1, self.cfg.hidden_size)[-1:].to(device=self.device, dtype=self.cfg.dtype)
        key = F.normalize((q @ self.proj).float(), dim=-1)
        val = key @ self.memory
        out = (val.to(self.cfg.dtype) @ self.proj.t()).reshape(-1)
        return out * self.cfg.read_scale

    def state_dict(self) -> dict[str, torch.Tensor | int | float]:
        return {"proj": self.proj.detach().cpu(), "memory": self.memory.detach().cpu(), "updates": self.updates}

    def load_state_dict(self, state: dict[str, torch.Tensor | int | float]) -> None:
        self.proj.copy_(state["proj"].to(device=self.device, dtype=self.cfg.dtype))  # type: ignore[union-attr]
        self.memory.copy_(state["memory"].to(device=self.device, dtype=torch.float32))  # type: ignore[union-attr]
        self.updates = int(state.get("updates", 0))
