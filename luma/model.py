from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LUMAConfig:
    vocab_size: int = 259
    d_model: int = 192
    n_layer: int = 4
    n_slots: int = 64
    topk: int = 4
    chunk_size: int = 32
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalDepthwiseMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dw = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.pw = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.dw(y).transpose(1, 2)
        y, gate = self.pw(y).chunk(2, dim=-1)
        return y * torch.sigmoid(gate)


def _batched_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    dim = x.size(-1)
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, dim))


def _batched_scatter(base: torch.Tensor, idx: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    dim = base.size(-1)
    return base.scatter(1, idx.unsqueeze(-1).expand(-1, -1, dim), values)


class SlotState(SimpleNamespace):
    key: torch.Tensor
    value: torch.Tensor
    confidence: torch.Tensor
    utility: torch.Tensor
    age: torch.Tensor
    lock: torch.Tensor


class LUMABlock(nn.Module):
    def __init__(self, cfg: LUMAConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.topk = cfg.topk
        self.event_norm = RMSNorm(d)
        self.event = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.local = CausalDepthwiseMixer(d)
        self.q = nn.Linear(d, d, bias=False)
        self.k_update = nn.Linear(2 * d, d)
        self.v_candidate = nn.Linear(2 * d, d)
        self.erase = nn.Linear(3 * d, d)
        self.write = nn.Linear(3 * d, d)
        self.protect = nn.Linear(3 * d, d)
        self.utility = nn.Linear(2 * d, 1)
        self.confidence = nn.Linear(2 * d, 1)
        self.out = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.out_norm = RMSNorm(d)

    def forward(self, x: torch.Tensor, slots: SlotState) -> tuple[torch.Tensor, SlotState, torch.Tensor]:
        e = self.local(self.event(self.event_norm(x)))
        chunk_event = e.mean(dim=1)
        q = F.normalize(self.q(chunk_event), dim=-1)
        scores = torch.einsum("bd,bkd->bk", q, F.normalize(slots.key, dim=-1))
        scores = scores + 0.05 * slots.utility.squeeze(-1) - 0.01 * slots.age.squeeze(-1)
        idx = scores.topk(k=min(self.topk, scores.size(-1)), dim=-1).indices

        selected_key = _batched_gather(slots.key, idx)
        selected_value = _batched_gather(slots.value, idx)
        weights = torch.softmax(_batched_gather(scores.unsqueeze(-1), idx).squeeze(-1), dim=-1)
        read = torch.sum(selected_value * weights.unsqueeze(-1), dim=1)

        memory_input = torch.cat([chunk_event, read], dim=-1)
        cand = self.v_candidate(memory_input)
        gate_input = torch.cat(
            [
                selected_value,
                cand.unsqueeze(1).expand_as(selected_value),
                read.unsqueeze(1).expand_as(selected_value),
            ],
            dim=-1,
        )
        erase = torch.sigmoid(self.erase(gate_input))
        write = torch.sigmoid(self.write(gate_input))
        protect = torch.sigmoid(self.protect(gate_input) + _batched_gather(slots.lock, idx))
        new_value = protect * selected_value + (1.0 - protect) * (
            (1.0 - erase) * selected_value + write * cand.unsqueeze(1)
        )
        new_key = F.normalize(0.95 * selected_key + 0.05 * self.k_update(memory_input).unsqueeze(1), dim=-1)
        new_utility = torch.sigmoid(self.utility(torch.cat([new_value, cand.unsqueeze(1).expand_as(new_value)], dim=-1)))
        new_conf = torch.sigmoid(self.confidence(torch.cat([new_value, selected_value], dim=-1)))

        slots = SlotState(
            key=_batched_scatter(slots.key, idx, new_key),
            value=_batched_scatter(slots.value, idx, new_value),
            confidence=_batched_scatter(slots.confidence, idx, new_conf),
            utility=_batched_scatter(slots.utility, idx, new_utility),
            age=slots.age + 1.0,
            lock=slots.lock,
        )
        y = self.out(torch.cat([e, read[:, None, :].expand_as(e)], dim=-1))
        return x + self.out_norm(y), slots, scores


class LUMALM(nn.Module):
    def __init__(self, cfg: LUMAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.slot_keys = nn.Parameter(torch.randn(cfg.n_layer, cfg.n_slots, cfg.d_model) * 0.02)
        self.slot_locks = nn.Parameter(torch.zeros(cfg.n_layer, cfg.n_slots, 1))
        self.blocks = nn.ModuleList([LUMABlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        nn.init.normal_(self.slot_keys, mean=0.0, std=0.02)
        nn.init.zeros_(self.slot_locks)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _initial_slots(self, batch: int, layer: int, device: torch.device, dtype: torch.dtype) -> SlotState:
        key = F.normalize(self.slot_keys[layer].to(device=device, dtype=dtype), dim=-1)
        return SlotState(
            key=key.unsqueeze(0).expand(batch, -1, -1).contiguous(),
            value=torch.zeros(batch, self.cfg.n_slots, self.cfg.d_model, device=device, dtype=dtype),
            confidence=torch.zeros(batch, self.cfg.n_slots, 1, device=device, dtype=dtype),
            utility=torch.zeros(batch, self.cfg.n_slots, 1, device=device, dtype=dtype),
            age=torch.zeros(batch, self.cfg.n_slots, 1, device=device, dtype=dtype),
            lock=self.slot_locks[layer].to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1),
        )

    def forward(self, input_ids: torch.Tensor, return_aux: bool = False) -> SimpleNamespace:
        batch, length = input_ids.shape
        x = self.embed(input_ids)
        chunks: list[torch.Tensor] = []
        aux_scores: list[torch.Tensor] = []
        layer_slots = [
            self._initial_slots(batch, layer, input_ids.device, x.dtype)
            for layer in range(self.cfg.n_layer)
        ]
        for start in range(0, length, self.cfg.chunk_size):
            h = x[:, start : start + self.cfg.chunk_size]
            for layer, block in enumerate(self.blocks):
                h, layer_slots[layer], scores = block(h, layer_slots[layer])
                aux_scores.append(scores)
            chunks.append(h)
        logits = self.lm_head(self.norm(torch.cat(chunks, dim=1)))
        aux = None
        if return_aux and aux_scores:
            aux = {"slot_entropy": torch.stack([-(s.softmax(-1) * s.log_softmax(-1)).sum(-1).mean() for s in aux_scores]).mean()}
        return SimpleNamespace(logits=logits, aux=aux)
