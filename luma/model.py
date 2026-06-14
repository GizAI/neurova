from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LUMAConfig:
    vocab_size: int = 259
    tokenizer_backend: str = "bytepatch"
    qwen_tokenizer_path: str = "tokenizers/qwen35"
    bytepatch_vocab_path: str = "tokenizers/luma_bytepatch/bytepatch_vocab.json"
    tokenizer_sha256: str = ""
    d_model: int = 192
    n_layer: int = 4
    n_slots: int = 64
    topk: int = 4
    chunk_size: int = 32
    dropout: float = 0.0
    local_heads: int = 4
    copy_window: int = 128
    memory_read_bias: float = -6.0
    token_read_topk: int = 8
    use_slots: bool = True
    use_local_attention: bool = True

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


class CausalChunkAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by local_heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        mask = torch.zeros(length, length, device=x.device, dtype=x.dtype)
        mask = mask.masked_fill(torch.ones(length, length, device=x.device, dtype=torch.bool).triu(1), -torch.inf)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, dim))


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
        self.cfg = cfg
        d = cfg.d_model
        self.topk = cfg.topk
        self.event_norm = RMSNorm(d)
        self.event = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.local = CausalDepthwiseMixer(d)
        self.local_attn_norm = RMSNorm(d) if cfg.use_local_attention and cfg.local_heads > 0 else None
        self.local_attn = CausalChunkAttention(d, cfg.local_heads) if cfg.use_local_attention and cfg.local_heads > 0 else None
        self.fact_pool = nn.Linear(d, 1)
        self.q = nn.Linear(d, d, bias=False)
        self.k_update = nn.Linear(2 * d, d)
        self.v_candidate = nn.Linear(2 * d, d)
        self.erase = nn.Linear(3 * d, d)
        self.write = nn.Linear(3 * d, d)
        self.protect = nn.Linear(3 * d, d)
        self.utility = nn.Linear(2 * d, 1)
        self.confidence = nn.Linear(2 * d, 1)
        self.read_gate = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.Linear(2 * d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
        self.out_norm = RMSNorm(d)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, cfg.memory_read_bias)

    def forward(
        self,
        x: torch.Tensor,
        slots: SlotState,
        *,
        disable_slots: bool = False,
        update_slots: bool = True,
        disable_local_attention: bool = False,
    ) -> tuple[torch.Tensor, SlotState, dict[str, torch.Tensor], torch.Tensor]:
        e = self.local(self.event(self.event_norm(x)))
        if not disable_local_attention and self.local_attn is not None and self.local_attn_norm is not None:
            e = e + self.local_attn(self.local_attn_norm(e))
        pool_weights = torch.softmax(self.fact_pool(e).squeeze(-1), dim=-1)
        chunk_event = torch.sum(e * pool_weights.unsqueeze(-1), dim=1)
        slots_disabled = disable_slots or (not self.cfg.use_slots) or slots.key.size(1) == 0
        if slots_disabled:
            token_read = torch.zeros_like(e)
            read_gate = torch.zeros_like(e)
            read = torch.zeros_like(chunk_event)
            idx = torch.empty(e.size(0), 0, device=e.device, dtype=torch.long)
            selected_value = torch.empty(e.size(0), 0, e.size(-1), device=e.device, dtype=e.dtype)
            new_value = selected_value
            protect = torch.empty(e.size(0), 0, e.size(-1), device=e.device, dtype=e.dtype)
            write = protect
            scores = torch.empty(e.size(0), 0, device=e.device, dtype=e.dtype)
            updated_slots = slots
            y = self.out(torch.cat([e, token_read], dim=-1))
            zero = e.new_tensor(0.0)
            diag = {
                "scores": scores,
                "selected_idx": idx,
                "slot_entropy": zero,
                "erase_mean": zero,
                "write_mean": zero,
                "protect_mean": zero,
                "overwrite_rate": zero,
                "slot_delta": zero,
                "confidence_mean": zero,
                "utility_mean": zero,
                "read_gate_mean": zero,
                "pool_entropy": -(pool_weights * pool_weights.clamp_min(1e-9).log()).sum(-1).mean(),
            }
            return x + self.out_norm(y), updated_slots, diag, token_read
        token_q = F.normalize(self.q(e), dim=-1)
        slot_key_norm = F.normalize(slots.key, dim=-1)
        token_scores = torch.einsum("btd,bkd->btk", token_q, slot_key_norm)
        token_scores = token_scores + 0.05 * slots.utility.squeeze(-1).unsqueeze(1) - 0.01 * slots.age.squeeze(-1).unsqueeze(1)
        read_k = min(self.cfg.token_read_topk, token_scores.size(-1))
        read_idx = token_scores.topk(k=read_k, dim=-1).indices
        read_selected_scores = token_scores.gather(-1, read_idx)
        read_weights = torch.softmax(read_selected_scores, dim=-1)
        selected_values = slots.value.unsqueeze(1).expand(-1, e.size(1), -1, -1)
        selected_values = selected_values.gather(
            2,
            read_idx.unsqueeze(-1).expand(-1, -1, -1, slots.value.size(-1)),
        )
        token_read = torch.sum(selected_values * read_weights.unsqueeze(-1), dim=2)
        read_gate = torch.sigmoid(self.read_gate(torch.cat([e, token_read], dim=-1)))
        token_read = read_gate * token_read

        q = F.normalize(self.q(chunk_event), dim=-1)
        scores = torch.einsum("bd,bkd->bk", q, slot_key_norm)
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

        updated_slots = slots
        if (not disable_slots) and update_slots:
            aged = slots.age + 1.0
            updated_slots = SlotState(
                key=_batched_scatter(slots.key, idx, new_key),
                value=_batched_scatter(slots.value, idx, new_value),
                confidence=_batched_scatter(slots.confidence, idx, new_conf),
                utility=_batched_scatter(slots.utility, idx, new_utility),
                age=_batched_scatter(aged, idx, torch.zeros_like(_batched_gather(aged, idx))),
                lock=slots.lock,
            )
        y = self.out(torch.cat([e, token_read], dim=-1))
        entropy = -(scores.softmax(-1) * scores.log_softmax(-1)).sum(-1).mean()
        overwrite = ((1.0 - protect) * write).mean()
        delta = (new_value - selected_value).pow(2).mean().sqrt()
        diag = {
            "scores": scores,
            "selected_idx": idx,
            "slot_entropy": entropy,
            "erase_mean": erase.mean(),
            "write_mean": write.mean(),
            "protect_mean": protect.mean(),
            "overwrite_rate": overwrite,
            "slot_delta": delta,
            "confidence_mean": updated_slots.confidence.mean(),
            "utility_mean": updated_slots.utility.mean(),
            "read_gate_mean": read_gate.mean(),
            "pool_entropy": -(pool_weights * pool_weights.clamp_min(1e-9).log()).sum(-1).mean(),
        }
        return x + self.out_norm(y), updated_slots, diag, token_read


class LUMALM(nn.Module):
    def __init__(self, cfg: LUMAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.chunk_size, cfg.d_model)
        self.slot_keys = nn.Parameter(torch.randn(cfg.n_layer, cfg.n_slots, cfg.d_model) * 0.02)
        self.slot_locks = nn.Parameter(torch.zeros(cfg.n_layer, cfg.n_slots, 1))
        self.blocks = nn.ModuleList([LUMABlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.copy_q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.copy_k = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.copy_gate = nn.Linear(cfg.d_model, 1)
        self.copy_scale = nn.Parameter(torch.tensor(0.5))
        self.memory_norm = RMSNorm(cfg.d_model)
        self.memory_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.memory_scale = nn.Parameter(torch.tensor(1.0))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.memory_head.weight = self.embed.weight
        self.apply(self._init_weights)
        self._init_memory_gates()
        if self.slot_keys.numel() > 0:
            nn.init.normal_(self.slot_keys, mean=0.0, std=0.02)
        if self.slot_locks.numel() > 0:
            nn.init.zeros_(self.slot_locks)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_memory_gates(self) -> None:
        for block in self.blocks:
            nn.init.zeros_(block.read_gate.weight)
            nn.init.constant_(block.read_gate.bias, self.cfg.memory_read_bias)

    def _initial_slots(
        self,
        batch: int,
        layer: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        random_keys: bool = False,
    ) -> SlotState:
        if random_keys:
            key = torch.randn(self.cfg.n_slots, self.cfg.d_model, device=device, dtype=dtype)
        else:
            key = self.slot_keys[layer].to(device=device, dtype=dtype)
        key = F.normalize(key, dim=-1)
        return SlotState(
            key=key.unsqueeze(0).expand(batch, -1, -1).contiguous(),
            value=torch.zeros(batch, self.cfg.n_slots, self.cfg.d_model, device=device, dtype=dtype),
            confidence=torch.zeros(batch, self.cfg.n_slots, 1, device=device, dtype=dtype),
            utility=torch.zeros(batch, self.cfg.n_slots, 1, device=device, dtype=dtype),
            age=torch.zeros(batch, self.cfg.n_slots, 1, device=device, dtype=dtype),
            lock=self.slot_locks[layer].to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1),
        )

    @staticmethod
    def detach_slots(slots: list[SlotState]) -> list[SlotState]:
        return [
            SlotState(
                key=slot.key.detach(),
                value=slot.value.detach(),
                confidence=slot.confidence.detach(),
                utility=slot.utility.detach(),
                age=slot.age.detach(),
                lock=slot.lock.detach(),
            )
            for slot in slots
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        slots_in: list[SlotState] | None = None,
        return_slots: bool = False,
        return_aux: bool = False,
        ablation: str = "normal",
    ) -> SimpleNamespace:
        if ablation not in {"normal", "no_slots", "random_slot_keys", "no_copy", "no_slots_no_copy", "no_local_attention"}:
            raise ValueError(f"unknown ablation={ablation!r}")
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device) % self.cfg.chunk_size
        x = self.embed(input_ids) + self.pos_embed(positions).unsqueeze(0)
        chunks: list[torch.Tensor] = []
        memory_chunks: list[torch.Tensor] = []
        aux_diags: list[dict[str, torch.Tensor]] = []
        if slots_in is not None:
            if len(slots_in) != self.cfg.n_layer:
                raise ValueError(f"slots_in must have {self.cfg.n_layer} layers, got {len(slots_in)}")
            layer_slots = slots_in
        else:
            layer_slots = [
                self._initial_slots(
                    batch,
                    layer,
                    input_ids.device,
                    x.dtype,
                    random_keys=(ablation == "random_slot_keys"),
                )
                for layer in range(self.cfg.n_layer)
            ]
        disable_slots = ablation in {"no_slots", "no_slots_no_copy"} or (not self.cfg.use_slots) or self.cfg.n_slots <= 0
        disable_copy = ablation in {"no_copy", "no_slots_no_copy"}
        disable_local_attention = ablation == "no_local_attention" or (not self.cfg.use_local_attention)
        for start in range(0, length, self.cfg.chunk_size):
            h = x[:, start : start + self.cfg.chunk_size]
            memory_sum = torch.zeros_like(h)
            for layer, block in enumerate(self.blocks):
                h, layer_slots[layer], diag, memory_signal = block(
                    h,
                    layer_slots[layer],
                    disable_slots=disable_slots,
                    disable_local_attention=disable_local_attention,
                )
                memory_sum = memory_sum + memory_signal
                if return_aux:
                    diag = {**diag, "layer": torch.tensor(layer, device=input_ids.device)}
                    aux_diags.append(diag)
            chunks.append(h)
            memory_chunks.append(memory_sum / max(1, self.cfg.n_layer))
        h = self.norm(torch.cat(chunks, dim=1))
        memory_h = self.memory_norm(torch.cat(memory_chunks, dim=1))
        logits = self.lm_head(h)
        memory_logits = self.memory_head(memory_h)
        if not disable_slots:
            logits = logits + torch.relu(self.memory_scale) * memory_logits
        if self.cfg.copy_window > 0 and not disable_copy:
            logits = logits + self._copy_logits(h, input_ids)
        aux = None
        if return_aux and aux_diags:
            aux = self._merge_aux(aux_diags)
        return SimpleNamespace(
            logits=logits,
            memory_logits=memory_logits,
            aux=aux,
            slots=layer_slots if return_slots else None,
        )

    def _merge_aux(self, diags: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        scalar_keys = [
            "slot_entropy",
            "erase_mean",
            "write_mean",
            "protect_mean",
            "overwrite_rate",
            "slot_delta",
            "confidence_mean",
            "utility_mean",
            "read_gate_mean",
            "pool_entropy",
        ]
        aux = {key: torch.stack([diag[key] for diag in diags]).mean() for key in scalar_keys}
        usage = torch.zeros(
            self.cfg.n_layer,
            self.cfg.n_slots,
            device=diags[0]["selected_idx"].device,
            dtype=torch.float32,
        )
        for diag in diags:
            layer = int(diag["layer"].item())
            selected = diag["selected_idx"].detach().reshape(-1)
            usage[layer].scatter_add_(0, selected, torch.ones_like(selected, dtype=torch.float32))
        if usage.size(-1) == 0:
            aux["slot_usage"] = usage
            aux["slot_usage_entropy"] = usage.new_tensor(0.0)
            aux["slot_update_frequency"] = usage.new_tensor(0.0)
            return aux
        total = usage.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = usage / total
        aux["slot_usage"] = usage
        aux["slot_usage_entropy"] = (-(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(-1)).mean()
        aux["slot_update_frequency"] = (usage > 0).float().mean()
        return aux

    def _copy_logits(self, h: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length, _ = h.shape
        q = F.normalize(self.copy_q(h), dim=-1)
        k = F.normalize(self.copy_k(h), dim=-1)
        scores = torch.einsum("btd,bsd->bts", q, k)
        pos = torch.arange(length, device=h.device)
        causal = pos[None, :] < pos[:, None]
        if self.cfg.copy_window > 0:
            causal = causal & ((pos[:, None] - pos[None, :]) <= self.cfg.copy_window)
        scores = scores.masked_fill(~causal.unsqueeze(0), 0.0)
        source_ids = input_ids[:, None, :].expand(batch, length, length)
        bonus = torch.zeros(batch, length, self.cfg.vocab_size, device=h.device, dtype=h.dtype)
        bonus.scatter_reduce_(2, source_ids, scores.to(h.dtype), reduce="amax", include_self=True)
        gate = torch.sigmoid(self.copy_gate(h))
        return gate * torch.relu(self.copy_scale) * bonus
