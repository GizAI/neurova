from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .config import Qwen36_27B_TextConfig
from .loader import QuantizedStore, LowBitTensor, LowBitMarlinTensor, FP16Tensor
from .ops import cuda_ops
from .state import DecodeState
from .tuning import lowbit_rows_per_cta

TensorLike = LowBitTensor | LowBitMarlinTensor | FP16Tensor


@dataclass
class BlockForwardResult:
    logits: list[torch.Tensor]
    hidden_taps: list[list[torch.Tensor]]
    state: DecodeState
    next_ids: list[int]
    accepted_next: int


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


def qwen_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return cuda_ops().rmsnorm_qwen(x.contiguous(), weight.to(device=x.device, dtype=x.dtype).contiguous(), eps)


def qwen_rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * (1.0 + weight.to(device=x.device, dtype=torch.float32))).to(x.dtype)


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
    out = []
    for row in x:
        out.append(depthwise_conv_update(buf, row.contiguous(), weight, bias))
    return torch.stack(out, dim=0).contiguous()


class QwenBurstMLP:
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
        return linear_any(self.down, F.silu(gate) * up)

    def forward_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_up is not None:
            mixed = linear_any(self.gate_up, x)
            gate, up = torch.split(mixed, [self.intermediate_size, self.intermediate_size], dim=-1)
        else:
            gate = linear_any(self.gate, x)
            up = linear_any(self.up, x)
        return linear_any(self.down, F.silu(gate) * up)


class QwenBurstGDNLayer:
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
        self.A_log = weights.fp16(*(f"{la}.A_log" for la in la_candidates)).float()
        self.dt_bias = weights.fp16(*(f"{la}.dt_bias" for la in la_candidates)).float()
        self.mlp = QwenBurstMLP(cfg, weights, layer)

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

        beta = torch.sigmoid(b).to(torch.float16).contiguous()
        g = (-torch.exp(self.A_log.to(a.device)) * F.softplus(a.float() + self.dt_bias.to(a.device))).contiguous()
        core = ops.gdn_recurrent(q, k, v, g, beta, state.gdn_states[self.layer])
        core_norm = qwen_gdn_norm_silu_gate(core, self.gdn_norm_w, z, self.cfg.rms_norm_eps)
        x = residual + linear_any(self.out_proj, core_norm)

        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(x)

    def forward_block(self, x: torch.Tensor, state: DecodeState) -> torch.Tensor:
        ops = cuda_ops()
        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        q, k, v, z, b, a = self.project_block(x)
        conv_in = torch.cat([q.reshape(x.size(0), -1), k.reshape(x.size(0), -1), v.reshape(x.size(0), -1)], dim=-1).contiguous()
        conv_out = depthwise_conv_update_block(state.gdn_conv_states[self.layer], conv_in, self.conv_weight, self.conv_bias)
        key_dim = self.cfg.linear_key_head_dim * self.cfg.linear_num_key_heads
        value_dim = self.cfg.linear_value_head_dim * self.cfg.linear_num_value_heads
        q2, k2, v2 = torch.split(conv_out, [key_dim, key_dim, value_dim], dim=-1)
        q = q2.view(x.size(0), self.cfg.linear_num_key_heads, self.cfg.linear_key_head_dim).contiguous()
        k = k2.view_as(q).contiguous()
        v = v2.view(x.size(0), self.cfg.linear_num_value_heads, self.cfg.linear_value_head_dim).contiguous()
        beta = torch.sigmoid(b).to(torch.float16).contiguous()
        g = (-torch.exp(self.A_log.to(a.device))[None, :] * F.softplus(a.float() + self.dt_bias.to(a.device)[None, :])).contiguous()
        core = ops.gdn_recurrent_scan(q, k, v, g, beta, state.gdn_states[self.layer])
        core_norm = qwen_gdn_norm_silu_gate(core, self.gdn_norm_w, z, self.cfg.rms_norm_eps).reshape(x.size(0), -1).contiguous()
        x = residual + linear_any(self.out_proj, core_norm)
        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp.forward_block(x)


class QwenBurstAttentionLayer:
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
        self.mlp = QwenBurstMLP(cfg, weights, layer)

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
            q = qwen_rmsnorm_torch(q, self.q_norm, self.cfg.rms_norm_eps)
        if self.k_norm is not None:
            k = qwen_rmsnorm_torch(k, self.k_norm, self.cfg.rms_norm_eps)
        q, k = apply_rope_single_token(q, k, pos=state.pos, rope_dim=self.cfg.rope_dim, rope_theta=self.cfg.rope_theta)

        state.append_attention_kv(self.layer, k.contiguous(), v.contiguous())
        k_cache, v_cache, length = state.attention_kv_view(self.layer)
        att = ops.attention_decode_fp16(q.contiguous(), k_cache, v_cache, length, self.cfg.attention_head_dim ** -0.5)
        att_flat = att.reshape(-1).contiguous()
        if gate_flat is not None:
            att_flat = att_flat * torch.sigmoid(gate_flat.to(att_flat.dtype))
        x = residual + linear_any(self.o_proj, att_flat)

        residual = x
        x = qwen_rmsnorm(x.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(x)

    def forward_block(self, x: torch.Tensor, state: DecodeState, *, base_pos: int, base_kv_len: int) -> torch.Tensor:
        out = []
        for offset, row in enumerate(x):
            logical_pos = base_pos + offset
            live_len = min(base_kv_len + offset + 1, state.max_seq_len)
            ops = cuda_ops()
            residual = row.contiguous()
            h = qwen_rmsnorm(residual, self.input_norm, self.cfg.rms_norm_eps)
            if self.qkv_proj is not None:
                qkv_all = linear_any(self.qkv_proj, h)
                kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
                q_all, k_all, v_all = torch.split(qkv_all, [self.qkv_q_rows, kv_rows, kv_rows], dim=0)
            else:
                q_all = linear_any(self.q_proj, h)  # type: ignore[arg-type]
                k_all = linear_any(self.k_proj, h)  # type: ignore[arg-type]
                v_all = linear_any(self.v_proj, h)  # type: ignore[arg-type]
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
                q = qwen_rmsnorm_torch(q, self.q_norm, self.cfg.rms_norm_eps)
            if self.k_norm is not None:
                k = qwen_rmsnorm_torch(k, self.k_norm, self.cfg.rms_norm_eps)
            q, k = apply_rope_single_token(q, k, pos=logical_pos, rope_dim=self.cfg.rope_dim, rope_theta=self.cfg.rope_theta)
            state.append_attention_kv_at(self.layer, k.contiguous(), v.contiguous(), logical_pos=logical_pos)
            k_cache, v_cache, length = state.attention_kv_view_at(self.layer, logical_pos=logical_pos, live_len=live_len)
            att = ops.attention_decode_fp16(q.contiguous(), k_cache, v_cache, length, self.cfg.attention_head_dim ** -0.5)
            att_flat = att.reshape(-1).contiguous()
            if gate_flat is not None:
                att_flat = att_flat * torch.sigmoid(gate_flat.to(att_flat.dtype))
            h = residual + linear_any(self.o_proj, att_flat)
            residual = h
            h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
            out.append(residual + self.mlp(h))
        return torch.stack(out, dim=0).contiguous()


class QwenBurstModel:
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
        self.layers = []
        for i in range(self.cfg.num_layers):
            if self.cfg.layer_type(i) == "gdn":
                self.layers.append(QwenBurstGDNLayer(self.cfg, wr, i, self.device))
            else:
                self.layers.append(QwenBurstAttentionLayer(self.cfg, wr, i))

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def forward_one(
        self,
        token: torch.Tensor | int,
        state: DecodeState,
        *,
        use_mtp: bool = False,
        return_hidden: bool = False,
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
                x = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
                if hidden_tap_layers is not None:
                    return None, x, hidden_taps
                return None, x
            if use_mtp:
                return None, []
            if hidden_tap_layers is not None:
                return None, hidden_taps
            return None
        x = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
        logits = lowbit_linear_on_device(self.lm_head, x, self.device) if isinstance(self.lm_head, LowBitTensor) else linear_any(self.lm_head, x)
        if return_hidden and use_mtp:
            if hidden_tap_layers is not None:
                return logits, [], x, hidden_taps
            return logits, [], x
        if return_hidden:
            if hidden_tap_layers is not None:
                return logits, x, hidden_taps
            return logits, x
        if use_mtp:
            if hidden_tap_layers is not None:
                return logits, [], hidden_taps
            return logits, []
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
        commit: bool = False,
        expected_next_tokens: Sequence[int] | None = None,
    ) -> BlockForwardResult:
        """Verify a candidate block under one canonical qwenburst target API."""
        if expected_next_tokens is not None:
            return self._forward_block_verify_expected(
                tokens,
                state,
                hidden_tap_layers=hidden_tap_layers,
                commit=commit,
                expected_next_tokens=expected_next_tokens,
            )
        branch = state if commit else state.fork()
        result = self._forward_block_raw(
            tokens,
            branch,
            hidden_tap_layers=hidden_tap_layers,
            return_logits=return_logits,
        )
        if commit and branch is not state:
            state.copy_from_(branch)
        return result

    @torch.no_grad()
    def _forward_block_raw(
        self,
        tokens: Sequence[int],
        state: DecodeState,
        *,
        hidden_tap_layers: Sequence[int] | None,
        return_logits: bool,
    ) -> BlockForwardResult:
        token_list = [int(t) for t in tokens]
        if not token_list:
            return BlockForwardResult(logits=[], hidden_taps=[], state=state, next_ids=[], accepted_next=0)
        token_tensor = torch.tensor(token_list, device=self.device, dtype=torch.long)
        x = torch.stack([embed_lookup(self.embed, tid).to(self.device, non_blocking=True) for tid in token_tensor], dim=0).contiguous()
        tap_set = set(hidden_tap_layers or ())
        taps_by_token: list[list[torch.Tensor]] = [[] for _ in token_list]
        base_pos = state.pos
        base_kv_len = state.kv_len
        for layer_idx, layer in enumerate(self.layers):
            if isinstance(layer, QwenBurstAttentionLayer):
                x = layer.forward_block(x, state, base_pos=base_pos, base_kv_len=base_kv_len)
            else:
                x = layer.forward_block(x, state)
            if layer_idx in tap_set:
                for i in range(x.size(0)):
                    taps_by_token[i].append(x[i].detach())
        for _ in token_list:
            state.finish_token()
        logits_out: list[torch.Tensor] = []
        if return_logits:
            h = qwen_rmsnorm(x.contiguous(), self.final_norm, self.cfg.rms_norm_eps)
            logits = linear_any(self.lm_head, h)
            logits_out = [row.contiguous() for row in logits]
        return BlockForwardResult(
            logits=logits_out,
            hidden_taps=taps_by_token,
            state=state,
            next_ids=[],
            accepted_next=0,
        )

    @torch.no_grad()
    def _forward_block_sequential(
        self,
        tokens: Sequence[int],
        state: DecodeState,
        *,
        hidden_tap_layers: Sequence[int] | None,
        return_logits: bool,
        commit: bool,
    ) -> BlockForwardResult:
        branch = state if commit else state.fork()
        logits_out: list[torch.Tensor] = []
        taps_out: list[list[torch.Tensor]] = []
        next_ids: list[int] = []
        accepted_next = 0
        for tid in tokens:
            if hidden_tap_layers is None:
                logits = self.forward_one(tid, branch, return_logits=return_logits)
                if return_logits:
                    logits_out.append(logits)
                taps_out.append([])
            else:
                result = self.forward_one(
                    tid,
                    branch,
                    return_logits=return_logits,
                    hidden_tap_layers=hidden_tap_layers,
                )
                if return_logits:
                    logits, taps = result
                    logits_out.append(logits)
                else:
                    _, taps = result
                taps_out.append(taps)
        if commit and branch is not state:
            state.copy_from_(branch)
        return BlockForwardResult(
            logits=logits_out,
            hidden_taps=taps_out,
            state=branch,
            next_ids=next_ids,
            accepted_next=accepted_next,
        )

    @torch.no_grad()
    def _forward_block_verify_expected(
        self,
        tokens: Sequence[int],
        state: DecodeState,
        *,
        hidden_tap_layers: Sequence[int] | None,
        commit: bool,
        expected_next_tokens: Sequence[int],
    ) -> BlockForwardResult:
        """Early-stop verifier for speculative decoding.

        This path intentionally does not run the full candidate block. It
        advances the target state until the first rejected next-token prediction,
        so the live state already equals the accepted prefix and no replay is
        required. A full C++/CUDA block verifier would still need this same
        prefix-commit contract unless the GDN/attention states become
        reversible or checkpointed per candidate token.
        """
        token_list = [int(t) for t in tokens]
        expected = [int(t) for t in expected_next_tokens]
        if len(expected) > max(0, len(token_list) - 1):
            raise ValueError("expected_next_tokens cannot be longer than tokens - 1")

        branch = state.fork()
        verified = self._forward_block_raw(
            token_list,
            branch,
            hidden_tap_layers=hidden_tap_layers,
            return_logits=True,
        )
        next_ids: list[int] = []
        accepted_next = 0
        argmax = cuda_ops().argmax if self.device.type == "cuda" else None

        for i, logits in enumerate(verified.logits):
            if argmax is None:
                pred = int(torch.argmax(logits, dim=-1).item())
            else:
                pred = int(argmax(logits.contiguous()).item())
            next_ids.append(pred)
            if i < len(expected):
                if pred != expected[i]:
                    break
                accepted_next += 1

        commit_n = min(accepted_next + 1, len(token_list))
        result = verified
        if commit:
            if commit_n == len(token_list):
                state.copy_from_(branch)
                result = verified
            else:
                result = self._forward_block_raw(
                    token_list[:commit_n],
                    state,
                    hidden_tap_layers=hidden_tap_layers,
                    return_logits=True,
                )
        return BlockForwardResult(
            logits=result.logits,
            hidden_taps=result.hidden_taps,
            state=state if commit else branch,
            next_ids=next_ids,
            accepted_next=accepted_next,
        )
