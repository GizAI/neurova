from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


_USE_LIGER_KERNELS = os.environ.get("SANEFLOW_USE_LIGER_KERNELS", "").strip().lower() in {"1", "true", "yes", "on"}
_LIGER_RMS_NORM_FUNCTION = None
_LIGER_SILU_MUL_FUNCTION = None
if _USE_LIGER_KERNELS:
    try:
        if not hasattr(torch.distributed, "tensor"):
            try:
                import torch.distributed.tensor as dist_tensor

                torch.distributed.tensor = dist_tensor
            except Exception:
                torch.distributed.tensor = types.SimpleNamespace(DTensor=type("DTensor", (), {}))
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        from liger_kernel.transformers.swiglu import LigerSiLUMulFunction

        _LIGER_RMS_NORM_FUNCTION = LigerRMSNormFunction
        _LIGER_SILU_MUL_FUNCTION = LigerSiLUMulFunction
    except Exception:
        _LIGER_RMS_NORM_FUNCTION = None
        _LIGER_SILU_MUL_FUNCTION = None


@dataclass
class SaneFlowConfig:
    vocab_size: int
    model_type: str = "saneflow"
    d_embed: int = 0
    d_model: int = 384
    n_layer: int = 8
    n_heads: int = 6
    n_kv_heads: int = 0
    d_ff: int = 1024
    rope_theta: float = 10000.0
    qk_norm: bool = False
    conv_kernel: int = 5
    syntax_mix_version: str = "v1"
    syntax_kernels: tuple[int, ...] = (3, 7, 15)
    dropout: float = 0.0
    state_mixer_version: str = "v1"
    state_clip: float = 0.0
    state_zoneout: float = 0.0
    attention_interval: int = 0
    attention_window: int = 64
    thought_slots: int = 0
    thought_chunk: int = 1
    thought_start_layer: int = 0
    landmark_interval: int = 0
    landmark_chunk: int = 64
    landmark_max: int = 64
    tokenizer_backend: str = "saneflow_bpe"
    tokenizer_path: str = "tokenizers/saneflow_fineweb_edu_16k"
    tokenizer_sha256: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _LIGER_RMS_NORM_FUNCTION is not None and x.is_cuda:
            return _LIGER_RMS_NORM_FUNCTION.apply(x, self.weight, self.eps, 0.0, "llama", False, None)
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dw = nn.Conv1d(dim, dim, kernel_size, groups=dim)
        self.pw = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x.transpose(1, 2), (self.kernel_size - 1, 0))
        y = self.dw(y).transpose(1, 2)
        y, gate = self.pw(y).chunk(2, dim=-1)
        return y * torch.sigmoid(gate)

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.dw.in_channels, self.kernel_size - 1, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = torch.cat([cache, x.unsqueeze(-1)], dim=-1)
        weight = self.dw.weight[:, 0, :].to(dtype=x.dtype)
        y = (window * weight.unsqueeze(0)).sum(dim=-1)
        if self.dw.bias is not None:
            y = y + self.dw.bias.to(dtype=x.dtype).unsqueeze(0)
        y, gate = self.pw(y).chunk(2, dim=-1)
        new_cache = window[:, :, 1:].contiguous()
        return y * torch.sigmoid(gate), new_cache


class MultiKernelSyntaxMix(nn.Module):
    def __init__(self, dim: int, kernels: tuple[int, ...]) -> None:
        super().__init__()
        if not kernels:
            raise ValueError("syntax_kernels must not be empty")
        for kernel in kernels:
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError(f"syntax kernel must be a positive odd integer, got {kernel}")
        self.kernels = tuple(kernels)
        self.dw = nn.ModuleList([nn.Conv1d(dim, dim, kernel, groups=dim) for kernel in self.kernels])
        self.pw = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)
        ys = []
        for kernel, conv in zip(self.kernels, self.dw):
            ys.append(conv(F.pad(xt, (kernel - 1, 0))).transpose(1, 2))
        y = torch.stack(ys, dim=0).mean(dim=0)
        y, gate = self.pw(y).chunk(2, dim=-1)
        return y * torch.sigmoid(gate)

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        return [
            torch.zeros(batch_size, conv.in_channels, kernel - 1, device=device, dtype=dtype)
            for kernel, conv in zip(self.kernels, self.dw)
        ]

    def step(self, x: torch.Tensor, cache: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        ys = []
        new_cache = []
        for conv, old in zip(self.dw, cache):
            window = torch.cat([old, x.unsqueeze(-1)], dim=-1)
            weight = conv.weight[:, 0, :].to(dtype=x.dtype)
            y = (window * weight.unsqueeze(0)).sum(dim=-1)
            if conv.bias is not None:
                y = y + conv.bias.to(dtype=x.dtype).unsqueeze(0)
            ys.append(y)
            new_cache.append(window[:, :, 1:].contiguous())
        y = torch.stack(ys, dim=0).mean(dim=0)
        y, gate = self.pw(y).chunk(2, dim=-1)
        return y * torch.sigmoid(gate), new_cache


class GatedStateMixer(nn.Module):
    def __init__(self, dim: int, heads: int, version: str = "v1", state_clip: float = 0.0, state_zoneout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by n_heads={heads}")
        if version not in {"v1", "v2", "v2_fixed"}:
            raise ValueError(f"unknown state mixer version={version!r}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.version = version
        self.state_clip = float(state_clip)
        self.state_zoneout = float(state_zoneout)
        self.in_proj = nn.Linear(dim, dim * (5 if version == "v2" else 4))
        self.out = nn.Linear(dim, dim)
        self.time_bias = nn.Parameter(torch.empty(heads, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Log-spaced forget timescales. Early channels are reactive, late
        # channels are slow memory. sigmoid(time_bias) is the base forget.
        taus = torch.logspace(0.0, 3.0, self.head_dim)
        forget = torch.exp(-1.0 / taus).clamp(0.05, 0.999)
        bias = torch.logit(forget).unsqueeze(0).repeat(self.heads, 1)
        with torch.no_grad():
            self.time_bias.copy_(bias)

    def init_gates(self) -> None:
        if self.in_proj.bias is None:
            return
        with torch.no_grad():
            self.in_proj.bias.zero_()
            if self.version in {"v2", "v2_fixed"}:
                # v2: value, erase_delta, [legacy forget_delta], write_gate, out_gate
                # v2_fixed: value, erase_delta, write_gate, out_gate
                start = self.dim * (3 if self.version == "v2" else 2)
                end = start + self.dim
                self.in_proj.bias[start:end].fill_(-2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        if self.version == "v1":
            value, forget_delta, write_gate, out_gate = self.in_proj(x).chunk(4, dim=-1)
            erase_delta = forget_delta
        elif self.version == "v2":
            value, erase_delta, forget_delta, write_gate, out_gate = self.in_proj(x).chunk(5, dim=-1)
        else:
            value, erase_delta, write_gate, out_gate = self.in_proj(x).chunk(4, dim=-1)
            forget_delta = erase_delta
        value = torch.tanh(value).view(batch, length, self.heads, self.head_dim)
        erase_delta = erase_delta.view(batch, length, self.heads, self.head_dim)
        if self.version == "v1":
            forget_delta = forget_delta.view(batch, length, self.heads, self.head_dim)
        write_gate = write_gate.view(batch, length, self.heads, self.head_dim)
        out_gate = out_gate.view(batch, length, self.heads, self.head_dim)

        state = torch.zeros(batch, self.heads, self.head_dim, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        base = self.time_bias.to(dtype=x.dtype, device=x.device)
        for t in range(length):
            write = torch.sigmoid(write_gate[:, t])
            if self.version == "v1":
                forget = torch.sigmoid(base.unsqueeze(0) + 0.5 * torch.tanh(forget_delta[:, t]))
                new_state = forget * state + (1.0 - forget) * write * value[:, t]
            else:
                # v2 follows the Gated DeltaNet-2/Kimi lesson: lifespan
                # control and value commit should be independent decisions.
                erase = torch.sigmoid(base.unsqueeze(0) + 0.5 * torch.tanh(erase_delta[:, t]))
                new_state = erase * state + write * value[:, t]
            if self.state_clip > 0:
                new_state = new_state.clamp(-self.state_clip, self.state_clip)
            if self.training and self.state_zoneout > 0:
                keep = torch.rand_like(new_state) < self.state_zoneout
                new_state = torch.where(keep, state, new_state)
            state = new_state
            ys.append((torch.sigmoid(out_gate[:, t]) * state).reshape(batch, self.dim))
        return self.out(torch.stack(ys, dim=1))

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.heads, self.head_dim, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.version == "v1":
            value, forget_delta, write_gate, out_gate = self.in_proj(x).chunk(4, dim=-1)
            erase_delta = forget_delta
        elif self.version == "v2":
            value, erase_delta, forget_delta, write_gate, out_gate = self.in_proj(x).chunk(5, dim=-1)
        else:
            value, erase_delta, write_gate, out_gate = self.in_proj(x).chunk(4, dim=-1)
            forget_delta = erase_delta
        batch = x.shape[0]
        value = torch.tanh(value).view(batch, self.heads, self.head_dim)
        erase_delta = erase_delta.view(batch, self.heads, self.head_dim)
        if self.version == "v1":
            forget_delta = forget_delta.view(batch, self.heads, self.head_dim)
        write_gate = write_gate.view(batch, self.heads, self.head_dim)
        out_gate = out_gate.view(batch, self.heads, self.head_dim)
        base = self.time_bias.to(dtype=x.dtype, device=x.device)
        write = torch.sigmoid(write_gate)
        if self.version == "v1":
            forget = torch.sigmoid(base.unsqueeze(0) + 0.5 * torch.tanh(forget_delta))
            new_state = forget * state + (1.0 - forget) * write * value
        else:
            erase = torch.sigmoid(base.unsqueeze(0) + 0.5 * torch.tanh(erase_delta))
            new_state = erase * state + write * value
        if self.state_clip > 0:
            new_state = new_state.clamp(-self.state_clip, self.state_clip)
        y = (torch.sigmoid(out_gate) * new_state).reshape(batch, self.dim)
        return self.out(y), new_state


class DeltaMatrixStateMixer(nn.Module):
    def __init__(self, dim: int, heads: int, state_clip: float = 0.0, state_zoneout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by n_heads={heads}")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.state_clip = float(state_clip)
        self.state_zoneout = float(state_zoneout)
        self.in_proj = nn.Linear(dim, dim * 6)
        self.out = nn.Linear(dim, dim)
        self.time_bias = nn.Parameter(torch.empty(heads, self.head_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        taus = torch.logspace(0.0, 3.0, self.head_dim)
        forget = torch.exp(-1.0 / taus).clamp(0.05, 0.999)
        bias = torch.logit(forget).unsqueeze(0).repeat(self.heads, 1)
        with torch.no_grad():
            self.time_bias.copy_(bias)

    def init_gates(self) -> None:
        if self.in_proj.bias is None:
            return
        with torch.no_grad():
            self.in_proj.bias.zero_()
            # chunks: q, k, value, erase_delta, write_gate, out_gate
            self.in_proj.bias[self.dim * 4 : self.dim * 5].fill_(-2.0)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, value, erase_delta, write_gate, out_gate = self.in_proj(x).chunk(6, dim=-1)
        shape = (*x.shape[:-1], self.heads, self.head_dim)
        q = F.normalize(torch.tanh(q).view(shape), dim=-1)
        k = F.normalize(torch.tanh(k).view(shape), dim=-1)
        value = torch.tanh(value).view(shape)
        erase_delta = erase_delta.view(shape)
        write_gate = write_gate.view(shape)
        out_gate = out_gate.view(shape)
        return q, k, value, erase_delta, write_gate, out_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, value, erase_delta, write_gate, out_gate = self._project(x)
        state = torch.zeros(batch, self.heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
        ys: list[torch.Tensor] = []
        base = self.time_bias.to(dtype=x.dtype, device=x.device)
        for t in range(length):
            read = torch.einsum("bhr,bhrd->bhd", q[:, t], state)
            erase = torch.sigmoid(base.unsqueeze(0) + 0.5 * torch.tanh(erase_delta[:, t])).unsqueeze(-1)
            write = torch.sigmoid(write_gate[:, t]).unsqueeze(-1)
            outer = k[:, t].unsqueeze(-1) * value[:, t].unsqueeze(-2)
            new_state = erase * state + write * outer
            if self.state_clip > 0:
                new_state = new_state.clamp(-self.state_clip, self.state_clip)
            if self.training and self.state_zoneout > 0:
                keep = torch.rand_like(new_state) < self.state_zoneout
                new_state = torch.where(keep, state, new_state)
            state = new_state
            ys.append((torch.sigmoid(out_gate[:, t]) * (read + value[:, t])).reshape(batch, self.dim))
        return self.out(torch.stack(ys, dim=1))

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.heads, self.head_dim, self.head_dim, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, value, erase_delta, write_gate, out_gate = self._project(x)
        read = torch.einsum("bhr,bhrd->bhd", q, state)
        base = self.time_bias.to(dtype=x.dtype, device=x.device)
        erase = torch.sigmoid(base.unsqueeze(0) + 0.5 * torch.tanh(erase_delta)).unsqueeze(-1)
        write = torch.sigmoid(write_gate).unsqueeze(-1)
        outer = k.unsqueeze(-1) * value.unsqueeze(-2)
        new_state = erase * state + write * outer
        if self.state_clip > 0:
            new_state = new_state.clamp(-self.state_clip, self.state_clip)
        y = (torch.sigmoid(out_gate) * (read + value)).reshape(x.shape[0], self.dim)
        return self.out(y), new_state


class ZeroStateMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def init_gates(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(batch_size, 0, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros_like(x), state


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden * 2)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.w1(x).chunk(2, dim=-1)
        if _LIGER_SILU_MUL_FUNCTION is not None and x.is_cuda:
            return self.w2(_LIGER_SILU_MUL_FUNCTION.apply(gate, x))
        return self.w2(F.silu(gate) * x)


class BiasFreeSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden * 2, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.w1(x).chunk(2, dim=-1)
        if _LIGER_SILU_MUL_FUNCTION is not None and x.is_cuda:
            return self.w2(_LIGER_SILU_MUL_FUNCTION.apply(gate, x))
        return self.w2(F.silu(gate) * x)


class SparseAttentionIsland(nn.Module):
    def __init__(self, dim: int, heads: int, window: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"d_model={dim} must be divisible by n_heads={heads}")
        if window <= 0:
            raise ValueError("attention_window must be positive")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window = int(window)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    def _rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.head_dim % 2 != 0:
            return x
        freqs = torch.outer(positions.to(dtype=self.rope_inv_freq.dtype), self.rope_inv_freq).to(device=x.device)
        sin = freqs.sin().to(dtype=x.dtype).view(1, 1, -1, self.head_dim // 2)
        cos = freqs.cos().to(dtype=x.dtype).view(1, 1, -1, self.head_dim // 2)
        x_pair = x.view(*x.shape[:-1], self.head_dim // 2, 2)
        x0 = x_pair[..., 0]
        x1 = x_pair[..., 1]
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split(q).transpose(1, 2)
        k = self._split(k).transpose(1, 2)
        v = self._split(v).transpose(1, 2)
        positions = torch.arange(length, device=x.device)
        q = self._rope(q, positions)
        k = self._rope(k, positions)
        chunk = min(length, max(1, self.window))
        ys = []
        scale = self.head_dim ** -0.5
        for start in range(0, length, chunk):
            end = min(start + chunk, length)
            key_start = max(0, start - self.window + 1)
            q_chunk = q[:, :, start:end]
            k_chunk = k[:, :, key_start:end]
            v_chunk = v[:, :, key_start:end]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
            q_pos = torch.arange(start, end, device=x.device)
            k_pos = torch.arange(key_start, end, device=x.device)
            valid = (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)) & (
                k_pos.unsqueeze(0) >= (q_pos.unsqueeze(1) - self.window + 1)
            )
            scores = scores.masked_fill(~valid.view(1, 1, end - start, end - key_start), torch.finfo(scores.dtype).min)
            ys.append(torch.matmul(F.softmax(scores, dim=-1), v_chunk))
        y = torch.cat(ys, dim=2)
        y = y.transpose(1, 2).contiguous().view(batch, length, self.dim)
        return self.out(y)

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "k": torch.zeros(batch_size, self.window, self.heads, self.head_dim, device=device, dtype=dtype),
            "v": torch.zeros(batch_size, self.window, self.heads, self.head_dim, device=device, dtype=dtype),
            "len": torch.zeros((), device=device, dtype=torch.long),
        }

    def step(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self._split(q)
        k = self._split(k)
        v = self._split(v)
        old_len = int(cache["len"].item())
        pos = torch.tensor([old_len], device=x.device)
        q = self._rope(q.unsqueeze(2), pos).squeeze(2)
        k = self._rope(k.unsqueeze(2), pos).squeeze(2)
        slot = old_len % self.window
        k_cache = cache["k"].clone()
        v_cache = cache["v"].clone()
        k_cache[:, slot] = k
        v_cache[:, slot] = v
        new_len = old_len + 1
        take = min(new_len, self.window)
        positions = [(new_len - take + i) % self.window for i in range(take)]
        keys = k_cache[:, positions]
        values = v_cache[:, positions]
        scores = torch.einsum("bhd,bthd->bht", q, keys) * (self.head_dim ** -0.5)
        probs = F.softmax(scores, dim=-1)
        y = torch.einsum("bht,bthd->bhd", probs, values).reshape(x.shape[0], self.dim)
        return self.out(y), {"k": k_cache, "v": v_cache, "len": cache["len"] + 1}


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, heads, length, head_dim = x.shape
    return x[:, :, None, :, :].expand(batch, heads, repeats, length, head_dim).reshape(
        batch, heads * repeats, length, head_dim
    )


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        if self.head_dim % 2 != 0:
            inv_freq = torch.empty(0)
        else:
            inv_freq = 1.0 / (float(theta) ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.inv_freq.numel() == 0:
            return x
        freqs = torch.outer(positions.to(dtype=self.inv_freq.dtype), self.inv_freq).to(device=x.device)
        sin = freqs.sin().to(dtype=x.dtype).view(1, 1, positions.numel(), self.head_dim // 2)
        cos = freqs.cos().to(dtype=x.dtype).view(1, 1, positions.numel(), self.head_dim // 2)
        pairs = x.view(*x.shape[:-1], self.head_dim // 2, 2)
        x0 = pairs[..., 0]
        x1 = pairs[..., 1]
        return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


class DenseGQAAttention(nn.Module):
    def __init__(self, cfg: SaneFlowConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} must be divisible by n_heads={cfg.n_heads}")
        kv_heads = cfg.n_kv_heads or cfg.n_heads
        if cfg.n_heads % kv_heads != 0:
            raise ValueError(f"n_heads={cfg.n_heads} must be divisible by n_kv_heads={kv_heads}")
        self.dim = cfg.d_model
        self.n_heads = cfg.n_heads
        self.n_kv_heads = kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if cfg.qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, cfg.rope_theta)

    def _project(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        positions = torch.arange(length, device=x.device)
        q, k, v = self._project(x, positions)
        repeats = self.n_heads // self.n_kv_heads
        k = _repeat_kv(k, repeats)
        v = _repeat_kv(v, repeats)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=True,
        )
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, length, self.dim))

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        max_cache_len = 8192
        return {
            "k": torch.empty(batch_size, self.n_kv_heads, max_cache_len, self.head_dim, device=device, dtype=dtype),
            "v": torch.empty(batch_size, self.n_kv_heads, max_cache_len, self.head_dim, device=device, dtype=dtype),
            "len": torch.zeros((), device=device, dtype=torch.long),
            "positions": torch.arange(max_cache_len, device=device),
        }

    def step(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = x.shape[0]
        length = int(cache["len"].item())
        if length >= cache["k"].shape[2]:
            grow_by = max(1024, cache["k"].shape[2])
            k_cache = torch.empty(batch, self.n_kv_heads, cache["k"].shape[2] + grow_by, self.head_dim, device=x.device, dtype=cache["k"].dtype)
            v_cache = torch.empty_like(k_cache)
            k_cache[:, :, :length].copy_(cache["k"][:, :, :length])
            v_cache[:, :, :length].copy_(cache["v"][:, :, :length])
            cache = {"k": k_cache, "v": v_cache, "len": cache["len"], "positions": torch.arange(k_cache.shape[2], device=x.device)}
        positions = cache["positions"][length : length + 1]
        q, k, v = self._project(x[:, None, :], positions)
        cache["k"][:, :, length : length + 1].copy_(k)
        cache["v"][:, :, length : length + 1].copy_(v)
        next_len = length + 1
        repeats = self.n_heads // self.n_kv_heads
        y = F.scaled_dot_product_attention(
            q,
            _repeat_kv(cache["k"][:, :, :next_len], repeats),
            _repeat_kv(cache["v"][:, :, :next_len], repeats),
            dropout_p=0.0,
            is_causal=False,
        )
        cache["len"].add_(1)
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, self.dim)), cache


class ThoughtSlotMixer(nn.Module):
    def __init__(self, dim: int, n_slots: int, chunk: int = 1) -> None:
        super().__init__()
        if n_slots <= 0:
            raise ValueError("thought_slots must be positive")
        if chunk <= 0:
            raise ValueError("thought_chunk must be positive")
        self.dim = dim
        self.n_slots = int(n_slots)
        self.chunk = int(chunk)
        self.slot_init = nn.Parameter(torch.zeros(n_slots, dim))
        self.norm_token = RMSNorm(dim)
        self.norm_slot = RMSNorm(dim)
        self.read_q = nn.Linear(dim, dim, bias=False)
        self.slot_k = nn.Linear(dim, dim, bias=False)
        self.slot_v = nn.Linear(dim, dim, bias=False)
        self.write_gate = nn.Linear(dim, n_slots)
        self.write_value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.write_scale = nn.Parameter(torch.tensor(0.1))

    def _initial_slots(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.slot_init.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _read(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        q = self.read_q(self.norm_token(x))
        k = self.slot_k(self.norm_slot(slots))
        v = self.slot_v(self.norm_slot(slots))
        scores = torch.einsum("bd,bmd->bm", q, k) * (self.dim ** -0.5)
        read = torch.einsum("bm,bmd->bd", F.softmax(scores, dim=-1), v)
        return self.out(read)

    def _read_many(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        q = self.read_q(self.norm_token(x))
        norm_slots = self.norm_slot(slots)
        k = self.slot_k(norm_slots)
        v = self.slot_v(norm_slots)
        scores = torch.einsum("btd,bmd->btm", q, k) * (self.dim ** -0.5)
        read = torch.einsum("btm,bmd->btd", F.softmax(scores, dim=-1), v)
        return self.out(read)

    def _write(self, x: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        token = self.norm_token(x)
        gate = torch.sigmoid(self.write_gate(token)).unsqueeze(-1)
        value = torch.tanh(self.write_value(token)).unsqueeze(1)
        updated = slots + self.write_scale.tanh() * gate * (value - slots)
        return self.norm_slot(updated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        slots = self._initial_slots(batch, device=x.device, dtype=x.dtype)
        if self.chunk > 1:
            ys = []
            for start in range(0, length, self.chunk):
                end = min(start + self.chunk, length)
                chunk = x[:, start:end]
                ys.append(self._read_many(chunk, slots))
                slots = self._write(chunk.mean(dim=1), slots)
            return torch.cat(ys, dim=1)
        ys = []
        for t in range(length):
            ys.append(self._read(x[:, t], slots))
            slots = self._write(x[:, t], slots)
        return torch.stack(ys, dim=1)

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | dict[str, torch.Tensor]:
        slots = self._initial_slots(batch_size, device=device, dtype=dtype)
        if self.chunk <= 1:
            return slots
        return {
            "slots": slots,
            "pos": torch.zeros((), device=device, dtype=torch.long),
            "acc": torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
        }

    def step(self, x: torch.Tensor, slots: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | dict[str, torch.Tensor]]:
        if isinstance(slots, dict):
            y = self._read(x, slots["slots"])
            pos = int(slots["pos"].item()) + 1
            acc = slots["acc"] + x
            next_slots = slots["slots"]
            if pos % self.chunk == 0:
                next_slots = self._write(acc / float(self.chunk), next_slots)
                acc = torch.zeros_like(acc)
            return y, {"slots": next_slots, "pos": slots["pos"] + 1, "acc": acc}
        y = self._read(x, slots)
        return y, self._write(x, slots)


class LandmarkMemory(nn.Module):
    def __init__(self, dim: int, chunk: int, max_landmarks: int) -> None:
        super().__init__()
        if chunk <= 0:
            raise ValueError("landmark_chunk must be positive")
        if max_landmarks <= 0:
            raise ValueError("landmark_max must be positive")
        self.dim = dim
        self.chunk = int(chunk)
        self.max_landmarks = int(max_landmarks)
        self.norm_token = RMSNorm(dim)
        self.norm_memory = RMSNorm(dim)
        self.summary = nn.Linear(dim, dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)

    def _empty_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "landmarks": torch.zeros(batch_size, self.max_landmarks, self.dim, device=device, dtype=dtype),
            "count": torch.zeros((), device=device, dtype=torch.long),
            "pos": torch.zeros((), device=device, dtype=torch.long),
            "acc": torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
        }

    def _read(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        count = int(cache["count"].item())
        take = min(count, self.max_landmarks)
        if take == 0:
            return torch.zeros_like(x)
        memory = cache["landmarks"][:, :take]
        q = self.q(self.norm_token(x))
        k = self.k(self.norm_memory(memory))
        v = self.v(self.norm_memory(memory))
        scores = torch.einsum("bd,bmd->bm", q, k) * (self.dim ** -0.5)
        read = torch.einsum("bm,bmd->bd", F.softmax(scores, dim=-1), v)
        return self.out(read)

    def _read_many(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
        count = int(cache["count"].item())
        take = min(count, self.max_landmarks)
        if take == 0:
            return torch.zeros_like(x)
        memory = cache["landmarks"][:, :take]
        q = self.q(self.norm_token(x))
        norm_memory = self.norm_memory(memory)
        k = self.k(norm_memory)
        v = self.v(norm_memory)
        scores = torch.einsum("btd,bmd->btm", q, k) * (self.dim ** -0.5)
        read = torch.einsum("btm,bmd->btd", F.softmax(scores, dim=-1), v)
        return self.out(read)

    def _write(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pos = int(cache["pos"].item())
        new_pos = pos + 1
        acc = cache["acc"] + self.norm_token(x)
        landmarks = cache["landmarks"]
        count = cache["count"]
        if new_pos % self.chunk == 0:
            slot = int(count.item()) % self.max_landmarks
            summary = self.summary(acc / float(self.chunk))
            landmarks = landmarks.clone()
            landmarks[:, slot] = summary
            acc = torch.zeros_like(acc)
            count = count + 1
        return {"landmarks": landmarks, "count": count, "pos": cache["pos"] + 1, "acc": acc}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        cache = self._empty_cache(batch, device=x.device, dtype=x.dtype)
        ys = []
        for start in range(0, length, self.chunk):
            end = min(start + self.chunk, length)
            chunk = x[:, start:end]
            ys.append(self._read_many(chunk, cache))
            if end - start == self.chunk:
                acc = self.norm_token(chunk).sum(dim=1)
                landmarks = cache["landmarks"]
                slot = int(cache["count"].item()) % self.max_landmarks
                landmarks = landmarks.clone()
                landmarks[:, slot] = self.summary(acc / float(self.chunk))
                cache = {
                    "landmarks": landmarks,
                    "count": cache["count"] + 1,
                    "pos": cache["pos"] + self.chunk,
                    "acc": torch.zeros_like(cache["acc"]),
                }
        return torch.cat(ys, dim=1)

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return self._empty_cache(batch_size, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = self._read(x, cache)
        return y, self._write(x, cache)


class SaneFlowBlock(nn.Module):
    def __init__(self, cfg: SaneFlowConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm_local = RMSNorm(cfg.d_model)
        if cfg.syntax_mix_version == "v1":
            self.local = CausalDepthwiseConv(cfg.d_model, cfg.conv_kernel)
        elif cfg.syntax_mix_version == "v2":
            self.local = MultiKernelSyntaxMix(cfg.d_model, tuple(cfg.syntax_kernels))
        else:
            raise ValueError(f"unknown syntax_mix_version={cfg.syntax_mix_version!r}")
        self.norm_state = RMSNorm(cfg.d_model)
        if cfg.state_mixer_version == "off":
            self.state = ZeroStateMixer(cfg.d_model)
        elif cfg.state_mixer_version == "delta_matrix":
            self.state = DeltaMatrixStateMixer(
                cfg.d_model,
                cfg.n_heads,
                state_clip=cfg.state_clip,
                state_zoneout=cfg.state_zoneout,
            )
        else:
            self.state = GatedStateMixer(
                cfg.d_model,
                cfg.n_heads,
                cfg.state_mixer_version,
                state_clip=cfg.state_clip,
                state_zoneout=cfg.state_zoneout,
            )
        self.attn = None
        if cfg.attention_interval > 0 and (layer_idx + 1) % cfg.attention_interval == 0:
            self.norm_attn = RMSNorm(cfg.d_model)
            self.attn = SparseAttentionIsland(cfg.d_model, cfg.n_heads, cfg.attention_window)
            self.gamma_attn = nn.Parameter(torch.tensor(0.1))
        self.thought = None
        if cfg.thought_slots > 0 and layer_idx >= cfg.thought_start_layer:
            self.norm_thought = RMSNorm(cfg.d_model)
            self.thought = ThoughtSlotMixer(cfg.d_model, cfg.thought_slots, cfg.thought_chunk)
            self.gamma_thought = nn.Parameter(torch.tensor(0.1))
        self.landmark = None
        if cfg.landmark_interval > 0 and (layer_idx + 1) % cfg.landmark_interval == 0:
            self.norm_landmark = RMSNorm(cfg.d_model)
            self.landmark = LandmarkMemory(cfg.d_model, cfg.landmark_chunk, cfg.landmark_max)
            self.gamma_landmark = nn.Parameter(torch.tensor(0.1))
        self.norm_ff = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)
        self.gamma_local = nn.Parameter(torch.tensor(0.5))
        self.gamma_state = nn.Parameter(torch.tensor(0.5))
        self.gamma_ff = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.gamma_local * self.local(self.norm_local(x)))
        x = x + self.drop(self.gamma_state * self.state(self.norm_state(x)))
        if self.attn is not None:
            x = x + self.drop(self.gamma_attn * self.attn(self.norm_attn(x)))
        if self.thought is not None:
            x = x + self.drop(self.gamma_thought * self.thought(self.norm_thought(x)))
        if self.landmark is not None:
            x = x + self.drop(self.gamma_landmark * self.landmark(self.norm_landmark(x)))
        x = x + self.drop(self.gamma_ff * self.ff(self.norm_ff(x)))
        return x

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        return {
            "local": self.local.init_cache(batch_size, device=device, dtype=dtype),
            "state": self.state.init_cache(batch_size, device=device, dtype=dtype),
            "attn": self.attn.init_cache(batch_size, device=device, dtype=dtype) if self.attn is not None else None,
            "thought": self.thought.init_cache(batch_size, device=device, dtype=dtype) if self.thought is not None else None,
            "landmark": self.landmark.init_cache(batch_size, device=device, dtype=dtype) if self.landmark is not None else None,
        }

    def step(self, x: torch.Tensor, cache: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
        local, local_cache = self.local.step(self.norm_local(x), cache["local"])
        x = x + self.gamma_local * local
        state, state_cache = self.state.step(self.norm_state(x), cache["state"])
        x = x + self.gamma_state * state
        attn_cache = cache.get("attn")
        if self.attn is not None:
            attn, attn_cache = self.attn.step(self.norm_attn(x), attn_cache)
            x = x + self.gamma_attn * attn
        thought_cache = cache.get("thought")
        if self.thought is not None:
            thought, thought_cache = self.thought.step(self.norm_thought(x), thought_cache)
            x = x + self.gamma_thought * thought
        landmark_cache = cache.get("landmark")
        if self.landmark is not None:
            landmark, landmark_cache = self.landmark.step(self.norm_landmark(x), landmark_cache)
            x = x + self.gamma_landmark * landmark
        x = x + self.gamma_ff * self.ff(self.norm_ff(x))
        return x, {
            "local": local_cache,
            "state": state_cache,
            "attn": attn_cache,
            "thought": thought_cache,
            "landmark": landmark_cache,
        }


class SaneFlowRecurrentLM(nn.Module):
    def __init__(self, cfg: SaneFlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_embed = cfg.d_embed if cfg.d_embed and cfg.d_embed > 0 else cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, self.d_embed)
        self.embed_proj = nn.Identity() if self.d_embed == cfg.d_model else nn.Linear(self.d_embed, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([SaneFlowBlock(cfg, i) for i in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.head_proj = nn.Identity() if self.d_embed == cfg.d_model else nn.Linear(cfg.d_model, self.d_embed, bias=False)
        self.lm_head = nn.Linear(self.d_embed, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)
        self._init_gates()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_gates(self) -> None:
        for block in self.blocks:
            block.state.init_gates()

    def forward_hidden(self, input_ids: torch.Tensor, *, activation_checkpointing: bool = False) -> torch.Tensor:
        x = self.embed_proj(self.embed(input_ids))
        for block in self.blocks:
            if activation_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.head_proj(self.norm(hidden)))

    def forward(self, input_ids: torch.Tensor, *, activation_checkpointing: bool = False) -> torch.Tensor:
        return self.logits_from_hidden(self.forward_hidden(input_ids, activation_checkpointing=activation_checkpointing))

    def init_cache(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> list[dict[str, Any]]:
        param = next(self.parameters())
        device = device or param.device
        dtype = dtype or param.dtype
        return [block.init_cache(batch_size, device=device, dtype=dtype) for block in self.blocks]

    def forward_step(
        self,
        input_ids: torch.Tensor,
        cache: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if input_ids.ndim == 2:
            if input_ids.shape[1] != 1:
                logits = None
                for t in range(input_ids.shape[1]):
                    logits, cache = self.forward_step(input_ids[:, t], cache)
                assert logits is not None
                return logits, cache
            input_ids = input_ids[:, 0]
        if cache is None:
            cache = self.init_cache(input_ids.shape[0], device=input_ids.device, dtype=self.embed.weight.dtype)
        x = self.embed_proj(self.embed(input_ids))
        new_cache = []
        for block, block_cache in zip(self.blocks, cache):
            x, next_cache = block.step(x, block_cache)
            new_cache.append(next_cache)
        return self.lm_head(self.head_proj(self.norm(x))), new_cache


class DenseTransformerBlock(nn.Module):
    def __init__(self, cfg: SaneFlowConfig) -> None:
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = DenseGQAAttention(cfg)
        self.norm_ff = RMSNorm(cfg.d_model)
        self.ff = BiasFreeSwiGLU(cfg.d_model, cfg.d_ff)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm_attn(x)))
        x = x + self.drop(self.ff(self.norm_ff(x)))
        return x

    def init_cache(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return self.attn.init_cache(batch_size, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, cache: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        attn, cache = self.attn.step(self.norm_attn(x), cache)
        x = x + attn
        x = x + self.ff(self.norm_ff(x))
        return x, cache


class DenseTransformerLM(nn.Module):
    def __init__(self, cfg: SaneFlowConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_embed = cfg.d_embed if cfg.d_embed and cfg.d_embed > 0 else cfg.d_model
        self.embed = nn.Embedding(cfg.vocab_size, self.d_embed)
        self.embed_proj = nn.Identity() if self.d_embed == cfg.d_model else nn.Linear(self.d_embed, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList([DenseTransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.head_proj = nn.Identity() if self.d_embed == cfg.d_model else nn.Linear(cfg.d_model, self.d_embed, bias=False)
        self.lm_head = nn.Linear(self.d_embed, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(SaneFlowRecurrentLM._init_weights)

    def forward_hidden(self, input_ids: torch.Tensor, *, activation_checkpointing: bool = False) -> torch.Tensor:
        x = self.embed_proj(self.embed(input_ids))
        for block in self.blocks:
            if activation_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.head_proj(self.norm(hidden)))

    def forward(self, input_ids: torch.Tensor, *, activation_checkpointing: bool = False) -> torch.Tensor:
        return self.logits_from_hidden(self.forward_hidden(input_ids, activation_checkpointing=activation_checkpointing))

    def init_cache(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> list[dict[str, torch.Tensor]]:
        param = next(self.parameters())
        device = device or param.device
        dtype = dtype or param.dtype
        return [block.init_cache(batch_size, device=device, dtype=dtype) for block in self.blocks]

    def forward_step(
        self,
        input_ids: torch.Tensor,
        cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        if input_ids.ndim == 2:
            if input_ids.shape[1] != 1:
                logits = None
                for t in range(input_ids.shape[1]):
                    logits, cache = self.forward_step(input_ids[:, t], cache)
                assert logits is not None
                return logits, cache
            input_ids = input_ids[:, 0]
        if cache is None:
            cache = self.init_cache(input_ids.shape[0], device=input_ids.device, dtype=self.embed.weight.dtype)
        x = self.embed_proj(self.embed(input_ids))
        new_cache = []
        for block, block_cache in zip(self.blocks, cache):
            x, next_cache = block.step(x, block_cache)
            new_cache.append(next_cache)
        return self.logits_from_hidden(x), new_cache


class SaneFlowLM(nn.Module):
    def __new__(cls, cfg: SaneFlowConfig):
        if cls is SaneFlowLM:
            if cfg.model_type == "dense_transformer":
                return DenseTransformerLM(cfg)
            if cfg.model_type == "saneflow":
                return SaneFlowRecurrentLM(cfg)
            raise ValueError(f"unknown model_type={cfg.model_type!r}")
        return super().__new__(cls)
