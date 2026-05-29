"""In-Place TTT v3 — 4-bit 양자화 호환 + 2-pass TTT.

핵심 설계 (4-bit 양자화 모델 호환):
  - Base 4-bit MLP: gate(act(x)) * up(x) → down (frozen)
  - TTT residual: 별도 bfloat16 weight, down_proj 출력에 더함
  - 2-pass TTT:
    Pass 1: hidden state target 수집 (hook)
    Pass 2+: 수집된 target → TTTMLP가 self-prediction으로 residual 업데이트

  Self-prediction:
    MLP(h[t]) → MLP hidden h[t]와 target target[t] (같은 position)의
    outer product를 chunk 단위로 계산 → down_proj delta → residual 누적

최적화 (RTX 4080 16GB 단일 GPU):
  - 멀티패스 TTT (3~10회)
  - TTT 모멘텀 (0.9)
  - SDPA (PyTorch native FlashAttention)
  - 가중치 변화 검증
"""
from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from opt_einsum import contract

__all__ = ["InPlaceTTTWrapper", "TTTMLP", "apply_ttt_to_model"]
_HAS_FLASH = True


class TTTMLP(nn.Module):
    """MLP with residual TTT adapter — 4-bit quantized compatible.

    TTT: Self-prediction on same-position hidden states.
    delta = sum_c MLP_hidden[c] ⊗ target_hidden[c], projected by ttt_proj.
    delta is averaged, momentum-applied, accumulated into _ttt_residual.
    """

    def __init__(self, hidden_size: int, intermediate_size: int,
                 layer_idx: int = -1, ttt_config: Optional[Dict] = None,
                 gate_proj: Optional[nn.Module] = None,
                 up_proj: Optional[nn.Module] = None,
                 down_proj: Optional[nn.Module] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_idx = layer_idx

        # Base 4-bit MLP (shared, frozen)
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

        cfg = ttt_config or {}
        self.ttt_lr = cfg.get("ttt_lr", 0.01)
        self.ttt_chunk = cfg.get("ttt_chunk", 2048)
        self.ttt_momentum = cfg.get("ttt_momentum", 0.9)

        if cfg.get("ttt_proj", True):
            self.ttt_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.ttt_proj = None

        self.ttt_conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size=cfg.get("ttt_conv_size", 5),
            padding=cfg.get("ttt_conv_size", 5) // 2,
            groups=hidden_size, bias=False,
        )

        # TTT residual: bfloat16, (intermediate, hidden) — same shape as down_proj.T
        self.register_buffer("_ttt_residual",
            torch.zeros(intermediate_size, hidden_size, dtype=torch.bfloat16))
        self.register_buffer("_delta_buffer",
            torch.zeros(intermediate_size, hidden_size, dtype=torch.bfloat16))

        # TTT state
        self._ttt_target = None       # collected by hook (Pass 1)
        self._ttt_collect = False     # hook enabler
        self._ttt_learn_target = None  # set by forward_ttt (Pass 2)
        self._weight_snapshot = None

    def _init_weights(self, std=0.02):
        if self.ttt_proj is not None:
            d = self.ttt_proj.weight.shape[0]
            self.ttt_proj.weight.data.zero_()
            vals = torch.randn(d, device=self.ttt_proj.weight.device,
                               dtype=self.ttt_proj.weight.dtype) * std
            self.ttt_proj.weight.data[range(d), range(d)] = vals
        self.ttt_conv.weight.data.zero_()
        # Init as averaging filter (In-Place TTT paper)
        # Depthwise conv weight shape: [out_ch, 1, kernel_size]
        k = self.ttt_conv.kernel_size[0] if hasattr(self.ttt_conv, 'kernel_size') else 5
        if isinstance(k, tuple): k = k[0]
        center = k // 2
        # Set center to uniform value
        self.ttt_conv.weight.data[:, 0, :] = 1.0 / k
        # Add tiny noise to break symmetry for depthwise
        self.ttt_conv.weight.data += torch.randn_like(self.ttt_conv.weight.data) * 0.001
        self._ttt_residual.zero_()
        self._delta_buffer.zero_()

    def snapshot_weights(self):
        self._weight_snapshot = self._ttt_residual.data.clone().cpu()

    def weight_diff(self) -> float:
        if self._weight_snapshot is None:
            return 0.0
        return (self._ttt_residual.float().cpu() - self._weight_snapshot.float()
                ).abs().mean().item()

    def _pad_chunk(self, x):
        b, s, d = x.shape
        if s <= 1:
            pad = self.ttt_chunk - s
            x = torch.cat([x, torch.zeros(b, pad, d, device=x.device, dtype=x.dtype)], dim=1)
        elif s % self.ttt_chunk != 0:
            pad = self.ttt_chunk - s % self.ttt_chunk
            x = torch.cat([x, torch.zeros(b, pad, d, device=x.device, dtype=x.dtype)], dim=1)
        return rearrange(x, "b (t c) d -> b t c d", c=self.ttt_chunk)

    def forward(self, x: torch.Tensor, target_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward: base MLP (4-bit) + TTT residual (bf16).

        Generation (target_states=None, _ttt_learn_target=None):
          base(h) + h @ residual

        Learning (_ttt_learn_target set):
          1. Self-prediction: h[t] → target[t] (same position)
          2. Compute delta via conv + contract
          3. Momentum + residual update
        """
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)  # [1, S, intermediate]

        # Determine learning target
        actual_target = target_states if target_states is not None else self._ttt_learn_target

        if actual_target is None:
            # ── Generation ──
            # Scale residual to avoid overwhelming base output
            return self.down_proj(h) + h @ (self._ttt_residual * 0.3)

        # ── TTT Learning ──
        with torch.no_grad():
            seq_len = min(h.shape[1], actual_target.shape[1])
            if seq_len < 3:
                return self.down_proj(h) + h @ self._ttt_residual

            if seq_len <= self.ttt_chunk + 1:
                # ── Flat (short sequence) ──
                # Treat entire sequence as one chunk: [1, 1, S, d]
                hp_4d = h.unsqueeze(1)           # [1, 1, S, intermediate]
                t_4d = actual_target.unsqueeze(1) # [1, 1, S, hidden]

                # Conv on target
                t_conv = self.ttt_conv(
                    t_4d.transpose(-1, -2).reshape(1, self.hidden_size, -1)
                ).reshape(1, 1, -1, self.hidden_size)

                if self.ttt_proj is not None:
                    dd = contract("b t c h, b t c d, d e -> b t e h",
                                  hp_4d, t_conv, self.ttt_proj.weight)
                else:
                    dd = contract("b t c h, b t c d -> b t d h",
                                  hp_4d, t_conv)
                dd_mean = dd.mean(dim=(0, 1)).T  # [hidden, intermediate] → [intermediate, hidden]
            else:
                # ── Chunked (long sequence) ──
                t_chunked = self._pad_chunk(actual_target)
                hp_chunked = self._pad_chunk(h)
                bs, cn, cs, _ = t_chunked.shape

                # Conv on target
                t_chunked = self.ttt_conv(
                    t_chunked.transpose(-1, -2).reshape(bs * cn, -1, cs)
                ).transpose(-1, -2).reshape(bs, cn, cs, -1)

                n = min(hp_chunked.shape[1], cn)
                if n <= 1:
                    return self.down_proj(h) + h @ self._ttt_residual

                if self.ttt_proj is not None:
                    dd = contract("b t c h, b t c d, d e -> b t e h",
                                  hp_chunked[:, :n-1],
                                  t_chunked[:, :n-1],
                                  self.ttt_proj.weight)
                else:
                    dd = contract("b t c h, b t c d -> b t d h",
                                  hp_chunked[:, :n-1],
                                  t_chunked[:, :n-1])
                dd_mean = dd.mean(dim=(0, 1, 2)).T  # [hidden, intermediate] → [intermediate, hidden]

            # Momentum
            self._delta_buffer = (
                self.ttt_momentum * self._delta_buffer +
                (1 - self.ttt_momentum) * dd_mean
            )

            # Update residual
            self._ttt_residual += self._delta_buffer * self.ttt_lr

            return self.down_proj(h) + h @ self._ttt_residual


class InPlaceTTTWrapper(nn.Module):
    """In-Place TTT wrapper v3 — 2-pass TTT, 4-bit compatible."""

    def __init__(self, base_model: nn.Module,
                 ttt_layers: Optional[List[int]] = None,
                 ttt_lr: float = 2.0, ttt_chunk: int = 2048,
                 ttt_momentum: float = 0.9,
                 ttt_proj: bool = True, ttt_conv_size: int = 5):
        super().__init__()
        self.base_model = base_model
        self.config = getattr(base_model, "config", getattr(base_model, "model", None))
        if hasattr(self.config, "text_config"):
            self.hidden_size = self.config.text_config.hidden_size
        else:
            self.hidden_size = getattr(self.config, "hidden_size", 2560)

        self.ttt_config = {
            "ttt_lr": ttt_lr, "ttt_chunk": ttt_chunk,
            "ttt_momentum": ttt_momentum,
            "ttt_proj": ttt_proj, "ttt_conv_size": ttt_conv_size,
        }
        self.ttt_layers = ttt_layers or list(range(0, 32, 6))

        self._replace_mlps()
        self._hooks = []
        self._register_ttt_hooks()
        print(f"[TTT] SDPA (PyTorch native) | {len(self.ttt_layers)} layers {self.ttt_layers}", flush=True)
        self._ttt_pass_count = 0

    def _find_mlp(self, idx):
        try:
            layer = self.base_model.model.layers[idx]
            if hasattr(layer, "mlp"):
                return layer, "mlp"
        except:
            pass
        return None, None

    def _replace_mlps(self):
        hs = self.hidden_size
        for idx in self.ttt_layers:
            layer, attr = self._find_mlp(idx)
            if layer is None:
                continue
            old = getattr(layer, attr)
            if hasattr(old, "down_proj"):
                inter = old.down_proj.in_features
            else:
                inter = getattr(old, "intermediate_size", None)
            if inter is None:
                continue

            ttt = TTTMLP(hidden_size=hs, intermediate_size=inter, layer_idx=idx,
                         ttt_config=self.ttt_config,
                         gate_proj=getattr(old, "gate_proj", None),
                         up_proj=getattr(old, "up_proj", None),
                         down_proj=getattr(old, "down_proj", None))
            ttt._init_weights()
            dev = next(old.parameters()).device
            ttt.to(dev).to(torch.bfloat16)
            setattr(layer, attr, ttt)

    def _register_ttt_hooks(self):
        """Register hooks on decoder layers to capture input (target for TTT)."""
        for idx in self.ttt_layers:
            layer, _ = self._find_mlp(idx)
            if layer is None:
                continue
            mlp = layer.mlp
            if not hasattr(mlp, "ttt_conv"):
                continue

            def make_hook(mref):
                def hook(mod, inp, out):
                    if hasattr(mref, "_ttt_collect") and mref._ttt_collect:
                        h = inp[0] if isinstance(inp, tuple) else inp
                        mref._ttt_target = h.detach()
                return hook

            self._hooks.append(layer.register_forward_hook(make_hook(mlp)))

    def snapshot_weights(self):
        for idx in self.ttt_layers:
            layer, _ = self._find_mlp(idx)
            if layer and hasattr(layer.mlp, "snapshot_weights"):
                layer.mlp.snapshot_weights()

    def weight_diff(self) -> Dict[int, float]:
        diffs = {}
        for idx in self.ttt_layers:
            layer, _ = self._find_mlp(idx)
            if layer and hasattr(layer.mlp, "weight_diff"):
                d = layer.mlp.weight_diff()
                if d > 0:
                    diffs[idx] = d
        return diffs

    def forward_ttt(self, input_ids=None, attention_mask=None,
                     num_passes: int = 1, **kwargs) -> torch.Tensor:
        """2-pass TTT: collect targets, then learn.

        Pass 1 (collect): Forward through model while hooks collect targets.
        Pass 2..N (learn): Forward through model again with targets set.
        """
        # ── Pass 1: Collect TTT targets ──
        for idx in self.ttt_layers:
            layer, _ = self._find_mlp(idx)
            if layer and hasattr(layer.mlp, "_ttt_collect"):
                layer.mlp._ttt_collect = True
                layer.mlp._ttt_target = None

        with torch.no_grad():
            self.base_model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=False, use_cache=False, **kwargs,
            )

        # Disable hooks
        for idx in self.ttt_layers:
            layer, _ = self._find_mlp(idx)
            if layer and hasattr(layer.mlp, "_ttt_collect"):
                layer.mlp._ttt_collect = False

        # ── Pass 2..N: TTT learning ──
        for pass_idx in range(num_passes):
            ramp_lr = min(1.0 + pass_idx * 0.5, 5.0) if pass_idx > 0 else None

            for idx in self.ttt_layers:
                layer, _ = self._find_mlp(idx)
                if layer is None:
                    continue
                if ramp_lr is not None and hasattr(layer.mlp, "ttt_lr"):
                    layer.mlp.ttt_lr = ramp_lr
                # Set learned target from collected hooks
                target = getattr(layer.mlp, "_ttt_target", None)
                if target is not None:
                    layer.mlp._ttt_learn_target = target.contiguous()

            with torch.no_grad():
                self.base_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=False, use_cache=False, **kwargs,
                )

            # Clear learn targets
            for idx in self.ttt_layers:
                layer, _ = self._find_mlp(idx)
                if layer:
                    layer.mlp._ttt_learn_target = None

        self._ttt_pass_count += num_passes
        
        return None

    def reset_momentum(self):
        for idx in self.ttt_layers:
            layer, _ = self._find_mlp(idx)
            if layer and hasattr(layer.mlp, "_delta_buffer"):
                layer.mlp._delta_buffer.zero_()
                layer.mlp._ttt_residual.zero_()
        self._ttt_pass_count = 0

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        return self.base_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def apply_ttt_to_model(model: nn.Module, **kwargs) -> InPlaceTTTWrapper:
    return InPlaceTTTWrapper(model, **kwargs)
