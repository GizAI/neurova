from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from .loader import QuantizedStore, LowBitTensor, FP16Tensor
from .model import WeightResolver, embed_lookup, linear_any
from .quantize import quantize_symmetric_lowbit, write_array
from .state import DecodeState

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover - optional until conversion/inspection
    safe_open = None


@dataclass(frozen=True)
class DFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    layer_types: tuple[str, ...]
    sliding_window: int | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DFlashConfig":
        dflash = data.get("dflash_config") if isinstance(data.get("dflash_config"), dict) else {}
        layer_types = tuple(data.get("layer_types") or ("full_attention",) * int(data["num_hidden_layers"]))
        target_layer_ids = tuple(int(x) for x in dflash.get("target_layer_ids", ()))
        if not target_layer_ids:
            target_layer_ids = _default_target_layer_ids(int(data["num_target_layers"]), int(data["num_hidden_layers"]))
        return cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=int(data.get("head_dim", int(data["hidden_size"]) // int(data["num_attention_heads"]))),
            vocab_size=int(data["vocab_size"]),
            block_size=int(data["block_size"]),
            target_layer_ids=target_layer_ids,
            num_target_layers=int(data["num_target_layers"]),
            mask_token_id=int(dflash.get("mask_token_id", 0)),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            layer_types=layer_types,
            sliding_window=int(data["sliding_window"]) if data.get("sliding_window") is not None else None,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "DFlashConfig":
        path = Path(path)
        cfg_path = path / "config.json" if path.is_dir() else path
        return cls.from_json(json.loads(cfg_path.read_text(encoding="utf-8")))


def _default_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> tuple[int, ...]:
    if num_draft_layers == 1:
        return (num_target_layers // 2,)
    start = 1
    end = num_target_layers - 3
    span = end - start
    return tuple(int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers))


def iter_safetensors(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
    else:
        yield from sorted(root.glob("*.safetensors"))


def inspect_dflash(root: str | Path) -> dict[str, Any]:
    if safe_open is None:
        raise RuntimeError("safetensors is required: pip install safetensors")
    root = Path(root)
    cfg = DFlashConfig.from_path(root)
    tensors: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for st_path in iter_safetensors(root):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                nbytes = tensor.numel() * tensor.element_size()
                tensors[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "bytes": nbytes,
                }
                total_bytes += nbytes
    return {
        "format": "dflash-draft",
        "config": {
            "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers,
            "block_size": cfg.block_size,
            "target_layer_ids": list(cfg.target_layer_ids),
            "layer_types": list(cfg.layer_types),
        },
        "tensor_count": len(tensors),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 1024**3,
        "tensors": tensors,
    }


def should_quantize_dflash_tensor(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim != 2:
        return False
    # DFlash is a drafter component: all 2D trainable matrices are projection
    # weights. Norm vectors and scalar metadata stay fp16 for exactness.
    return name.endswith(".weight")


def convert_dflash_lowbit(
    in_dir: str | Path,
    out_dir: str | Path,
    *,
    bits: int = 3,
    group_size: int = 128,
) -> None:
    if safe_open is None:
        raise RuntimeError("safetensors is required: pip install safetensors")
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_json = json.loads((in_dir / "config.json").read_text(encoding="utf-8"))
    cfg = DFlashConfig.from_json(cfg_json)
    index: dict[str, Any] = {
        "format": f"qwenburst-dflash-q{bits}-v1",
        "bits": bits,
        "group_size": group_size,
        "dflash_config_json": cfg_json,
        "target_layer_ids": list(cfg.target_layer_ids),
        "block_size": cfg.block_size,
        "tensors": {},
    }
    for st_path in iter_safetensors(in_dir):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                rel = name.replace(".", "__")
                if should_quantize_dflash_tensor(name, tensor):
                    packed, scales, meta = quantize_symmetric_lowbit(tensor, group_size=group_size, bits=bits)
                    q_path = Path(f"q{bits}") / f"{rel}.q{bits}.bin"
                    s_path = Path(f"q{bits}") / f"{rel}.scale.fp16.bin"
                    write_array(out_dir / q_path, packed)
                    write_array(out_dir / s_path, scales)
                    index["tensors"][name] = {
                        "kind": "lowbit_symmetric_groupwise",
                        "qweight": str(q_path),
                        "scales": str(s_path),
                        **meta,
                    }
                else:
                    arr = tensor.detach().cpu().to(torch.float16).contiguous().numpy()
                    raw_path = Path("fp16") / f"{rel}.fp16.bin"
                    write_array(out_dir / raw_path, arr)
                    index["tensors"][name] = {
                        "kind": "fp16_raw",
                        "path": str(raw_path),
                        "shape": list(tensor.shape),
                        "dtype": "float16",
                    }
    (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "qwenburst_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


class DFlashDraftAdapter:
    """QwenBurst-side boundary for DFlash speculative decoding.

    This class owns the draft checkpoint and exposes the target-layer taps
    qwenburst must collect. The actual block-diffusion proposal path should live
    behind `propose()`; server code must never replace qwenburst target
    verification with a third-party runtime.
    """

    def __init__(self, store: QuantizedStore, cfg: DFlashConfig, device: str | torch.device = "cuda"):
        self.store = store
        self.cfg = cfg
        self.device = torch.device(device)
        self.weights = WeightResolver(store)
        self.fc = self.weights.any_linear("fc.weight")
        self.hidden_norm = self.weights.fp16("hidden_norm.weight")
        self.norm = self.weights.fp16("norm.weight")
        self.layers = [DFlashDraftLayer(self.cfg, self.weights, i, self.device) for i in range(self.cfg.num_hidden_layers)]

    @classmethod
    def from_lowbit_dir(cls, path: str | Path, device: str | torch.device = "cuda") -> "DFlashDraftAdapter":
        path = Path(path)
        if not (path / "qwenburst_index.json").exists():
            raise FileNotFoundError(f"missing converted DFlash qwenburst_index.json: {path}")
        store = QuantizedStore(path, device=device)
        raw_cfg = store.index.get("dflash_config_json")
        if not isinstance(raw_cfg, dict):
            raise ValueError("converted DFlash index is missing dflash_config_json")
        return cls(store=store, cfg=DFlashConfig.from_json(raw_cfg), device=device)

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        return self.cfg.target_layer_ids

    def forward_block(
        self,
        block_ids: list[int],
        target_hidden_history: list[torch.Tensor],
        target_model: Any,
        *,
        logits_start: int = 0,
    ) -> torch.Tensor:
        if not target_hidden_history:
            raise ValueError("DFlash forward requires at least one target hidden tap")
        target_hidden = torch.stack(target_hidden_history, dim=0).to(self.device, dtype=torch.float16)
        ctx = _rmsnorm(_linear_seq(self.fc, target_hidden, self.device), self.hidden_norm, self.cfg.rms_norm_eps)
        h = torch.stack([embed_lookup(target_model.embed, torch.tensor(tid, device=self.device)) for tid in block_ids], dim=0).to(self.device)
        for layer in self.layers:
            h = layer(h, ctx)
        h = _rmsnorm(h, self.norm, self.cfg.rms_norm_eps)
        if logits_start:
            h = h[logits_start:]
        logits = _linear_seq(target_model.lm_head, h, self.device)
        return logits

    @torch.no_grad()
    def generate(
        self,
        target_model: Any,
        state: DecodeState,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_ids: tuple[int, ...] = (),
        block_size: int | None = None,
    ):
        if not prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        target_hidden_history: list[torch.Tensor] = []
        logits = None
        for tid in prompt_ids:
            logits, taps = target_model.forward_one(
                tid,
                state,
                return_logits=True,
                hidden_tap_layers=self.target_layer_ids,
            )
            target_hidden_history.append(torch.cat([t.detach() for t in taps], dim=0))
        assert logits is not None
        next_id = int(torch.argmax(logits, dim=-1).item())
        produced = 0
        accepted_lengths: list[int] = []
        runtime_block_size = int(block_size or self.cfg.block_size)
        if runtime_block_size < 2 or runtime_block_size > self.cfg.block_size:
            raise ValueError(f"block_size must be in [2, {self.cfg.block_size}], got {runtime_block_size}")
        while produced < max_new_tokens:
            if eos_token_ids and next_id in eos_token_ids:
                break
            block = [next_id] + [self.cfg.mask_token_id] * (runtime_block_size - 1)
            draft_logits = self.forward_block(block, target_hidden_history, target_model, logits_start=1 - runtime_block_size)
            draft_ids = torch.argmax(draft_logits, dim=-1).detach().tolist()
            for i, tid in enumerate(draft_ids, start=1):
                block[i] = int(tid)

            verified = target_model.forward_block(block, state, hidden_tap_layers=self.target_layer_ids)
            accepted = 0
            next_id_after_commit: int | None = None
            for i, (logits_i, taps_i) in enumerate(zip(verified.logits, verified.hidden_taps)):
                pred = int(torch.argmax(logits_i, dim=-1).item())
                if i == len(block) - 1:
                    next_id_after_commit = pred
                    break
                if int(block[i + 1]) != pred:
                    next_id_after_commit = pred
                    break
                accepted += 1

            commit_n = min(accepted + 1, max_new_tokens - produced)
            committed = target_model.forward_block(block[:commit_n], state, hidden_tap_layers=self.target_layer_ids, commit=True)
            for i in range(commit_n):
                target_hidden_history.append(torch.cat([t.detach() for t in committed.hidden_taps[i]], dim=0))
                produced += 1
                yield int(block[i]), accepted + 1
                if eos_token_ids and int(block[i]) in eos_token_ids:
                    return
            accepted_lengths.append(accepted + 1)
            if next_id_after_commit is None:
                raise RuntimeError("DFlash verification did not produce a next token")
            next_id = next_id_after_commit


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (y * weight.to(device=x.device, dtype=torch.float32)).to(x.dtype)


def _linear_seq(w: LowBitTensor | FP16Tensor, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.ndim == 1:
        return linear_any(w, x)
    return torch.stack([linear_any(w, row.to(device)) for row in x], dim=0)


def _rope_seq(x: torch.Tensor, positions: torch.Tensor, rope_theta: float) -> torch.Tensor:
    # x: [T, H, D]
    d = x.shape[-1]
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).to(x.dtype)[:, None, :]
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).to(x.dtype)[:, None, :]
    half = d // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rot * sin


class DFlashMLP:
    def __init__(self, weights: WeightResolver, layer: int):
        p = f"layers.{layer}.mlp"
        self.gate = weights.any_linear(f"{p}.gate_proj.weight")
        self.up = weights.any_linear(f"{p}.up_proj.weight")
        self.down = weights.any_linear(f"{p}.down_proj.weight")

    def __call__(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        gate = _linear_seq(self.gate, x, device)
        up = _linear_seq(self.up, x, device)
        return _linear_seq(self.down, F.silu(gate) * up, device)


class DFlashAttention:
    def __init__(self, cfg: DFlashConfig, weights: WeightResolver, layer: int):
        self.cfg = cfg
        self.layer = layer
        p = f"layers.{layer}.self_attn"
        self.q_proj = weights.any_linear(f"{p}.q_proj.weight")
        self.k_proj = weights.any_linear(f"{p}.k_proj.weight")
        self.v_proj = weights.any_linear(f"{p}.v_proj.weight")
        self.o_proj = weights.any_linear(f"{p}.o_proj.weight")
        self.q_norm = weights.fp16(f"{p}.q_norm.weight")
        self.k_norm = weights.fp16(f"{p}.k_norm.weight")

    def __call__(self, x: torch.Tensor, ctx: torch.Tensor, device: torch.device) -> torch.Tensor:
        cfg = self.cfg
        q = _linear_seq(self.q_proj, x, device).view(x.shape[0], cfg.num_attention_heads, cfg.head_dim)
        k_ctx = _linear_seq(self.k_proj, ctx, device).view(ctx.shape[0], cfg.num_key_value_heads, cfg.head_dim)
        v_ctx = _linear_seq(self.v_proj, ctx, device).view(ctx.shape[0], cfg.num_key_value_heads, cfg.head_dim)
        k_prop = _linear_seq(self.k_proj, x, device).view(x.shape[0], cfg.num_key_value_heads, cfg.head_dim)
        v_prop = _linear_seq(self.v_proj, x, device).view(x.shape[0], cfg.num_key_value_heads, cfg.head_dim)
        q = _rmsnorm(q, self.q_norm, cfg.rms_norm_eps)
        k = _rmsnorm(torch.cat([k_ctx, k_prop], dim=0), self.k_norm, cfg.rms_norm_eps)
        v = torch.cat([v_ctx, v_prop], dim=0)

        ctx_len = ctx.shape[0]
        if cfg.layer_types[self.layer] == "sliding_attention" and cfg.sliding_window is not None:
            keep = max(1, cfg.sliding_window - x.shape[0])
            if ctx_len > keep:
                k = k[ctx_len - keep :]
                v = v[ctx_len - keep :]
                ctx_len = keep

        q_pos = torch.arange(ctx_len, ctx_len + x.shape[0], device=device)
        k_pos = torch.arange(0, k.shape[0], device=device)
        q = _rope_seq(q, q_pos, cfg.rope_theta)
        k = _rope_seq(k, k_pos, cfg.rope_theta)

        repeat = cfg.num_attention_heads // cfg.num_key_value_heads
        k = k.repeat_interleave(repeat, dim=1).transpose(0, 1)
        v = v.repeat_interleave(repeat, dim=1).transpose(0, 1)
        q = q.transpose(0, 1)
        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (cfg.head_dim ** -0.5)
        probs = torch.softmax(scores, dim=-1).to(x.dtype)
        out = torch.matmul(probs, v).transpose(0, 1).reshape(x.shape[0], cfg.num_attention_heads * cfg.head_dim)
        return _linear_seq(self.o_proj, out, device)


class DFlashDraftLayer:
    def __init__(self, cfg: DFlashConfig, weights: WeightResolver, layer: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        p = f"layers.{layer}"
        self.input_norm = weights.fp16(f"{p}.input_layernorm.weight")
        self.post_norm = weights.fp16(f"{p}.post_attention_layernorm.weight")
        self.attn = DFlashAttention(cfg, weights, layer)
        self.mlp = DFlashMLP(weights, layer)

    def __call__(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(_rmsnorm(x, self.input_norm, self.cfg.rms_norm_eps), ctx, self.device)
        return x + self.mlp(_rmsnorm(x, self.post_norm, self.cfg.rms_norm_eps), self.device)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect or convert a DFlash draft checkpoint for qwenburst")
    sub = ap.add_subparsers(dest="cmd", required=True)
    inspect_ap = sub.add_parser("inspect")
    inspect_ap.add_argument("path", type=Path)
    convert_ap = sub.add_parser("convert")
    convert_ap.add_argument("in_dir", type=Path)
    convert_ap.add_argument("out_dir", type=Path)
    convert_ap.add_argument("--bits", type=int, default=3)
    convert_ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()
    if args.cmd == "inspect":
        print(json.dumps(inspect_dflash(args.path), indent=2, ensure_ascii=False))
    elif args.cmd == "convert":
        convert_dflash_lowbit(args.in_dir, args.out_dir, bits=args.bits, group_size=args.group_size)
    else:  # pragma: no cover
        raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
