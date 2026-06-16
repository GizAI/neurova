from __future__ import annotations

import sys
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn


PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_MAMBA = PACKAGE_ROOT / "vendor" / "mamba"
if LOCAL_MAMBA.exists() and str(LOCAL_MAMBA) not in sys.path:
    sys.path.insert(0, str(LOCAL_MAMBA))

from .presets import MODEL_PRESETS

MambaConfig = None
MambaLMHeadModel = None
_MAMBA_IMPORT_ERROR: BaseException | None = None
try:
    from mamba_ssm.models.config_mamba import MambaConfig as _MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as _MambaLMHeadModel
except BaseException as exc:  # pragma: no cover - depends on local CUDA/Mamba env
    _MAMBA_IMPORT_ERROR = exc
else:
    MambaConfig = _MambaConfig
    MambaLMHeadModel = _MambaLMHeadModel


@dataclass(frozen=True)
class NeuroMambaConfig:
    architecture: str = "mamba3"
    vocab_size: int = 259
    d_model: int = 256
    n_layer: int = 4
    d_intermediate: int = 512
    d_state: int = 64
    d_intermediate_schedule: tuple[int, ...] | None = None
    d_state_schedule: tuple[int, ...] | None = None
    headdim: int = 64
    is_mimo: bool = False
    mimo_rank: int = 1
    chunk_size: int = 64
    mlp_multiple_of: int = 128
    moe_num_experts: int = 1
    moe_top_k: int = 1
    n_meta_tokens: int = 0
    is_outproj_norm: bool = False
    state_edit_gates: bool = False
    layer_scale_init: float | None = None
    rope_fraction: float = 0.5
    rope_fraction_schedule: tuple[float, ...] | None = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_min_schedule: tuple[float, ...] | None = None
    dt_max_schedule: tuple[float, ...] | None = None
    A_floor: float = 1e-4
    A_floor_schedule: tuple[float, ...] | None = None
    activation_checkpointing: bool = False
    attention_interval: int = 0
    attention_num_heads: int = 8
    attention_num_heads_kv: int = 2
    attention_head_dim: int = 64
    rms_norm: bool = True
    fused_add_norm: bool = False
    residual_in_fp32: bool = True


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: NeuroMambaConfig, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.config = cfg
        self.max_position_embeddings = 4096
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(self.max_position_embeddings, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.attention_num_heads,
            dim_feedforward=cfg.d_intermediate,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layer)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.to(device=device, dtype=dtype)

    def forward(self, input_ids, inference_params=None, num_last_tokens: int = 0, **kwargs):
        del inference_params, kwargs
        batch, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        x = self.embedding(input_ids) + self.position_embedding(pos % self.max_position_embeddings)
        causal_mask = torch.ones(length, length, device=input_ids.device, dtype=torch.bool).triu(1)
        x = self.blocks(x, mask=causal_mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        if num_last_tokens:
            logits = logits[:, -num_last_tokens:]
        return SimpleNamespace(logits=logits)


def build_model(cfg: NeuroMambaConfig, device: str, dtype: torch.dtype) -> MambaLMHeadModel:
    if cfg.architecture == "transformer":
        return TinyTransformerLM(cfg, device=device, dtype=dtype)
    if MambaConfig is None or MambaLMHeadModel is None:
        raise RuntimeError(f"Mamba dependencies are unavailable: {_MAMBA_IMPORT_ERROR}")
    attn_layer_idx = []
    if cfg.attention_interval > 0:
        attn_layer_idx = [
            idx
            for idx in range(cfg.attention_interval - 1, cfg.n_layer, cfg.attention_interval)
        ]
    mamba_cfg = MambaConfig(
        d_model=cfg.d_model,
        n_layer=cfg.n_layer,
        d_intermediate=cfg.d_intermediate_schedule or cfg.d_intermediate,
        mlp_multiple_of=cfg.mlp_multiple_of,
        moe_num_experts=cfg.moe_num_experts,
        moe_top_k=cfg.moe_top_k,
        n_meta_tokens=cfg.n_meta_tokens,
        activation_checkpointing=cfg.activation_checkpointing,
        vocab_size=cfg.vocab_size,
        ssm_cfg={
            "layer": "Mamba3",
            "d_state": cfg.d_state,
            "d_state_schedule": cfg.d_state_schedule,
            "headdim": cfg.headdim,
            "is_mimo": cfg.is_mimo,
            "mimo_rank": cfg.mimo_rank,
            "chunk_size": cfg.chunk_size,
            "is_outproj_norm": cfg.is_outproj_norm,
            "state_edit_gates": cfg.state_edit_gates,
            "layer_scale_init": cfg.layer_scale_init,
            "rope_fraction": cfg.rope_fraction,
            "rope_fraction_schedule": cfg.rope_fraction_schedule,
            "dt_min": cfg.dt_min,
            "dt_max": cfg.dt_max,
            "dt_min_schedule": cfg.dt_min_schedule,
            "dt_max_schedule": cfg.dt_max_schedule,
            "A_floor": cfg.A_floor,
            "A_floor_schedule": cfg.A_floor_schedule,
        },
        attn_layer_idx=attn_layer_idx,
        attn_cfg={
            "num_heads": cfg.attention_num_heads,
            "num_heads_kv": cfg.attention_num_heads_kv,
            "head_dim": cfg.attention_head_dim,
            "causal": True,
            "rotary_emb_dim": 0,
        },
        rms_norm=cfg.rms_norm,
        fused_add_norm=cfg.fused_add_norm,
        residual_in_fp32=cfg.residual_in_fp32,
        pad_vocab_size_multiple=8,
        tie_embeddings=True,
    )
    return MambaLMHeadModel(mamba_cfg, device=device, dtype=dtype)


def preset_config(mode: str = "mimo-r4-tiny", vocab_size: int = 259) -> NeuroMambaConfig:
    if mode == "transformer-tiny":
        return NeuroMambaConfig(
            architecture="transformer",
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=2048,
            attention_num_heads=8,
            attention_num_heads_kv=8,
            attention_head_dim=64,
        )
    if mode == "mimo-r2":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=2,
            chunk_size=16,
        )
    if mode == "mimo-r2-fast-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=64,
            is_mimo=True,
            mimo_rank=2,
            chunk_size=32,
        )
    if mode == "mimo-r2-attn-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=2,
            chunk_size=16,
            mlp_multiple_of=1,
            attention_interval=4,
            attention_num_heads=8,
            attention_num_heads_kv=2,
            attention_head_dim=64,
        )
    if mode == "mamba3-recall-r2-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=2,
            chunk_size=16,
            mlp_multiple_of=1,
            n_meta_tokens=4,
            attention_interval=4,
            attention_num_heads=8,
            attention_num_heads_kv=2,
            attention_head_dim=64,
        )
    if mode == "mimo-r4-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-official-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=128,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-fast-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-attn-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            attention_interval=4,
            attention_num_heads=8,
            attention_num_heads_kv=2,
            attention_head_dim=64,
        )
    if mode == "mamba3-recall-r4-tiny":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            n_meta_tokens=4,
            attention_interval=4,
            attention_num_heads=8,
            attention_num_heads_kv=2,
            attention_head_dim=64,
        )
    if mode == "mimo-r4-16gb-120m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=640,
            n_layer=10,
            d_intermediate=1280,
            d_state=64,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-16gb-180m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=768,
            n_layer=12,
            d_intermediate=1536,
            d_state=64,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-moe-260m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=4096,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=4,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-520m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=4,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-900m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=8,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-1.1b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=10,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-1.3b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=12,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-1.7b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=16,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-2.1b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=20,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-2.3b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=22,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-2.4b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=23,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-2.5b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=24,
            moe_top_k=1,
        )
    if mode == "mimo-r4-moe-2.9b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=8192,
            d_state=32,
            headdim=32,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
            moe_num_experts=28,
            moe_top_k=1,
        )
    if mode == "mimo-r4-440m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=1792,
            d_state=128,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-paper-180m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=768,
            n_layer=12,
            d_intermediate=1264,
            d_state=128,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-880m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1536,
            n_layer=20,
            d_intermediate=2800,
            d_state=128,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mimo-r4-1.5b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=2048,
            n_layer=24,
            d_intermediate=3824,
            d_state=128,
            headdim=64,
            is_mimo=True,
            mimo_rank=4,
            chunk_size=16,
            mlp_multiple_of=1,
        )
    if mode == "mamba3-siso-hybrid-95m":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=512,
            n_layer=8,
            d_intermediate=1024,
            d_state=64,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=5,
            attention_num_heads=8,
            attention_num_heads_kv=2,
            attention_head_dim=64,
        )
    if mode == "mamba3-siso-fast-0.3b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=64,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
        )
    if mode == "mamba3-siso-fast-0.3b-ds128":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
        )
    if mode == "mamba3-siso-fast-0.3b-ds128-outnorm":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
            is_outproj_norm=True,
        )
    if mode == "mamba3-siso-fast-0.3b-ds128-outnorm-meta8":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
            n_meta_tokens=8,
            is_outproj_norm=True,
        )
    if mode == "mamba3-siso-fast-0.3b-intel-v2":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
            n_meta_tokens=8,
            is_outproj_norm=True,
            layer_scale_init=0.1,
            dt_min_schedule=(0.002,) * 4 + (0.001,) * 8 + (0.0005,) * 4,
            dt_max_schedule=(0.05,) * 4 + (0.1,) * 8 + (0.2,) * 4,
            A_floor_schedule=(2e-4,) * 4 + (1e-4,) * 8 + (5e-5,) * 4,
        )
    if mode == "mamba3-siso-fast-0.3b-stateedit-v1":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
            state_edit_gates=True,
        )
    if mode == "mamba3-siso-fast-0.3b-intel-v3":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
            n_meta_tokens=8,
            is_outproj_norm=True,
            state_edit_gates=True,
            layer_scale_init=0.1,
            dt_min_schedule=(0.002,) * 4 + (0.001,) * 8 + (0.0005,) * 4,
            dt_max_schedule=(0.05,) * 4 + (0.1,) * 8 + (0.2,) * 4,
            A_floor_schedule=(2e-4,) * 4 + (1e-4,) * 8 + (5e-5,) * 4,
        )
    if mode == "mamba3-siso-deep-0.35b-intel":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=960,
            n_layer=18,
            d_intermediate=1920,
            d_intermediate_schedule=(1536,) * 6 + (1920,) * 6 + (2304,) * 6,
            d_state=128,
            d_state_schedule=(64,) * 4 + (96,) * 6 + (128,) * 8,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=0,
            n_meta_tokens=8,
            is_outproj_norm=True,
            layer_scale_init=0.1,
            dt_min_schedule=(0.002,) * 4 + (0.001,) * 8 + (0.0005,) * 6,
            dt_max_schedule=(0.05,) * 4 + (0.1,) * 8 + (0.2,) * 6,
            A_floor_schedule=(2e-4,) * 4 + (1e-4,) * 8 + (5e-5,) * 6,
        )
    if mode == "mamba3-siso-hybrid-0.3b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1024,
            n_layer=16,
            d_intermediate=2048,
            d_state=64,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=5,
            attention_num_heads=16,
            attention_num_heads_kv=4,
            attention_head_dim=64,
        )
    if mode == "mamba3-siso-hybrid-0.7b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1280,
            n_layer=20,
            d_intermediate=2560,
            d_state=96,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=5,
            attention_num_heads=20,
            attention_num_heads_kv=5,
            attention_head_dim=64,
        )
    if mode == "mamba3-siso-hybrid-1.3b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=1792,
            n_layer=28,
            d_intermediate=3584,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=5,
            attention_num_heads=28,
            attention_num_heads_kv=7,
            attention_head_dim=64,
        )
    if mode == "mamba3-siso-hybrid-2b":
        return NeuroMambaConfig(
            vocab_size=vocab_size,
            d_model=2048,
            n_layer=32,
            d_intermediate=4096,
            d_state=128,
            headdim=64,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=64,
            mlp_multiple_of=1,
            attention_interval=5,
            attention_num_heads=32,
            attention_num_heads_kv=8,
            attention_head_dim=64,
        )
    if mode == "siso":
        return NeuroMambaConfig(vocab_size=vocab_size)
    raise ValueError(f"unknown mode: {mode}")


def tiny_config(mode: str = "siso", vocab_size: int = 259) -> NeuroMambaConfig:
    return preset_config(mode, vocab_size)


def estimate_parameters(cfg: NeuroMambaConfig) -> int:
    def schedule_sum(value: int, schedule: tuple[int, ...] | None, n_layer: int) -> int:
        if not schedule:
            return value * n_layer
        total = 0
        for idx in range(n_layer):
            total += schedule[min(idx, len(schedule) - 1)]
        return total

    if cfg.architecture == "transformer":
        embeddings = cfg.vocab_size * cfg.d_model
        attn = cfg.n_layer * (4 * cfg.d_model * cfg.d_model)
        mlp = cfg.n_layer * (2 * cfg.d_model * cfg.d_intermediate)
        norms = cfg.n_layer * cfg.d_model * 4 + cfg.d_model * 2
        pos = 4096 * cfg.d_model
        return embeddings + pos + attn + mlp + norms
    model = cfg.vocab_size * cfg.d_model
    meta = cfg.n_meta_tokens * cfg.d_model
    mlp_multiplier = max(1, cfg.moe_num_experts)
    router = cfg.n_layer * cfg.d_model * cfg.moe_num_experts if cfg.moe_num_experts > 1 else 0
    d_intermediate_total = schedule_sum(cfg.d_intermediate, cfg.d_intermediate_schedule, cfg.n_layer)
    mlp = mlp_multiplier * (3 * cfg.d_model * d_intermediate_total) + router
    ssm = cfg.n_layer * (8 * cfg.d_model * cfg.d_model)
    attn_layers = cfg.n_layer // cfg.attention_interval if cfg.attention_interval > 0 else 0
    qkv_dim = cfg.attention_head_dim * (cfg.attention_num_heads + 2 * cfg.attention_num_heads_kv)
    attn = attn_layers * (cfg.d_model * qkv_dim + cfg.d_model * cfg.attention_num_heads * cfg.attention_head_dim)
    norms = cfg.n_layer * cfg.d_model * 4
    return model + meta + mlp + ssm + attn + norms


def effective_mlp_hidden_dim(cfg: NeuroMambaConfig) -> int:
    multiple = max(1, cfg.mlp_multiple_of)
    return (cfg.d_intermediate + multiple - 1) // multiple * multiple
