from __future__ import annotations

from .model import NeuroMambaConfig, effective_mlp_hidden_dim


def check_architecture_contract(cfg: NeuroMambaConfig, tokenizer: str) -> list[str]:
    errors: list[str] = []
    if tokenizer != "llama31":
        errors.append("tokenizer must be llama31 for the English-first Mamba-3 target path")
    is_siso_hybrid = (
        not cfg.is_mimo
        and cfg.headdim == 64
        and cfg.chunk_size == 64
        and cfg.attention_interval > 0
    )
    is_siso_fast = (
        not cfg.is_mimo
        and cfg.headdim == 64
        and cfg.chunk_size == 64
        and cfg.attention_interval == 0
    )
    is_siso_target = is_siso_fast or is_siso_hybrid
    if not is_siso_target:
        if not cfg.is_mimo:
            errors.append("target path must be SISO fast, SISO hybrid, or Mamba-3 MIMO")
        if cfg.is_mimo and cfg.mimo_rank != 4:
            errors.append("MIMO target path must preserve MIMO rank 4")
    if cfg.is_mimo and cfg.d_state != 128 and cfg.d_model >= 768:
        errors.append("paper-scale Mamba-3 MIMO target must keep d_state=128")
    if cfg.headdim != 64 and cfg.d_model >= 1024:
        errors.append("large target should keep headdim=64")
    if is_siso_hybrid and not (4 <= cfg.attention_interval <= 6):
        errors.append("SISO hybrid should insert attention every 4-6 blocks")
    if cfg.d_intermediate <= 0:
        errors.append("SwiGLU MLP interleave must stay enabled")
    paper_mlp_dims = {
        768: 1264,
        1024: 1792,
        1536: 2800,
        2048: 3824,
    }
    expected_mlp = paper_mlp_dims.get(cfg.d_model)
    if not is_siso_target and expected_mlp is not None and cfg.d_intermediate != expected_mlp:
        errors.append(f"MIMO-R4 parameter-matched MLP dim must be {expected_mlp} for d_model={cfg.d_model}")
    if not is_siso_target and expected_mlp is not None and effective_mlp_hidden_dim(cfg) != expected_mlp:
        errors.append(
            f"effective GatedMLP hidden dim must remain {expected_mlp}; "
            f"set mlp_multiple_of=1 for paper-matched MIMO-R4"
        )
    return errors
