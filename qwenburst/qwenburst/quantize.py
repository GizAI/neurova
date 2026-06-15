from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
try:
    from safetensors import safe_open
except Exception:  # allows CPU unit tests without safetensors installed
    safe_open = None
from tqdm import tqdm

# Qwen3.6/Qwen3.5 hybrid checkpoints use both normal attention projections
# and Gated-DeltaNet/linear-attention projections.  Keep this list explicit so
# the converted checkpoint is actually executable and not a hidden fp16 memory bomb.
DEFAULT_LINEAR_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
    # Qwen3-Next fused naming
    "in_proj_qkvz.weight",
    "in_proj_ba.weight",
    # Qwen3.5/Qwen3.6 split GDN naming observed in the user's dmc8 checkpoint
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_a.weight",
    "in_proj_b.weight",
    "out_proj.weight",
    "lm_head.weight",
    "embed_tokens.weight",
)

# Some users want a maximal-correctness first-chat build.  16GB target defaults
# to low-bit embeddings because fp16 embeddings alone cost ~2.4GB at vocab
# 248,320 x hidden 5120.
FORCE_FP16_SUFFIXES = ()


def iter_safetensors(model_dir: Path) -> Iterable[Path]:
    return sorted(model_dir.glob("*.safetensors"))


def should_quantize(name: str, *, fp16_embed: bool = False) -> bool:
    if fp16_embed and name.endswith("embed_tokens.weight"):
        return False
    return name.endswith(DEFAULT_LINEAR_SUFFIXES)


def quantize_symmetric_lowbit(w: torch.Tensor, group_size: int = 128, bits: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """Groupwise signed low-bit quantization, vectorized over row blocks.

    The original prototype used Python loops over every row and group, which is
    too slow for 27B. This implementation uses the packed layout consumed by the
    CUDA kernels while allowing different bit widths under one loader contract.
    """
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(w.shape)}")
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    w = w.detach().float().cpu().contiguous()
    rows, cols = w.shape
    n_groups = math.ceil(cols / group_size)
    packed_cols = math.ceil(cols * bits / 8)
    padded_cols = n_groups * group_size

    packed = np.zeros((rows, packed_cols), dtype=np.uint8)
    scales = np.zeros((rows, n_groups), dtype=np.float16)
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    zero = 1 << (bits - 1)

    block_rows = int(os.environ.get("QWENBURST_QUANT_BLOCK_ROWS", "256"))
    block_rows = max(1, block_rows)

    for start_row in range(0, rows, block_rows):
        end_row = min(rows, start_row + block_rows)
        block = w[start_row:end_row]
        if padded_cols != cols:
            padded = torch.zeros((end_row - start_row, padded_cols), dtype=block.dtype)
            padded[:, :cols] = block
            block = padded
        grouped = block.view(end_row - start_row, n_groups, group_size)
        scale = grouped.abs().amax(dim=2) / float(qmax)
        scale = torch.where(torch.isfinite(scale) & (scale > 0), scale, torch.ones_like(scale))
        q = torch.clamp(torch.round(grouped / scale.unsqueeze(-1)), qmin, qmax).to(torch.int16)
        q = q.view(end_row - start_row, padded_cols)[:, :cols]
        qplus = (q + zero).to(torch.int16).numpy()
        out = packed[start_row:end_row]
        for c in range(cols):
            bit_pos = c * bits
            byte_i = bit_pos // 8
            shift = bit_pos % 8
            vals = qplus[:, c].astype(np.uint16)
            out[:, byte_i] |= ((vals << shift) & 0xFF).astype(np.uint8)
            spill = shift + bits - 8
            if spill > 0 and byte_i + 1 < packed_cols:
                out[:, byte_i + 1] |= (vals >> (bits - spill)).astype(np.uint8)
        scales[start_row:end_row] = scale.to(torch.float16).numpy()

    meta = {"rows": rows, "cols": cols, "group_size": group_size, "n_groups": n_groups, "packed_cols": packed_cols, "bits": bits}
    return packed, scales, meta


def write_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))


def convert(
    model_dir: Path,
    out_dir: Path,
    group_size: int = 128,
    max_tensors: int | None = None,
    *,
    fp16_embed: bool = False,
    bits: int = 4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {
        "format": f"qwenburst-q{bits}-v4",
        "group_size": group_size,
        "bits": bits,
        "fp16_embed": fp16_embed,
        "tensors": {},
    }
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        index["hf_config_json"] = json.loads(cfg_path.read_text(encoding="utf-8"))
    count = 0

    if safe_open is None:
        raise RuntimeError("safetensors is required for checkpoint conversion: pip install safetensors")

    for st_path in iter_safetensors(model_dir):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for name in tqdm(f.keys(), desc=st_path.name):
                if max_tensors is not None and count >= max_tensors:
                    break
                tensor = f.get_tensor(name)
                rel = name.replace(".", "__")
                if should_quantize(name, fp16_embed=fp16_embed) and tensor.ndim == 2:
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
                    count += 1
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
        if max_tensors is not None and count >= max_tensors:
            break

    with open(out_dir / "qwenburst_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert HF safetensors to QwenBurst low-bit format")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4, help="weight bits for quantized 2D tensors")
    ap.add_argument("--max-tensors", type=int, default=None, help="debug: convert only N quantized tensors")
    ap.add_argument("--fp16-embed", action="store_true", help="keep embed_tokens.weight fp16; not recommended for 16GB")
    args = ap.parse_args()
    convert(args.model_dir, args.out_dir, group_size=args.group_size, max_tensors=args.max_tensors, fp16_embed=args.fp16_embed, bits=args.bits)


if __name__ == "__main__":
    main()
