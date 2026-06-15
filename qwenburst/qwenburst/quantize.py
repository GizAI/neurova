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
    "qkv_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "gate_up_proj.weight",
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


def _marlin_perms() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])
    perm_np = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm_np = perm_np.reshape((-1, 8))[:, interleave].ravel()
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return (
        torch.from_numpy(perm_np).long(),
        torch.tensor(scale_perm, dtype=torch.long),
        torch.tensor(scale_perm_single, dtype=torch.long),
    )


_MARLIN_PERM, _MARLIN_SCALE_PERM, _MARLIN_SCALE_PERM_SINGLE = _marlin_perms()


def iter_safetensors(model_dir: Path) -> Iterable[Path]:
    return sorted(model_dir.glob("*.safetensors"))


def should_quantize(name: str, *, fp16_embed: bool = False) -> bool:
    if fp16_embed and name.endswith("embed_tokens.weight"):
        return False
    return name.endswith(DEFAULT_LINEAR_SUFFIXES)


def should_use_marlin(name: str, tensor: torch.Tensor, *, layout: str, hybrid_policy: str, group_size: int) -> bool:
    if name.endswith("embed_tokens.weight"):
        return False
    if tensor.ndim != 2:
        return False
    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    if rows % 256 != 0 or cols % 128 != 0:
        return False
    if group_size != 128:
        return False
    if layout == "marlin":
        return True
    if hybrid_policy == "q3q4_hot":
        hot_suffixes = (
            "q_proj.weight",
            "qkv_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "o_proj.weight",
            "in_proj_qkvz.weight",
            "in_proj_ba.weight",
            "in_proj_qkv.weight",
            "in_proj_z.weight",
            "in_proj_a.weight",
            "in_proj_b.weight",
            "out_proj.weight",
            "lm_head.weight",
        )
        return name.endswith(hot_suffixes)
    return False


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


def quantize_marlin4_groupwise(w: torch.Tensor, group_size: int = 128, out_chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, dict]:
    if w.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(w.shape)}")
    if group_size not in (-1, 128):
        raise ValueError("Marlin layout supports group_size -1 or 128")
    w = w.detach().to(torch.float16).cpu().contiguous()
    rows, cols = w.shape
    if cols % 128 != 0 or rows % 256 != 0:
        raise ValueError(f"Marlin requires cols % 128 == 0 and rows % 256 == 0, got {rows}x{cols}")
    effective_group = cols if group_size == -1 else group_size
    if cols % effective_group != 0:
        raise ValueError("cols must be divisible by group_size")
    if out_chunk % 256 != 0:
        raise ValueError("out_chunk must be divisible by 256")

    qweight = np.zeros((cols // 16, rows * 16 // 8), dtype=np.int32)
    scales = np.zeros((cols // effective_group, rows), dtype=np.float16)
    maxq = 15
    zero = 8
    perm = _MARLIN_PERM
    scale_perm = _MARLIN_SCALE_PERM if effective_group != cols else _MARLIN_SCALE_PERM_SINGLE

    for start in range(0, rows, out_chunk):
        end = min(rows, start + out_chunk)
        if (end - start) % 256 != 0:
            raise ValueError("final Marlin output chunk must be divisible by 256")
        wc = w[start:end].t().contiguous()  # [K, N_chunk]
        n_chunk = wc.size(1)
        if effective_group != cols:
            grouped = wc.reshape(cols // effective_group, effective_group, n_chunk)
            s = grouped.abs().amax(dim=1) * (2.0 / maxq)
            s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s)).to(torch.float16)
            w_work = wc.reshape((-1, effective_group, n_chunk)).permute(1, 0, 2).reshape((effective_group, -1))
            q = torch.round(w_work / s.reshape(1, -1)).to(torch.int32)
            q = torch.clamp(q + zero, 0, maxq)
            q = q.reshape((effective_group, -1, n_chunk)).permute(1, 0, 2).reshape((cols, n_chunk)).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm].reshape((-1, n_chunk)).contiguous()
        else:
            s = wc.abs().amax(dim=0, keepdim=True) * (2.0 / maxq)
            s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s)).to(torch.float16)
            q = torch.round(wc / s).to(torch.int32)
            q = torch.clamp(q + zero, 0, maxq).contiguous()
            s = s.reshape((-1, len(scale_perm)))[:, scale_perm].reshape((-1, n_chunk)).contiguous()

        q = q.reshape((cols // 16, 16, n_chunk // 16, 16))
        q = q.permute((0, 2, 1, 3)).reshape((cols // 16, n_chunk * 16))
        q = q.reshape((-1, perm.numel()))[:, perm].reshape(q.shape).cpu().numpy().astype(np.uint32)
        packed = np.zeros((q.shape[0], q.shape[1] // 8), dtype=np.uint32)
        for i in range(8):
            packed |= q[:, i::8] << (4 * i)
        qweight[:, start * 2 : end * 2] = packed.astype(np.int32)
        scales[:, start:end] = s.to(torch.float16).numpy()

    meta = {
        "rows": rows,
        "cols": cols,
        "group_size": effective_group,
        "n_groups": cols // effective_group,
        "packed_cols": rows * 16 // 8,
        "bits": 4,
        "exec_bits": 4,
        "layout": "marlin_v1",
    }
    return qweight, scales, meta


def write_array(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))


def write_fp16_tensor(*, out_dir: Path, index: dict, name: str, tensor: torch.Tensor) -> None:
    rel = name.replace(".", "__")
    arr = tensor.detach().cpu().to(torch.float16).contiguous().numpy()
    raw_path = Path("fp16") / f"{rel}.fp16.bin"
    write_array(out_dir / raw_path, arr)
    index["tensors"][name] = {
        "kind": "fp16_raw",
        "path": str(raw_path),
        "shape": list(tensor.shape),
        "dtype": "float16",
    }


def build_tensor_locations(model_dir: Path) -> dict[str, Path]:
    locations: dict[str, Path] = {}
    for st_path in iter_safetensors(model_dir):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                locations[name] = st_path
    return locations


def build_fusion_plan(names: set[str]) -> tuple[dict[str, tuple[str, ...]], set[str]]:
    plan: dict[str, tuple[str, ...]] = {}
    skip: set[str] = set()
    for name in sorted(names):
        if name.endswith(".mlp.gate_proj.weight"):
            up = name.replace(".mlp.gate_proj.weight", ".mlp.up_proj.weight")
            if up in names:
                fused = name.replace(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight")
                plan[fused] = (name, up)
                skip.update((name, up))
        elif name.endswith(".linear_attn.in_proj_qkv.weight") or name.endswith(".linear_attention.in_proj_qkv.weight"):
            z = name.replace(".in_proj_qkv.weight", ".in_proj_z.weight")
            if z in names:
                fused = name.replace(".in_proj_qkv.weight", ".in_proj_qkvz.weight")
                plan[fused] = (name, z)
                skip.update((name, z))
        elif name.endswith(".linear_attn.in_proj_b.weight") or name.endswith(".linear_attention.in_proj_b.weight"):
            a = name.replace(".in_proj_b.weight", ".in_proj_a.weight")
            if a in names:
                fused = name.replace(".in_proj_b.weight", ".in_proj_ba.weight")
                plan[fused] = (name, a)
                skip.update((name, a))
        elif name.endswith(".self_attn.q_proj.weight"):
            k = name.replace(".q_proj.weight", ".k_proj.weight")
            v = name.replace(".q_proj.weight", ".v_proj.weight")
            if k in names and v in names:
                fused = name.replace(".q_proj.weight", ".qkv_proj.weight")
                plan[fused] = (name, k, v)
                skip.update((name, k, v))
    return plan, skip


def read_source_tensor(name: str, locations: dict[str, Path]) -> torch.Tensor:
    path = locations[name]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def write_quantized_tensor(
    *,
    out_dir: Path,
    index: dict,
    name: str,
    tensor: torch.Tensor,
    group_size: int,
    bits: int,
    layout: str,
    hybrid_policy: str,
) -> None:
    rel = name.replace(".", "__")
    use_marlin = should_use_marlin(name, tensor, layout=layout, hybrid_policy=hybrid_policy, group_size=group_size)
    if use_marlin:
        packed, scales, meta = quantize_marlin4_groupwise(tensor, group_size=group_size)
        q_path = Path("marlin_q4") / f"{rel}.marlin.int32.bin"
        s_path = Path("marlin_q4") / f"{rel}.scale.fp16.bin"
    else:
        packed, scales, meta = quantize_symmetric_lowbit(tensor, group_size=group_size, bits=bits)
        q_path = Path(f"q{bits}") / f"{rel}.q{bits}.bin"
        s_path = Path(f"q{bits}") / f"{rel}.scale.fp16.bin"
    write_array(out_dir / q_path, packed)
    write_array(out_dir / s_path, scales)
    index["tensors"][name] = {
        "kind": "lowbit_marlin_groupwise" if use_marlin else "lowbit_symmetric_groupwise",
        "qweight": str(q_path),
        "scales": str(s_path),
        **meta,
    }


def convert(
    model_dir: Path,
    out_dir: Path,
    group_size: int = 128,
    max_tensors: int | None = None,
    *,
    fp16_embed: bool = False,
    bits: int = 4,
    layout: str = "rowwise",
    hybrid_policy: str = "none",
    fuse_projections: bool = False,
) -> None:
    if layout not in ("rowwise", "marlin"):
        raise ValueError("layout must be rowwise or marlin")
    if hybrid_policy not in ("none", "q3q4_hot"):
        raise ValueError("hybrid_policy must be none or q3q4_hot")
    if layout == "marlin" and bits != 4:
        raise ValueError("Marlin layout currently supports bits=4 only")
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {
        "format": f"qwenburst-q{bits}-{layout}-v1" if layout != "rowwise" else f"qwenburst-q{bits}-v4",
        "group_size": group_size,
        "bits": bits,
        "layout": layout,
        "hybrid_policy": hybrid_policy,
        "fuse_projections": fuse_projections,
        "fp16_embed": fp16_embed,
        "tensors": {},
    }
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        index["hf_config_json"] = json.loads(cfg_path.read_text(encoding="utf-8"))
    count = 0

    if safe_open is None:
        raise RuntimeError("safetensors is required for checkpoint conversion: pip install safetensors")

    locations = build_tensor_locations(model_dir)
    fusion_plan, fused_sources = build_fusion_plan(set(locations)) if fuse_projections else ({}, set())

    for st_path in iter_safetensors(model_dir):
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for name in tqdm(f.keys(), desc=st_path.name):
                if max_tensors is not None and count >= max_tensors:
                    break
                if name in fused_sources:
                    continue
                tensor = f.get_tensor(name)
                if should_quantize(name, fp16_embed=fp16_embed) and tensor.ndim == 2:
                    write_quantized_tensor(
                        out_dir=out_dir,
                        index=index,
                        name=name,
                        tensor=tensor,
                        group_size=group_size,
                        bits=bits,
                        layout=layout,
                        hybrid_policy=hybrid_policy,
                    )
                    count += 1
                else:
                    write_fp16_tensor(out_dir=out_dir, index=index, name=name, tensor=tensor)
        if max_tensors is not None and count >= max_tensors:
            break

    if max_tensors is None:
        for fused_name, source_names in tqdm(fusion_plan.items(), desc="fused projections"):
            parts = [read_source_tensor(src, locations) for src in source_names]
            tensor = torch.cat(parts, dim=0)
            if fused_name.endswith(".in_proj_ba.weight") and layout == "marlin":
                # Qwen3.6 in_proj_ba is only 96 output rows, below Marlin's
                # N%256 contract. FP16 fused ba is faster than two tiny scalar
                # low-bit GEMVs and avoids quantizing recurrent gates.
                write_fp16_tensor(out_dir=out_dir, index=index, name=fused_name, tensor=tensor)
                continue
            write_quantized_tensor(
                out_dir=out_dir,
                index=index,
                name=fused_name,
                tensor=tensor,
                group_size=group_size,
                bits=bits,
                layout=layout,
                hybrid_policy=hybrid_policy,
            )

    with open(out_dir / "qwenburst_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert HF safetensors to QwenBurst low-bit format")
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=4, help="weight bits for quantized 2D tensors")
    ap.add_argument("--layout", choices=("rowwise", "marlin"), default="rowwise", help="runtime weight layout")
    ap.add_argument("--hybrid-policy", choices=("none", "q3q4_hot"), default="none", help="Q3/Q4 mixed runtime policy")
    ap.add_argument("--fuse-projections", action="store_true", help="store fused MLP/GDN/attention projections to reduce decode kernel launches")
    ap.add_argument("--max-tensors", type=int, default=None, help="debug: convert only N quantized tensors")
    ap.add_argument("--fp16-embed", action="store_true", help="keep embed_tokens.weight fp16; not recommended for 16GB")
    args = ap.parse_args()
    convert(
        args.model_dir,
        args.out_dir,
        group_size=args.group_size,
        max_tensors=args.max_tensors,
        fp16_embed=args.fp16_embed,
        bits=args.bits,
        layout=args.layout,
        hybrid_policy=args.hybrid_policy,
        fuse_projections=args.fuse_projections,
    )


if __name__ == "__main__":
    main()
