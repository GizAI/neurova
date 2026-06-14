#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


VARIANTS = {
    "state_v1": ["--state-mixer-version", "v1", "--syntax-mix-version", "v1"],
    "state_v2": ["--state-mixer-version", "v2", "--syntax-mix-version", "v1"],
    "state_v2_fixed": ["--state-mixer-version", "v2_fixed", "--syntax-mix-version", "v2", "--syntax-kernels", "3,7,15"],
    "state_v2_stable": ["--state-mixer-version", "v2", "--state-clip", "8.0", "--state-zoneout", "0.02", "--syntax-mix-version", "v1"],
    "delta_matrix": ["--state-mixer-version", "delta_matrix", "--state-clip", "8.0", "--syntax-mix-version", "v1"],
    "delta_matrix_syntax_v2": ["--state-mixer-version", "delta_matrix", "--state-clip", "8.0", "--syntax-mix-version", "v2", "--syntax-kernels", "3,7,15"],
    "delta_matrix_syntax_wide": ["--state-mixer-version", "delta_matrix", "--state-clip", "8.0", "--syntax-mix-version", "v2", "--syntax-kernels", "3,7,15,31"],
    "factorized_v2": ["--state-mixer-version", "v2", "--d-embed", "128", "--syntax-mix-version", "v1"],
    "factorized_delta": ["--state-mixer-version", "delta_matrix", "--state-clip", "8.0", "--d-embed", "128", "--syntax-mix-version", "v1"],
    "sparse_island_v2": ["--state-mixer-version", "v2", "--attention-interval", "3", "--attention-window", "64", "--syntax-mix-version", "v1"],
    "sparse_island_v2_fixed": ["--state-mixer-version", "v2_fixed", "--attention-interval", "4", "--attention-window", "64", "--syntax-mix-version", "v2", "--syntax-kernels", "3,7,15"],
    "neurova_r_v0": [
        "--state-mixer-version", "delta_matrix",
        "--state-clip", "8.0",
        "--attention-interval", "3",
        "--attention-window", "128",
        "--thought-slots", "8",
        "--landmark-interval", "4",
        "--landmark-chunk", "64",
        "--landmark-max", "64",
        "--syntax-mix-version", "v2",
        "--syntax-kernels", "3,7,15",
    ],
    "sparse_island_delta": ["--state-mixer-version", "delta_matrix", "--state-clip", "8.0", "--attention-interval", "3", "--attention-window", "64", "--syntax-mix-version", "v1"],
    "no_state": ["--state-mixer-version", "off", "--syntax-mix-version", "v1"],
    "syntax_v2": ["--state-mixer-version", "v2", "--syntax-mix-version", "v2", "--syntax-kernels", "3,7,15"],
    "conv_k3": ["--state-mixer-version", "v2", "--syntax-mix-version", "v1", "--conv-kernel", "3"],
    "conv_k7": ["--state-mixer-version", "v2", "--syntax-mix-version", "v1", "--conv-kernel", "7"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run or print a reproducible SaneFlow ablation grid.")
    p.add_argument("--out-root", default="runs/saneflow_ablation_grid")
    p.add_argument("--train-data", required=True)
    p.add_argument("--valid-data", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--variants", default="state_v1,state_v2,state_v2_fixed,sparse_island_v2_fixed,state_v2_stable,delta_matrix,delta_matrix_syntax_v2,no_state,syntax_v2,conv_k3,conv_k7")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--d-embed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=768)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--optimizer", choices=["adamw", "ademamix", "muon"], default="adamw")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--max-records", type=int, default=30000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def run(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    names = [x.strip() for x in args.variants.split(",") if x.strip()]
    manifest = {
        "out_root": str(out_root),
        "variants": names,
        "train_data": args.train_data,
        "valid_data": args.valid_data,
        "tokenizer_path": args.tokenizer_path,
        "steps": args.steps,
        "model": {
            "d_model": args.d_model,
            "d_embed": args.d_embed,
            "layers": args.layers,
            "heads": args.heads,
            "d_ff": args.d_ff,
        },
        "optimizer": args.optimizer,
    }
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"unknown variant {name!r}; choices={sorted(VARIANTS)}")
        run_dir = out_root / name
        model_path = run_dir / "model.pt"
        gate_path = run_dir / "quality_gate.json"
        train_cmd = [
            sys.executable,
            "scripts/saneflow_train.py",
            "--out", str(run_dir),
            "--train-data", args.train_data,
            "--valid-data", args.valid_data,
            "--tokenizer-path", args.tokenizer_path,
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--seq-len", str(args.seq_len),
            "--d-embed", str(args.d_embed),
            "--d-model", str(args.d_model),
            "--layers", str(args.layers),
            "--heads", str(args.heads),
            "--d-ff", str(args.d_ff),
            "--lr", str(args.lr),
            "--optimizer", args.optimizer,
            "--muon-lr", str(args.muon_lr),
            "--muon-momentum", str(args.muon_momentum),
            "--weight-decay", str(args.weight_decay),
            "--grad-clip", str(args.grad_clip),
            "--warmup-steps", str(args.warmup_steps),
            "--min-lr-ratio", str(args.min_lr_ratio),
            "--max-records", str(args.max_records),
            "--save-every", str(max(args.steps // 2, 1)),
            "--log-every", "20",
            "--device", args.device,
            "--dtype", args.dtype,
            *VARIANTS[name],
        ]
        if args.tf32:
            train_cmd.append("--tf32")
        if args.compile:
            train_cmd.append("--compile")
        if model_path.exists():
            print(f"skip train existing {model_path}", flush=True)
        else:
            run(train_cmd, args.dry_run)
        gate_cmd = [
            sys.executable,
            "scripts/saneflow_quality_gate.py",
            "--ckpt", str(model_path),
            "--out", str(gate_path),
            "--valid-data", args.valid_data,
            "--context", str(args.seq_len),
            "--context-probes", "128,256,512",
            "--max-new", "64",
            "--device", args.device,
            "--dtype", args.dtype,
        ]
        if gate_path.exists():
            print(f"skip gate existing {gate_path}", flush=True)
        else:
            run(gate_cmd, args.dry_run)


if __name__ == "__main__":
    main()
