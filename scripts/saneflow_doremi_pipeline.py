#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DoReMi-style proxy Group DRO and rebuild the practical mix.")
    parser.add_argument("--recipe", default="configs/saneflow_practical_pretrain_mix.json")
    parser.add_argument("--out", default="runs/doremi_proxy_practical_v1")
    parser.add_argument("--reference", default="")
    parser.add_argument("--reference-out", default="runs/doremi_reference_practical_v1")
    parser.add_argument("--reference-steps", type=int, default=300)
    parser.add_argument("--tokenizer-path", default="tokenizers/saneflow_fineweb_edu_16k")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--skip-proxy-if-ratios-exist", action="store_true")
    args = parser.parse_args()

    recipe = json.loads((ROOT / args.recipe).read_text(encoding="utf-8"))
    ratio_override = recipe.get("ratio_override") or str(Path(recipe["manifest_out"]).with_name("doremi_ratios.json"))
    ratio_out = ROOT / ratio_override
    proxy_ratio = ROOT / args.out / "doremi_ratios.json"
    if not (args.skip_proxy_if_ratios_exist and ratio_out.exists()):
        reference = args.reference
        if not reference:
            reference_ckpt = ROOT / args.reference_out / "model.pt"
            if not reference_ckpt.exists():
                ref_cmd = [
                    sys.executable,
                    "scripts/saneflow_train_doremi_proxy.py",
                    "--recipe",
                    args.recipe,
                    "--out",
                    args.reference_out,
                    "--tokenizer-path",
                    args.tokenizer_path,
                    "--steps",
                    str(args.reference_steps),
                    "--seq-len",
                    str(args.seq_len),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--mode",
                    "prior",
                ]
                if args.activation_checkpointing:
                    ref_cmd.append("--activation-checkpointing")
                if args.tf32:
                    ref_cmd.append("--tf32")
                run(ref_cmd)
            reference = str(reference_ckpt)
        cmd = [
            sys.executable,
            "scripts/saneflow_train_doremi_proxy.py",
            "--recipe",
            args.recipe,
            "--out",
            args.out,
            "--tokenizer-path",
            args.tokenizer_path,
            "--steps",
            str(args.steps),
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--mode",
            "dro",
            "--reference",
            reference,
        ]
        if args.activation_checkpointing:
            cmd.append("--activation-checkpointing")
        if args.tf32:
            cmd.append("--tf32")
        run(cmd)
        if not proxy_ratio.exists():
            raise SystemExit(f"missing proxy ratio output: {proxy_ratio}")
        ratio_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(proxy_ratio, ratio_out)
    run([sys.executable, "scripts/saneflow_build_practical_pretrain_mix.py", "--recipe", args.recipe])


if __name__ == "__main__":
    main()
