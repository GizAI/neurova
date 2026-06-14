#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "A_delta_only": "runs/saneflow_r_ablation/a_delta_only",
    "B_delta_sparse_attn": "runs/saneflow_r_ablation/b_delta_sparse_attn",
    "C_delta_thought_late": "runs/saneflow_r_ablation/c_delta_thought_late",
    "D_delta_landmark": "runs/saneflow_r_ablation/d_delta_landmark",
    "E_full_lite": "runs/saneflow_r_ablation/e_full_lite",
}

PROMPTS = [
    "Once upon a time",
    "Explain what a computer is in simple English.",
    "The moon is",
]


def load_metrics(run_dir: Path) -> dict:
    rows = []
    log = run_dir / "train_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.startswith("{"):
                rows.append(json.loads(line))
    valid_rows = [r for r in rows if "valid_loss" in r]
    return {
        "last_step": rows[-1]["step"] if rows else None,
        "last_loss": rows[-1].get("loss") if rows else None,
        "last_valid_loss": valid_rows[-1]["valid_loss"] if valid_rows else None,
        "best_valid_loss": min((r["valid_loss"] for r in valid_rows), default=None),
        "tok_s": rows[-1].get("tok_s") if rows else None,
    }


def generate(ckpt: Path, prompt: str) -> str:
    if not ckpt.exists():
        return ""
    cmd = [
        sys.executable,
        "scripts/saneflow_generate.py",
        "--ckpt",
        str(ckpt),
        "--prompt",
        prompt,
        "--max-new",
        "80",
        "--context",
        "256",
        "--temperature",
        "0.6",
        "--top-k",
        "30",
        "--top-p",
        "0.9",
        "--device",
        "cuda",
        "--dtype",
        "bf16",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=60)
    if result.returncode != 0:
        return result.stderr.strip()[-500:]
    return result.stdout.strip()


def main() -> None:
    report = {}
    for name, run in RUNS.items():
        run_dir = ROOT / run
        row = load_metrics(run_dir)
        ckpt = run_dir / "model.pt"
        row["samples"] = {prompt: generate(ckpt, prompt) for prompt in PROMPTS}
        report[name] = row
    out = ROOT / "runs/saneflow_r_ablation/report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
