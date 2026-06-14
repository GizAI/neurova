#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("SANEFLOW_PYTHON", sys.executable)


@dataclass(frozen=True)
class RunProfile:
    name: str
    out: str
    train_data: tuple[str, ...]
    valid_data: tuple[str, ...]
    tokenizer_path: str = "tokenizers/saneflow_fineweb_edu_16k"
    init_from: str = ""
    steps: int = 3000
    batch_size: int = 44
    grad_accum_steps: int = 1
    loss_chunk_tokens: int = 8192
    loss_mode: str = "causal"
    seq_len: int = 384
    d_embed: int = 512
    d_model: int = 672
    layers: int = 12
    heads: int = 8
    d_ff: int = 2016
    syntax_mix_version: str = "v2"
    syntax_kernels: str = "3,7,15"
    lr: str = "3e-4"
    optimizer: str = "muon"
    muon_lr: str = "0.02"
    warmup_steps: int = 300
    min_lr_ratio: str = "0.1"
    state_mixer_version: str = "v2_fixed"
    state_clip: str = "0.0"
    attention_interval: int = 4
    attention_window: int = 64
    thought_slots: int = 0
    thought_chunk: int = 1
    thought_start_layer: int = 0
    landmark_interval: int = 0
    landmark_chunk: int = 64
    landmark_max: int = 64
    save_every: int = 250
    log_every: int = 10
    dtype: str = "bf16"
    dataset_device: str = "cuda"
    activation_checkpointing: bool = False
    compile: bool = False
    cuda_visible_devices: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def train_cmd(self) -> list[str]:
        cmd = [
            PYTHON,
            "scripts/saneflow_train.py",
            "--out", self.out,
            "--train-data", *self.train_data,
            "--valid-data", *self.valid_data,
            "--tokenizer-path", self.tokenizer_path,
            "--steps", str(self.steps),
            "--batch-size", str(self.batch_size),
            "--grad-accum-steps", str(self.grad_accum_steps),
            "--loss-chunk-tokens", str(self.loss_chunk_tokens),
            "--loss-mode", self.loss_mode,
            "--seq-len", str(self.seq_len),
            "--d-embed", str(self.d_embed),
            "--d-model", str(self.d_model),
            "--layers", str(self.layers),
            "--heads", str(self.heads),
            "--d-ff", str(self.d_ff),
            "--syntax-mix-version", self.syntax_mix_version,
            "--syntax-kernels", self.syntax_kernels,
            "--lr", self.lr,
            "--optimizer", self.optimizer,
            "--muon-lr", self.muon_lr,
            "--warmup-steps", str(self.warmup_steps),
            "--min-lr-ratio", self.min_lr_ratio,
            "--state-mixer-version", self.state_mixer_version,
            "--state-clip", self.state_clip,
            "--attention-interval", str(self.attention_interval),
            "--attention-window", str(self.attention_window),
            "--thought-slots", str(self.thought_slots),
            "--thought-chunk", str(self.thought_chunk),
            "--thought-start-layer", str(self.thought_start_layer),
            "--landmark-interval", str(self.landmark_interval),
            "--landmark-chunk", str(self.landmark_chunk),
            "--landmark-max", str(self.landmark_max),
            "--save-every", str(self.save_every),
            "--log-every", str(self.log_every),
            "--device", "cuda",
            "--dtype", self.dtype,
            "--dataset-device", self.dataset_device,
            "--fused-adamw",
            "--tf32",
        ]
        if self.init_from:
            cmd.extend(["--init-from", self.init_from])
        if self.activation_checkpointing:
            cmd.append("--activation-checkpointing")
        if self.compile:
            cmd.append("--compile")
        return cmd


FINEWEB_TRAIN = "data/corpus/sources/fineweb_edu_sample10bt/train.jsonl"
FINEWEB_VALID = "data/corpus/sources/fineweb_edu_sample10bt/valid.jsonl"
SPEAK_PRETRAIN_TRAIN = "data/corpus/mixes/saneflow_speak_pretrain_v1/train.jsonl"
SPEAK_PRETRAIN_VALID = "data/corpus/mixes/saneflow_speak_pretrain_v1/valid.jsonl"
PRACTICAL_PRETRAIN_TRAIN = "data/corpus/mixes/saneflow_practical_pretrain_v1/train.jsonl"
PRACTICAL_PRETRAIN_VALID = "data/corpus/mixes/saneflow_practical_pretrain_v1/valid.jsonl"
CHATML_SFT_TRAIN = "data/corpus/mixes/saneflow_chatml_sft_train_v1.jsonl"
CHATML_SFT_VALID = "data/corpus/mixes/saneflow_chatml_sft_valid_v1.jsonl"


R_ABLATION_COMMON = dict(
    train_data=(FINEWEB_TRAIN,),
    valid_data=(FINEWEB_VALID,),
    steps=800,
    batch_size=96,
    grad_accum_steps=1,
    loss_chunk_tokens=2048,
    seq_len=256,
    d_embed=256,
    d_model=384,
    layers=8,
    heads=32,
    d_ff=1152,
    syntax_mix_version="v2",
    lr="2e-4",
    muon_lr="0.012",
    state_mixer_version="delta_matrix",
    state_clip="8.0",
    save_every=200,
    cuda_visible_devices="1",
)


PROFILES: dict[str, RunProfile] = {
    "dmc8-base-100m": RunProfile(
        name="dmc8-base-100m",
        out="runs/saneflow_fineweb_edu_base_v3_100m_muon_mem",
        train_data=(FINEWEB_TRAIN,),
        valid_data=(FINEWEB_VALID,),
        steps=5000,
        batch_size=24,
        grad_accum_steps=3,
        loss_chunk_tokens=0,
        syntax_mix_version="v1",
        state_mixer_version="v2",
        attention_interval=0,
        muon_lr="0.02",
        lr="3e-4",
        save_every=250,
    ),
    "dmc8-chatml-sft-100m": RunProfile(
        name="dmc8-chatml-sft-100m",
        out="runs/saneflow_chatml_masked_sft_v3_100m_muon_mem",
        init_from="runs/saneflow_fineweb_edu_base_v3_100m_muon_mem/model.pt",
        train_data=(CHATML_SFT_TRAIN,),
        valid_data=(CHATML_SFT_VALID,),
        steps=1800,
        batch_size=24,
        grad_accum_steps=3,
        loss_chunk_tokens=0,
        loss_mode="chatml_assistant",
        syntax_mix_version="v1",
        state_mixer_version="v2",
        attention_interval=0,
        lr="5e-5",
        muon_lr="0.004",
        warmup_steps=50,
        save_every=100,
    ),
    "dmc8-speak-base-v1": RunProfile(
        name="dmc8-speak-base-v1",
        out="runs/saneflow_speak_base_v1_100m",
        init_from="runs/saneflow_fineweb_edu_base_v3_100m_muon_mem/model.pt",
        train_data=(SPEAK_PRETRAIN_TRAIN,),
        valid_data=(SPEAK_PRETRAIN_VALID,),
        steps=12000,
        batch_size=24,
        grad_accum_steps=3,
        loss_chunk_tokens=0,
        loss_mode="causal",
        dataset_device="cpu",
        seq_len=384,
        d_embed=512,
        d_model=672,
        layers=12,
        heads=8,
        d_ff=2016,
        syntax_mix_version="v1",
        state_mixer_version="v2",
        attention_interval=0,
        lr="8e-5",
        muon_lr="0.006",
        warmup_steps=120,
        min_lr_ratio="0.12",
        save_every=500,
        cuda_visible_devices="0",
    ),
    "dmc8-chatml-sft-v9": RunProfile(
        name="dmc8-chatml-sft-v9",
        out="runs/saneflow_chatml_sft_v9_assistant",
        init_from="runs/saneflow_speak_base_v1_100m/model.pt",
        train_data=(CHATML_SFT_TRAIN,),
        valid_data=(CHATML_SFT_VALID,),
        steps=1600,
        batch_size=24,
        grad_accum_steps=3,
        loss_chunk_tokens=0,
        loss_mode="chatml_assistant",
        dataset_device="cpu",
        seq_len=512,
        d_embed=512,
        d_model=672,
        layers=12,
        heads=8,
        d_ff=2016,
        syntax_mix_version="v1",
        state_mixer_version="v2",
        attention_interval=0,
        lr="1.2e-5",
        muon_lr="0.0008",
        warmup_steps=100,
        min_lr_ratio="0.2",
        save_every=100,
        cuda_visible_devices="0",
    ),
    "dmc9-practical-base-100m": RunProfile(
        name="dmc9-practical-base-100m",
        out="runs/saneflow_practical_pretrain_v1_100m_muon",
        train_data=(PRACTICAL_PRETRAIN_TRAIN,),
        valid_data=(PRACTICAL_PRETRAIN_VALID,),
        steps=8000,
        batch_size=44,
        grad_accum_steps=1,
        loss_chunk_tokens=8192,
        seq_len=384,
        d_embed=512,
        d_model=672,
        layers=12,
        heads=8,
        d_ff=2016,
        syntax_mix_version="v2",
        state_mixer_version="v2_fixed",
        attention_interval=4,
        lr="2e-4",
        muon_lr="0.012",
        warmup_steps=400,
        save_every=500,
        cuda_visible_devices="0",
    ),
    "dmc9-sparse-chatml-sft": RunProfile(
        name="dmc9-sparse-chatml-sft",
        out="runs/saneflow_100m_research_v4/v2_fixed_sparse_chatml_masked_sft_v3",
        init_from="runs/saneflow_100m_research_v4/v2_fixed_sparse_100m_muon_b44_cuda/model.pt",
        train_data=(CHATML_SFT_TRAIN,),
        valid_data=(CHATML_SFT_VALID,),
        steps=1800,
        batch_size=44,
        grad_accum_steps=1,
        loss_chunk_tokens=8192,
        loss_mode="chatml_assistant",
        syntax_mix_version="v2",
        state_mixer_version="v2_fixed",
        attention_interval=4,
        lr="2e-5",
        muon_lr="0.002",
        warmup_steps=80,
        save_every=100,
        cuda_visible_devices="0",
    ),
    "dmc9-neurova-r-full": RunProfile(
        name="dmc9-neurova-r-full",
        out="runs/saneflow_neurova_r_full_chunked/delta_sparse_thought_landmark_d384_l8_h32_b96_s256_tc64_eval50",
        train_data=(FINEWEB_TRAIN,),
        valid_data=(FINEWEB_VALID,),
        steps=3000,
        batch_size=96,
        grad_accum_steps=1,
        loss_chunk_tokens=2048,
        seq_len=256,
        d_embed=256,
        d_model=384,
        layers=8,
        heads=32,
        d_ff=1152,
        syntax_mix_version="v2",
        lr="2e-4",
        muon_lr="0.012",
        state_mixer_version="delta_matrix",
        state_clip="8.0",
        attention_interval=3,
        attention_window=128,
        thought_slots=8,
        thought_chunk=64,
        landmark_interval=4,
        landmark_chunk=64,
        landmark_max=64,
        save_every=50,
        cuda_visible_devices="1",
    ),
    "dmc9-r-a-delta-only": RunProfile(
        name="dmc9-r-a-delta-only",
        out="runs/saneflow_r_ablation/a_delta_only",
        attention_interval=0,
        thought_slots=0,
        landmark_interval=0,
        **R_ABLATION_COMMON,
    ),
    "dmc9-r-b-delta-sparse-attn": RunProfile(
        name="dmc9-r-b-delta-sparse-attn",
        out="runs/saneflow_r_ablation/b_delta_sparse_attn",
        attention_interval=3,
        attention_window=128,
        thought_slots=0,
        landmark_interval=0,
        **R_ABLATION_COMMON,
    ),
    "dmc9-r-c-delta-thought-late": RunProfile(
        name="dmc9-r-c-delta-thought-late",
        out="runs/saneflow_r_ablation/c_delta_thought_late",
        attention_interval=0,
        thought_slots=4,
        thought_chunk=64,
        thought_start_layer=6,
        landmark_interval=0,
        **R_ABLATION_COMMON,
    ),
    "dmc9-r-d-delta-landmark": RunProfile(
        name="dmc9-r-d-delta-landmark",
        out="runs/saneflow_r_ablation/d_delta_landmark",
        attention_interval=0,
        thought_slots=0,
        landmark_interval=4,
        landmark_chunk=64,
        landmark_max=64,
        **R_ABLATION_COMMON,
    ),
    "dmc9-r-e-full-lite": RunProfile(
        name="dmc9-r-e-full-lite",
        out="runs/saneflow_r_ablation/e_full_lite",
        attention_interval=3,
        attention_window=128,
        thought_slots=4,
        thought_chunk=64,
        thought_start_layer=6,
        landmark_interval=4,
        landmark_chunk=64,
        landmark_max=64,
        **R_ABLATION_COMMON,
    ),
    "dmc9-r-champion-delta-landmark-long": RunProfile(
        name="dmc9-r-champion-delta-landmark-long",
        out="runs/saneflow_r_champion/d_delta_landmark_long",
        init_from="runs/saneflow_r_ablation/d_delta_landmark/model.pt",
        train_data=(FINEWEB_TRAIN,),
        valid_data=(FINEWEB_VALID,),
        steps=2200,
        batch_size=96,
        grad_accum_steps=1,
        loss_chunk_tokens=2048,
        seq_len=256,
        d_embed=256,
        d_model=384,
        layers=8,
        heads=32,
        d_ff=1152,
        syntax_mix_version="v2",
        lr="1.2e-4",
        muon_lr="0.007",
        warmup_steps=120,
        min_lr_ratio="0.1",
        state_mixer_version="delta_matrix",
        state_clip="8.0",
        attention_interval=0,
        thought_slots=0,
        landmark_interval=4,
        landmark_chunk=64,
        landmark_max=64,
        save_every=200,
        cuda_visible_devices="1",
    ),
    "dmc9-r-champion-practical-cont": RunProfile(
        name="dmc9-r-champion-practical-cont",
        out="runs/saneflow_r_champion/d_delta_landmark_practical_cont",
        init_from="runs/saneflow_r_champion/d_delta_landmark_long/model.pt",
        train_data=(PRACTICAL_PRETRAIN_TRAIN,),
        valid_data=(PRACTICAL_PRETRAIN_VALID,),
        steps=5000,
        batch_size=96,
        grad_accum_steps=1,
        loss_chunk_tokens=2048,
        seq_len=256,
        d_embed=256,
        d_model=384,
        layers=8,
        heads=32,
        d_ff=1152,
        syntax_mix_version="v2",
        lr="9e-5",
        muon_lr="0.005",
        warmup_steps=150,
        min_lr_ratio="0.12",
        state_mixer_version="delta_matrix",
        state_clip="8.0",
        attention_interval=0,
        thought_slots=0,
        landmark_interval=4,
        landmark_chunk=64,
        landmark_max=64,
        save_every=250,
        cuda_visible_devices="1",
    ),
}

ACTIVE_PROFILES = (
    "dmc8-speak-base-v1",
    "dmc8-chatml-sft-v9",
    "dmc9-practical-base-100m",
    "dmc9-r-champion-delta-landmark-long",
    "dmc9-r-champion-practical-cont",
)


def require_profile(name: str) -> RunProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise SystemExit(f"unknown profile {name!r}; choices: {', '.join(sorted(PROFILES))}") from exc


def profile_env(profile: RunProfile) -> dict[str, str]:
    env = os.environ.copy()
    env.update(profile.env)
    if profile.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = profile.cuda_visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def print_status(profile: RunProfile) -> None:
    out = ROOT / profile.out
    print(f"== {profile.name} ==")
    print(f"out={profile.out}")
    for path in (out / "train_log.jsonl", out / "latest.pt", out / "model.pt", out / "train.out"):
        if path.exists():
            print(f"{path.relative_to(ROOT)} size={path.stat().st_size}")
    log = out / "train_log.jsonl"
    if log.exists():
        rows = [line for line in log.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
        for line in rows[-10:]:
            print(line)


def build_chatml() -> None:
    subprocess.check_call([
        PYTHON,
        "scripts/saneflow_build_chatml_sft.py",
        "--train-out", CHATML_SFT_TRAIN,
        "--valid-out", CHATML_SFT_VALID,
    ], cwd=ROOT)


def build_speak_pretrain() -> None:
    subprocess.check_call([
        PYTHON,
        "scripts/saneflow_build_speak_pretrain_v1.py",
        "--out-dir", "data/corpus/mixes/saneflow_speak_pretrain_v1",
    ], cwd=ROOT)


def wait_for(path: Path) -> None:
    print(f"waiting for {path.relative_to(ROOT)}", flush=True)
    while not path.exists():
        time.sleep(60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SaneFlow profile registry and launcher.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    sub.add_parser("list-active")
    p_print = sub.add_parser("print-cmd")
    p_print.add_argument("profile")
    p_train = sub.add_parser("train")
    p_train.add_argument("profile")
    p_start = sub.add_parser("start")
    p_start.add_argument("profile")
    p_status = sub.add_parser("status")
    p_status.add_argument("profiles", nargs="*")
    p_wait = sub.add_parser("wait-train")
    p_wait.add_argument("profile")
    p_wait.add_argument("--build-chatml", action="store_true")
    p_wait.add_argument("--build-speak", action="store_true")

    args = parser.parse_args()
    if args.cmd == "list":
        print(json.dumps({name: profile.__dict__ for name, profile in PROFILES.items()}, indent=2))
        return
    if args.cmd == "list-active":
        print(json.dumps({name: PROFILES[name].__dict__ for name in ACTIVE_PROFILES}, indent=2))
        return
    if args.cmd == "status":
        names = args.profiles or sorted(PROFILES)
        for name in names:
            print_status(require_profile(name))
        return
    if args.cmd == "wait-train":
        profile = require_profile(args.profile)
        if profile.init_from:
            wait_for(ROOT / profile.init_from)
        if args.build_speak:
            build_speak_pretrain()
        if args.build_chatml:
            build_chatml()
        subprocess.check_call(profile.train_cmd(), cwd=ROOT, env=profile_env(profile))
        return

    profile = require_profile(args.profile)
    cmd = profile.train_cmd()
    if args.cmd == "print-cmd":
        print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    elif args.cmd == "train":
        os.execvpe(cmd[0], cmd, profile_env(profile))
    elif args.cmd == "start":
        out = ROOT / profile.out
        out.mkdir(parents=True, exist_ok=True)
        log = out / "train.out"
        with log.open("ab") as f:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=profile_env(profile), stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"started pid={proc.pid} profile={profile.name} log={log.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
