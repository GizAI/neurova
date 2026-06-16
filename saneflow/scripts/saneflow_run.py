#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.profile_registry import ProfileRegistry, RunProfile, load_registry


PYTHON = os.environ.get("SANEFLOW_PYTHON", sys.executable)


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


def build_chatml(registry: ProfileRegistry) -> None:
    subprocess.check_call(
        [
            PYTHON,
            "saneflow/scripts/saneflow_build_chatml_sft.py",
            "--train-out", registry.paths["chatml_sft_train"],
            "--valid-out", registry.paths["chatml_sft_valid"],
        ],
        cwd=ROOT,
    )


def build_speak_pretrain(registry: ProfileRegistry) -> None:
    subprocess.check_call(
        [
            PYTHON,
            "saneflow/scripts/saneflow_build_speak_pretrain_v1.py",
            "--out-dir", registry.paths["speak_pretrain_dir"],
        ],
        cwd=ROOT,
    )


def wait_for_any(paths: list[Path]) -> None:
    if not paths:
        return
    labels = ", ".join(str(path.relative_to(ROOT)) for path in paths)
    print(f"waiting for one of: {labels}", flush=True)
    while not any(path.exists() for path in paths):
        time.sleep(60)


def main() -> None:
    registry = load_registry()
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
        print(json.dumps({name: profile.as_dict() for name, profile in registry.profiles.items()}, indent=2))
        return
    if args.cmd == "list-active":
        print(json.dumps({name: registry.profiles[name].as_dict() for name in registry.active_names()}, indent=2))
        return
    if args.cmd == "status":
        names = args.profiles or list(registry.active_names())
        for name in names:
            print_status(registry.require(name))
        return
    if args.cmd == "wait-train":
        profile = registry.require(args.profile)
        if profile.init_from:
            wait_for_any([ROOT / path for path in profile.init_candidates()])
        if args.build_speak:
            build_speak_pretrain(registry)
        if args.build_chatml:
            build_chatml(registry)
        subprocess.check_call(profile.train_cmd(python=PYTHON, root=ROOT), cwd=ROOT, env=profile_env(profile))
        return

    profile = registry.require(args.profile)
    cmd = profile.train_cmd(python=PYTHON, root=ROOT)
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
