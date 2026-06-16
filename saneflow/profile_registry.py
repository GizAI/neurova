from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_CONFIG = ROOT / "saneflow/configs/saneflow_profiles.json"


@dataclass(frozen=True)
class RunProfile:
    name: str
    out: str
    train_data: tuple[str, ...]
    valid_data: tuple[str, ...]
    tokenizer_path: str
    init_from: str = ""
    init_from_fallbacks: tuple[str, ...] = ()
    steps: int = 3000
    batch_size: int = 44
    eval_batch_size: int = 0
    grad_accum_steps: int = 1
    loss_chunk_tokens: int = 8192
    loss_mode: str = "causal"
    seq_len: int = 384
    model_type: str = "saneflow"
    d_embed: int = 512
    d_model: int = 672
    layers: int = 12
    heads: int = 8
    kv_heads: int = 0
    d_ff: int = 2016
    rope_theta: str = "10000.0"
    qk_norm: bool = False
    syntax_mix_version: str = "v2"
    syntax_kernels: str = "3,7,15"
    lr: str = "3e-4"
    optimizer: str = "muon"
    muon_lr: str = "0.02"
    galore_rank: int = 128
    galore_update_proj_gap: int = 200
    galore_scale: str = "0.25"
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
    no_save_optimizer: bool = False
    liger_fused_linear_ce: bool = False
    cuda_visible_devices: str = ""
    env: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def init_candidates(self) -> tuple[str, ...]:
        return tuple(x for x in (self.init_from, *self.init_from_fallbacks) if x)

    def resolved_init_from(self, root: Path = ROOT) -> str:
        candidates = self.init_candidates()
        for path in candidates:
            if (root / path).exists():
                return path
        return candidates[0] if candidates else ""

    def train_cmd(self, *, python: str | None = None, root: Path = ROOT) -> list[str]:
        executable = python or os.environ.get("SANEFLOW_PYTHON", sys.executable)
        cmd = [
            executable,
            "saneflow/scripts/saneflow_train.py",
            "--out", self.out,
            "--train-data", *self.train_data,
            "--valid-data", *self.valid_data,
            "--tokenizer-path", self.tokenizer_path,
            "--steps", str(self.steps),
            "--batch-size", str(self.batch_size),
            "--eval-batch-size", str(self.eval_batch_size),
            "--grad-accum-steps", str(self.grad_accum_steps),
            "--loss-chunk-tokens", str(self.loss_chunk_tokens),
            "--loss-mode", self.loss_mode,
            "--seq-len", str(self.seq_len),
            "--model-type", self.model_type,
            "--d-embed", str(self.d_embed),
            "--d-model", str(self.d_model),
            "--layers", str(self.layers),
            "--heads", str(self.heads),
            "--kv-heads", str(self.kv_heads),
            "--d-ff", str(self.d_ff),
            "--rope-theta", self.rope_theta,
            "--syntax-mix-version", self.syntax_mix_version,
            "--syntax-kernels", self.syntax_kernels,
            "--lr", self.lr,
            "--optimizer", self.optimizer,
            "--muon-lr", self.muon_lr,
            "--galore-rank", str(self.galore_rank),
            "--galore-update-proj-gap", str(self.galore_update_proj_gap),
            "--galore-scale", self.galore_scale,
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
        init_from = self.resolved_init_from(root)
        if init_from:
            cmd.extend(["--init-from", init_from])
        if self.activation_checkpointing:
            cmd.append("--activation-checkpointing")
        if self.no_save_optimizer:
            cmd.append("--no-save-optimizer")
        if self.liger_fused_linear_ce:
            cmd.append("--liger-fused-linear-ce")
        if self.qk_norm:
            cmd.append("--qk-norm")
        if self.compile:
            cmd.append("--compile")
        return cmd


@dataclass(frozen=True)
class ProfileRegistry:
    profiles: dict[str, RunProfile]
    paths: dict[str, str]
    tokenizer_path: str

    def require(self, name: str) -> RunProfile:
        try:
            return self.profiles[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self.profiles))
            raise SystemExit(f"unknown profile {name!r}; choices: {choices}") from exc

    def active_names(self, program_config: Path = ROOT / "saneflow/configs/saneflow_research_program.json") -> tuple[str, ...]:
        if not program_config.exists():
            return tuple(self.profiles)
        data = json.loads(program_config.read_text(encoding="utf-8"))
        names: list[str] = []
        for line in data.get("active_lines", []):
            for name in line.get("profiles", []):
                if name in self.profiles and name not in names:
                    names.append(name)
        return tuple(names) or tuple(self.profiles)


def _coerce_profile(name: str, raw: dict, defaults: dict, tokenizer_path: str) -> RunProfile:
    data = {**defaults, **raw}
    data["name"] = name
    data.setdefault("tokenizer_path", tokenizer_path)
    data["train_data"] = tuple(data.get("train_data") or ())
    data["valid_data"] = tuple(data.get("valid_data") or ())
    data["init_from_fallbacks"] = tuple(data.get("init_from_fallbacks") or ())
    return RunProfile(**data)


def load_registry(path: Path = DEFAULT_PROFILE_CONFIG) -> ProfileRegistry:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_path = data["tokenizer_path"]
    defaults = {"tokenizer_path": tokenizer_path, **data.get("defaults", {})}
    profiles = {
        name: _coerce_profile(name, raw, defaults, tokenizer_path)
        for name, raw in data.get("profiles", {}).items()
    }
    return ProfileRegistry(profiles=profiles, paths=data.get("paths", {}), tokenizer_path=tokenizer_path)
