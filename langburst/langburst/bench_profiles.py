from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import torch

from .cli_features import (
    add_adapter_arg,
    add_model_path_args,
    add_runtime_feature_args,
    create_runtime_engine_from_args,
    runtime_feature_override_from_args,
)
from .core.features import RuntimeFeatureOverride, RuntimeFeatures, RuntimeProfile
from .core.runtime import GenerationConfig, RuntimeEngine


@dataclass(frozen=True)
class ProfileBenchResult:
    profile: str
    features: dict[str, object]
    effective_features: dict[str, object]
    disabled: tuple[str, ...]
    generated: int
    elapsed_s: float

    @property
    def tok_s(self) -> float:
        return self.generated / max(self.elapsed_s, 1e-9)


def bench_profile(
    engine: RuntimeEngine,
    prompt_ids: list[int],
    profile: RuntimeProfile,
    cfg: GenerationConfig,
    *,
    override: RuntimeFeatureOverride | None = None,
) -> ProfileBenchResult:
    engine.features = RuntimeFeatures.from_profile(profile).with_overrides(override)
    plan = engine.resolve_plan()
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate_ids_greedy_gpu(prompt_ids, cfg)
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return ProfileBenchResult(
        profile=profile,
        features=engine.features.summary(),
        effective_features=plan.effective.summary(),
        disabled=plan.disabled,
        generated=len(out),
        elapsed_s=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark runtime feature profiles with one adapter/model load")
    add_model_path_args(parser)
    add_adapter_arg(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Write a concise technical note about quantized LLM inference.")
    parser.add_argument("--profiles", default="original,stateful,research")
    add_runtime_feature_args(parser, include_profile=False)
    args = parser.parse_args()
    override = runtime_feature_override_from_args(args)

    engine = create_runtime_engine_from_args(args, features=RuntimeFeatures.from_profile("stateful"))
    prompt_ids = engine.encode_prompt(args.prompt)
    cfg = GenerationConfig.greedy(max_new_tokens=args.max_new_tokens)
    summary_keys = tuple(RuntimeFeatures().summary().keys())
    print("profile,generated,elapsed_s,tok_s,disabled," + ",".join(summary_keys))
    for raw in args.profiles.split(","):
        profile = raw.strip()
        if not profile:
            continue
        result = bench_profile(
            engine,
            prompt_ids,
            profile,  # type: ignore[arg-type]
            cfg,
            override=override,
        )
        values = ",".join(str(result.effective_features[key]) for key in summary_keys)
        print(f"{result.profile},{result.generated},{result.elapsed_s:.3f},{result.tok_s:.2f},{'|'.join(result.disabled)},{values}")


if __name__ == "__main__":
    main()
