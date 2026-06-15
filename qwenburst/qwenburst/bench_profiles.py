from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters import Qwen36Adapter  # noqa: F401 - registers qwen36
from .core.adapter import adapter_registry
from .core.features import RuntimeFeatureOverride, RuntimeFeatures, RuntimeProfile
from .core.runtime import GenerationConfig, RuntimeEngine


@dataclass(frozen=True)
class ProfileBenchResult:
    profile: str
    features: dict[str, object]
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
    block_prefill: bool | None = None,
    prefill_chunk_size: int | None = None,
) -> ProfileBenchResult:
    engine.features = RuntimeFeatures.from_profile(profile).with_overrides(
        RuntimeFeatureOverride(block_prefill=block_prefill, prefill_chunk_size=prefill_chunk_size)
    )
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate_ids_greedy_gpu(prompt_ids, cfg)
    if torch.cuda.is_available() and str(engine.device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return ProfileBenchResult(profile=profile, features=engine.features.summary(), generated=len(out), elapsed_s=elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark qwenburst runtime feature profiles with one model load")
    parser.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    parser.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    parser.add_argument("--adapter", default="qwen36", choices=("qwen36",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recent-window", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", default="Write a concise technical note about quantized LLM inference.")
    parser.add_argument("--profiles", default="original,stateful,research")
    parser.add_argument("--block-prefill", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    args = parser.parse_args()
    block_prefill = None if args.block_prefill == "auto" else args.block_prefill == "on"

    engine = RuntimeEngine(
        adapter=adapter_registry.get(args.adapter),
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        features=RuntimeFeatures.from_profile("stateful"),
    )
    prompt_ids = engine.encode_prompt(args.prompt)
    cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0, top_k=0, eos_token_ids=())
    summary_keys = tuple(RuntimeFeatures().summary().keys())
    print("profile,generated,elapsed_s,tok_s," + ",".join(summary_keys))
    for raw in args.profiles.split(","):
        profile = raw.strip()
        if not profile:
            continue
        result = bench_profile(
            engine,
            prompt_ids,
            profile,  # type: ignore[arg-type]
            cfg,
            block_prefill=block_prefill,
            prefill_chunk_size=args.prefill_chunk_size,
        )
        values = ",".join(str(result.features[key]) for key in summary_keys)
        print(f"{result.profile},{result.generated},{result.elapsed_s:.3f},{result.tok_s:.2f},{values}")


if __name__ == "__main__":
    main()
