#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        raise ValueError("all weights are zero")
    return {name: max(0.0, value) / total for name, value in weights.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Update practical-pretrain source ratios from per-source losses.")
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_practical_pretrain_mix.json"))
    parser.add_argument("--source-losses", type=Path, default=Path("saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/source_losses.json"))
    parser.add_argument("--reference-losses", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/doremi_ratios.json"))
    parser.add_argument("--eta", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--min-ratio", type=float, default=0.03)
    parser.add_argument("--max-ratio", type=float, default=0.55)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    priors = normalize({source["name"]: float(source["ratio"]) for source in recipe["sources"]})
    proxy_payload = json.loads(args.source_losses.read_text(encoding="utf-8")) if args.source_losses.exists() else {}
    reference_payload = (
        json.loads(args.reference_losses.read_text(encoding="utf-8"))
        if args.reference_losses is not None and args.reference_losses.exists() and args.reference_losses.is_file()
        else {}
    )
    proxy_losses = {
        name: float(row["loss"])
        for name, row in proxy_payload.get("source_losses", {}).items()
        if row.get("loss") is not None and name in priors
    }
    reference_losses = {
        name: float(row["loss"])
        for name, row in reference_payload.get("source_losses", {}).items()
        if row.get("loss") is not None and name in priors
    }
    if not proxy_losses:
        ratios = priors
        reason = "no_source_losses_available; using normalized priors"
    else:
        if reference_losses:
            signal = {
                name: proxy_losses[name] - reference_losses[name]
                for name in proxy_losses
                if name in reference_losses
            }
            reason = "doremi_excess_loss_adaptive_ratios"
        else:
            signal = proxy_losses
            reason = "proxy_loss_adaptive_ratios_no_reference"
        mean_loss = sum(signal.values()) / len(signal)
        variance = sum((value - mean_loss) ** 2 for value in signal.values()) / max(1, len(signal))
        std = math.sqrt(max(variance, 1e-12))
        raw = {}
        for name, prior in priors.items():
            if name not in signal:
                raw[name] = prior
                continue
            z = (signal[name] - mean_loss) / std
            # DoReMi uses a reference-normalized excess-loss signal. Domains
            # where the proxy is worse than the reference have larger remaining
            # headroom and get adversarially upweighted, while priors/clipping
            # keep the sampler from collapsing to one source.
            candidate = prior * math.exp(args.eta * z)
            raw[name] = (1.0 - args.smoothing) * candidate + args.smoothing * prior
        clipped = {name: min(args.max_ratio, max(args.min_ratio, value)) for name, value in raw.items()}
        ratios = normalize(clipped)

    out = {
        "method": "doremi_style_reference_excess_loss_mixing",
        "reason": reason,
        "recipe": str(args.recipe),
        "source_losses": str(args.source_losses),
        "reference_losses": str(args.reference_losses) if args.reference_losses is not None else "",
        "eta": args.eta,
        "smoothing": args.smoothing,
        "min_ratio": args.min_ratio,
        "max_ratio": args.max_ratio,
        "priors": priors,
        "proxy_losses": proxy_losses,
        "reference_losses_by_source": reference_losses,
        "excess_losses": {
            name: proxy_losses[name] - reference_losses[name]
            for name in proxy_losses
            if name in reference_losses
        },
        "ratios": ratios,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
