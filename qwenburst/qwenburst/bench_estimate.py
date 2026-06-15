from __future__ import annotations

import argparse
from dataclasses import dataclass

from .config import Qwen36_27B_TextConfig


@dataclass
class Estimate:
    weight_gib: float
    raw_forward_per_s: float
    efficient_forward_per_s_low: float
    efficient_forward_per_s_high: float
    output_tok_s_low: float
    output_tok_s_high: float


def estimate(
    bpw: float = 3.08,
    params_b: float = 27.0,
    bandwidth_gbs: float = 716.8,
    overhead_gib: float = 2.0,
    bandwidth_eff_low: float = 0.62,
    bandwidth_eff_high: float = 0.82,
    mtp_accept_low: float = 1.7,
    mtp_accept_high: float = 2.6,
) -> Estimate:
    weight_gib = params_b * 1e9 * bpw / 8 / 1024**3 + overhead_gib
    raw = bandwidth_gbs / (weight_gib * 1024**3 / 1e9)
    f_low = raw * bandwidth_eff_low
    f_high = raw * bandwidth_eff_high
    return Estimate(
        weight_gib=weight_gib,
        raw_forward_per_s=raw,
        efficient_forward_per_s_low=f_low,
        efficient_forward_per_s_high=f_high,
        output_tok_s_low=f_low * mtp_accept_low,
        output_tok_s_high=f_high * mtp_accept_high,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpw", type=float, default=3.08)
    ap.add_argument("--params-b", type=float, default=27.0)
    ap.add_argument("--bandwidth-gbs", type=float, default=716.8)
    ap.add_argument("--overhead-gib", type=float, default=2.0)
    ap.add_argument("--mtp-low", type=float, default=1.7)
    ap.add_argument("--mtp-high", type=float, default=2.6)
    args = ap.parse_args()
    cfg = Qwen36_27B_TextConfig()
    est = estimate(
        bpw=args.bpw,
        params_b=args.params_b,
        bandwidth_gbs=args.bandwidth_gbs,
        overhead_gib=args.overhead_gib,
        mtp_accept_low=args.mtp_low,
        mtp_accept_high=args.mtp_high,
    )
    print(f"Qwen3.6-27B text config: {len(cfg.gdn_layers)} GDN layers, {len(cfg.attention_layers)} attention layers")
    print(f"GDN recurrent state fp16: {cfg.gdn_state_mib_fp16:.2f} MiB")
    print(f"Estimated resident weight+overhead: {est.weight_gib:.2f} GiB")
    print(f"Theoretical 1-forward upper bound: {est.raw_forward_per_s:.1f} f/s")
    print(f"Practical target forward: {est.efficient_forward_per_s_low:.1f}..{est.efficient_forward_per_s_high:.1f} f/s")
    print(f"MTP output speed target: {est.output_tok_s_low:.1f}..{est.output_tok_s_high:.1f} tok/s")


if __name__ == "__main__":
    main()
