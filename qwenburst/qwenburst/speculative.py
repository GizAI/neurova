from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .adapters import Qwen36Adapter
from .core.adapter import adapter_registry
from .core.runtime import GenerationConfig, RuntimeEngine, sample_next_tensor
from .model import QwenBurstModel
from .qwen_mtp import QwenNativeMTP1, QwenNativeMTP1Proposer
from .speculation import DraftRequest, SpeculativeBenchmarkResult, SpeculativeProbeResult
from .speculative_verifier import NativeNextNVerifier, VerifierMode, tensor_ids
from .tuning import speculative_verifier_mode


@torch.no_grad()
def probe_native_mtp1(
    *,
    hf_model: Path,
    qb_model: Path,
    prompt: str,
    steps: int,
    min_accept_rate: float = 0.55,
    recent_window: int = 256,
) -> SpeculativeProbeResult:
    adapter = adapter_registry.get("qwen36")
    engine = RuntimeEngine(
        adapter=adapter,
        hf_model=hf_model,
        qb_model=qb_model,
        device="cuda",
        recent_window=recent_window,
        weight_device="cuda",
    )
    if not isinstance(engine.model, QwenBurstModel):
        raise TypeError("native_mtp1 currently supports QwenBurstModel only")
    proposer = QwenNativeMTP1Proposer(QwenNativeMTP1(engine.model, engine.model.store))
    state = engine.new_state()
    prompt_ids = engine.encode_prompt(prompt)
    logits: torch.Tensor | None = None
    raw_hidden: torch.Tensor | None = None
    for i, tid in enumerate(prompt_ids):
        if i == len(prompt_ids) - 1:
            logits, raw_hidden = engine.model.forward_one(tid, state, return_hidden=True, return_raw_hidden=False)
        else:
            engine.model.forward_one(tid, state, return_logits=False)
    assert logits is not None and raw_hidden is not None
    gen_cfg = GenerationConfig(max_new_tokens=steps)
    accepted = 0
    total = 0
    for _ in range(steps):
        first = sample_next_tensor(logits, gen_cfg)
        first_pos = int(state.pos)
        candidate_second = proposer.propose_tensor(
            DraftRequest(
                history=[],
                max_draft=1,
                signals={"raw_hidden": raw_hidden, "first_token": first, "pos": first_pos},
            )
        )
        logits, raw_hidden = engine.model.forward_one(first, state, return_hidden=True, return_raw_hidden=False)
        target_second = sample_next_tensor(logits, gen_cfg)
        accepted += int(torch.equal(candidate_second.reshape(()), target_second.reshape(())))
        total += 1
    return SpeculativeProbeResult("native_mtp1", total=total, accepted=accepted, min_accept_rate=min_accept_rate)


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _target_only_ids(engine: RuntimeEngine, prompt_ids: list[int], *, max_new_tokens: int) -> tuple[list[int], float]:
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    features = engine.features.with_overrides(speculative_decoding=False)
    _sync_if_cuda()
    t0 = time.perf_counter()
    out = engine.generate_ids_greedy_gpu(prompt_ids, cfg, features=features)
    _sync_if_cuda()
    return out, time.perf_counter() - t0


@torch.no_grad()
def generate_native_mtp1_ids(
    engine: RuntimeEngine,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    max_draft: int = 1,
    verifier_mode: VerifierMode | None = None,
    adaptive: bool = False,
    min_verified: int = 1,
    accept_threshold: float = 1.00,
    max_rejections: int | None = None,
) -> tuple[list[int], float, int, int]:
    if not isinstance(engine.model, QwenBurstModel):
        raise TypeError("native_mtp1 currently supports QwenBurstModel only")
    proposer = QwenNativeMTP1Proposer(QwenNativeMTP1(engine.model, engine.model.store))
    state = engine.new_state()
    _sync_if_cuda()
    t0 = time.perf_counter()
    features = engine.resolve_plan(engine.features).effective
    logits, raw_hidden = engine._prefill_with_raw_hidden(prompt_ids, state, features)
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    out: list[int] = []
    recent_accepts: list[int] = []
    verifier = NativeNextNVerifier(
        model=engine.model,
        proposer=proposer,
        sample_next=lambda current_logits: sample_next_tensor(current_logits, cfg),
        max_draft=max_draft,
        mode=(verifier_mode or speculative_verifier_mode()),  # type: ignore[arg-type]
    )
    while len(out) < max_new_tokens:
        step = verifier.step(
            logits=logits,
            raw_hidden=raw_hidden,
            state=state,
            remaining_tokens=max_new_tokens - len(out),
        )
        out.extend(tensor_ids(step.tokens))
        logits = step.logits
        raw_hidden = step.raw_hidden
        if step.verified:
            recent_accepts.extend([1] * step.accepted)
            if step.rejected:
                recent_accepts.append(0)
        if len(recent_accepts) > min_verified:
            del recent_accepts[: len(recent_accepts) - min_verified]
        if adaptive and max_rejections is not None and verifier.rejected >= max_rejections:
            while len(out) < max_new_tokens:
                next_token = sample_next_tensor(logits, cfg)
                next_id = int(next_token.detach().cpu().item())
                out.append(next_id)
                if len(out) >= max_new_tokens:
                    break
                logits = engine.model.forward_one(next_id, state, return_logits=True)
            break
        recent_ready = len(recent_accepts) >= min_verified
        recent_rate = sum(recent_accepts) / len(recent_accepts) if recent_accepts else 0.0
        if adaptive and recent_ready and recent_rate < accept_threshold:
            while len(out) < max_new_tokens:
                next_token = sample_next_tensor(logits, cfg)
                next_id = int(next_token.detach().cpu().item())
                out.append(next_id)
                if len(out) >= max_new_tokens:
                    break
                logits = engine.model.forward_one(next_id, state, return_logits=True)
            break
    _sync_if_cuda()
    return out[:max_new_tokens], time.perf_counter() - t0, verifier.accepted, verifier.verified


@torch.no_grad()
def benchmark_native_mtp1_with_engine(
    engine: RuntimeEngine,
    *,
    prompt: str,
    max_new_tokens: int,
    max_draft: int = 1,
    verifier_mode: VerifierMode | None = None,
) -> SpeculativeBenchmarkResult:
    if not isinstance(engine.model, QwenBurstModel):
        raise TypeError("native_mtp1 currently supports QwenBurstModel only")
    prompt_ids = engine.encode_prompt(prompt)
    target_ids, target_s = _target_only_ids(engine, prompt_ids, max_new_tokens=max_new_tokens)
    speculative_ids, spec_s, accepted, verified = generate_native_mtp1_ids(
        engine,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        max_draft=max_draft,
        verifier_mode=verifier_mode,
        adaptive=False,
    )
    return SpeculativeBenchmarkResult(
        method="native_mtp1",
        tokens=max_new_tokens,
        target_seconds=target_s,
        speculative_seconds=spec_s,
        accepted_second_tokens=accepted,
        verified_steps=verified,
        identical=target_ids == speculative_ids,
    )


@torch.no_grad()
def benchmark_native_mtp1(
    *,
    hf_model: Path,
    qb_model: Path,
    prompt: str,
    max_new_tokens: int,
    max_draft: int = 1,
    verifier_mode: VerifierMode | None = None,
    recent_window: int = 256,
) -> SpeculativeBenchmarkResult:
    engine = RuntimeEngine(
        adapter=adapter_registry.get("qwen36"),
        hf_model=hf_model,
        qb_model=qb_model,
        device="cuda",
        recent_window=recent_window,
        weight_device="cuda",
    )
    return benchmark_native_mtp1_with_engine(
        engine,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        max_draft=max_draft,
        verifier_mode=verifier_mode,
    )


def main() -> None:
    # Importing adapters registers qwen36. Keep it explicit for command-line use.
    _ = Qwen36Adapter
    ap = argparse.ArgumentParser(description="QwenBurst speculative decoding probes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mtp = sub.add_parser("native-mtp1", help="probe Qwen3.6 native MTP1 acceptance")
    mtp.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    mtp.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    mtp.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    mtp.add_argument("--steps", type=int, default=16)
    mtp.add_argument("--min-accept-rate", type=float, default=0.55)
    mtp.add_argument("--recent-window", type=int, default=256)
    bench = sub.add_parser("bench-mtp1", help="benchmark exact native MTP1 verifier against target-only")
    bench.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    bench.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    bench.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    bench.add_argument("--max-new-tokens", type=int, default=64)
    bench.add_argument("--max-draft", type=int, default=1)
    verifier_choices = ("sequential", "transaction_block")
    bench.add_argument("--verifier-mode", choices=verifier_choices, default=speculative_verifier_mode())
    bench.add_argument("--recent-window", type=int, default=256)
    bench.add_argument("--adaptive", action="store_true")
    bench.add_argument("--min-verified", type=int, default=1)
    bench.add_argument("--accept-threshold", type=float, default=1.00)
    bench.add_argument("--max-rejections", type=int, default=None)
    suite = sub.add_parser("bench-suite-mtp1", help="benchmark MTP1 over several prompts with one model load")
    suite.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    suite.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    suite.add_argument("--recent-window", type=int, default=64)
    suite.add_argument("--max-new-tokens", type=int, default=32)
    suite.add_argument("--max-draft", type=int, default=1)
    suite.add_argument("--verifier-mode", choices=verifier_choices, default=speculative_verifier_mode())
    suite.add_argument("--adaptive", action="store_true")
    suite.add_argument("--min-verified", type=int, default=1)
    suite.add_argument("--accept-threshold", type=float, default=1.00)
    suite.add_argument("--max-rejections", type=int, default=None)
    sweep = sub.add_parser("bench-policy-sweep-mtp1", help="sweep adaptive MTP1 policy with one model load")
    sweep.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    sweep.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    sweep.add_argument("--recent-window", type=int, default=64)
    sweep.add_argument("--max-new-tokens", type=int, default=128)
    sweep.add_argument("--max-draft-values", default="1,2,4")
    sweep.add_argument("--verifier-modes", default="sequential,transaction_block")
    sweep.add_argument("--min-verified-values", default="1,2,4,8")
    sweep.add_argument("--accept-threshold-values", default="0.70,0.75,0.80,0.85,0.90")
    sweep.add_argument("--max-rejections-values", default="none,1,2")
    args = ap.parse_args()
    if args.cmd == "native-mtp1":
        result = probe_native_mtp1(
            hf_model=args.hf_model,
            qb_model=args.qb_model,
            prompt=args.prompt,
            steps=args.steps,
            min_accept_rate=args.min_accept_rate,
            recent_window=args.recent_window,
        )
        print(result)
        print(f"accept_rate={result.accept_rate:.3f}")
        print(f"viable={str(result.viable).lower()}")
    elif args.cmd == "bench-mtp1":
        if args.adaptive:
            engine = RuntimeEngine(
                adapter=adapter_registry.get("qwen36"),
                hf_model=args.hf_model,
                qb_model=args.qb_model,
                device="cuda",
                recent_window=args.recent_window,
                weight_device="cuda",
            )
            prompt_ids = engine.encode_prompt(args.prompt)
            target_ids, target_s = _target_only_ids(engine, prompt_ids, max_new_tokens=args.max_new_tokens)
            spec_ids, spec_s, accepted, verified = generate_native_mtp1_ids(
                engine,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                max_draft=args.max_draft,
                verifier_mode=args.verifier_mode,
                adaptive=True,
                min_verified=args.min_verified,
                accept_threshold=args.accept_threshold,
                max_rejections=args.max_rejections,
            )
            result = SpeculativeBenchmarkResult(
                method="native_mtp1_adaptive",
                tokens=args.max_new_tokens,
                target_seconds=target_s,
                speculative_seconds=spec_s,
                accepted_second_tokens=accepted,
                verified_steps=verified,
                identical=target_ids == spec_ids,
            )
        else:
            result = benchmark_native_mtp1(
                hf_model=args.hf_model,
                qb_model=args.qb_model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                max_draft=args.max_draft,
                verifier_mode=args.verifier_mode,
                recent_window=args.recent_window,
            )
        print(result)
        print(f"target_tok_s={result.target_tok_s:.2f}")
        print(f"speculative_tok_s={result.speculative_tok_s:.2f}")
        print(f"speedup={result.speedup:.3f}")
        print(f"accept_rate={result.accept_rate:.3f}")
        print(f"identical={str(result.identical).lower()}")
        print(f"keep={str(result.keep).lower()}")
    elif args.cmd == "bench-suite-mtp1":
        prompts = [
            ("sky", "Explain why the sky is blue in one paragraph.", args.max_new_tokens),
            ("math", "Solve: if a train travels 120 km in 2 hours, what is its speed? Answer briefly.", args.max_new_tokens),
            ("technical", "Write a concise technical note about quantized LLM inference.", max(args.max_new_tokens, 64)),
        ]
        engine = RuntimeEngine(
            adapter=adapter_registry.get("qwen36"),
            hf_model=args.hf_model,
            qb_model=args.qb_model,
            device="cuda",
            recent_window=args.recent_window,
            weight_device="cuda",
        )
        rows = []
        for name, prompt, tokens in prompts:
            if args.adaptive:
                prompt_ids = engine.encode_prompt(prompt)
                target_ids, target_s = _target_only_ids(engine, prompt_ids, max_new_tokens=tokens)
                spec_ids, spec_s, accepted, verified = generate_native_mtp1_ids(
                    engine,
                    prompt_ids,
                    max_new_tokens=tokens,
                    max_draft=args.max_draft,
                    verifier_mode=args.verifier_mode,
                    adaptive=True,
                    min_verified=args.min_verified,
                    accept_threshold=args.accept_threshold,
                    max_rejections=args.max_rejections,
                )
                result = SpeculativeBenchmarkResult(
                    method="native_mtp1_adaptive",
                    tokens=tokens,
                    target_seconds=target_s,
                    speculative_seconds=spec_s,
                    accepted_second_tokens=accepted,
                    verified_steps=verified,
                    identical=target_ids == spec_ids,
                )
            else:
                result = benchmark_native_mtp1_with_engine(
                    engine,
                    prompt=prompt,
                    max_new_tokens=tokens,
                    max_draft=args.max_draft,
                    verifier_mode=args.verifier_mode,
                )
            row = {
                "name": name,
                "tokens": result.tokens,
                "max_draft": args.max_draft,
                "verifier_mode": args.verifier_mode,
                "target_tok_s": round(result.target_tok_s, 2),
                "speculative_tok_s": round(result.speculative_tok_s, 2),
                "speedup": round(result.speedup, 3),
                "accept_rate": round(result.accept_rate, 3),
                "identical": result.identical,
                "keep": result.keep,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
        keep_rows = [r for r in rows if r["keep"]]
        print("summary=" + json.dumps({
            "keep_count": len(keep_rows),
            "total": len(rows),
            "all_identical": all(r["identical"] for r in rows),
            "avg_speedup": round(sum(float(r["speedup"]) for r in rows) / len(rows), 3),
        }, ensure_ascii=False))
    elif args.cmd == "bench-policy-sweep-mtp1":
        prompts = [
            ("sky", "Explain why the sky is blue in one paragraph.", args.max_new_tokens),
            ("math", "Solve: if a train travels 120 km in 2 hours, what is its speed? Answer briefly.", args.max_new_tokens),
            ("technical", "Write a concise technical note about quantized LLM inference.", max(args.max_new_tokens, 64)),
        ]
        min_verified_values = [int(x) for x in args.min_verified_values.split(",") if x.strip()]
        max_draft_values = [int(x) for x in args.max_draft_values.split(",") if x.strip()]
        verifier_modes = [x.strip() for x in args.verifier_modes.split(",") if x.strip()]
        accept_threshold_values = [float(x) for x in args.accept_threshold_values.split(",") if x.strip()]
        max_rejections_values = [
            None if x.strip().lower() in {"none", "null", "-1"} else int(x)
            for x in args.max_rejections_values.split(",")
            if x.strip()
        ]
        engine = RuntimeEngine(
            adapter=adapter_registry.get("qwen36"),
            hf_model=args.hf_model,
            qb_model=args.qb_model,
            device="cuda",
            recent_window=args.recent_window,
            weight_device="cuda",
        )
        encoded = [(name, engine.encode_prompt(prompt), tokens) for name, prompt, tokens in prompts]
        baselines: dict[str, tuple[list[int], float]] = {}
        for name, prompt_ids, tokens in encoded:
            baselines[name] = _target_only_ids(engine, prompt_ids, max_new_tokens=tokens)
        best: dict[str, object] | None = None
        for verifier_mode in verifier_modes:
            for max_draft in max_draft_values:
                for min_verified in min_verified_values:
                    for accept_threshold in accept_threshold_values:
                        for max_rejections in max_rejections_values:
                            rows = []
                            for name, prompt_ids, tokens in encoded:
                                target_ids, target_s = baselines[name]
                                spec_ids, spec_s, accepted, verified = generate_native_mtp1_ids(
                                    engine,
                                    prompt_ids,
                                    max_new_tokens=tokens,
                                    max_draft=max_draft,
                                    verifier_mode=verifier_mode,  # type: ignore[arg-type]
                                    adaptive=True,
                                    min_verified=min_verified,
                                    accept_threshold=accept_threshold,
                                    max_rejections=max_rejections,
                                )
                                result = SpeculativeBenchmarkResult(
                                    method="native_mtp1_adaptive",
                                    tokens=tokens,
                                    target_seconds=target_s,
                                    speculative_seconds=spec_s,
                                    accepted_second_tokens=accepted,
                                    verified_steps=verified,
                                    identical=target_ids == spec_ids,
                                )
                                rows.append({
                                    "name": name,
                                    "speedup": result.speedup,
                                    "accept_rate": result.accept_rate,
                                    "identical": result.identical,
                                })
                            avg_speedup = sum(float(r["speedup"]) for r in rows) / len(rows)
                            all_identical = all(bool(r["identical"]) for r in rows)
                            candidate = {
                                "verifier_mode": verifier_mode,
                                "max_draft": max_draft,
                                "min_verified": min_verified,
                                "accept_threshold": accept_threshold,
                                "max_rejections": max_rejections,
                                "avg_speedup": round(avg_speedup, 3),
                                "all_identical": all_identical,
                                "rows": [
                                    {
                                        "name": r["name"],
                                        "speedup": round(float(r["speedup"]), 3),
                                        "accept_rate": round(float(r["accept_rate"]), 3),
                                        "identical": r["identical"],
                                    }
                                    for r in rows
                                ],
                            }
                            print(json.dumps(candidate, ensure_ascii=False))
                            if all_identical and (best is None or avg_speedup > float(best["avg_speedup"])):
                                best = candidate
        print("best=" + json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
