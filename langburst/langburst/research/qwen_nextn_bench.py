from __future__ import annotations

import argparse
import json
import traceback
import time
from pathlib import Path
from typing import Iterable

import torch

from ..adapters import ensure_adapters_loaded
from ..core.adapter import adapter_registry
from ..engines.native.policy import RuntimePolicyResolver
from ..engines.native.runtime import GenerationConfig, RuntimeEngine, sample_next_tensor
from ..adapters.qwen36_impl.model import Qwen36Model
from ..adapters.qwen36_mtp import QwenNativeMTP1, QwenNativeMTP1Proposer
from ..speculation import (
    DraftRequest,
    SpeculativeBenchmarkResult,
    SpeculativeDecodeResult,
    SpeculativeDecodeStats,
    SpeculativeProbeResult,
    TargetVerification,
)
from .speculative_verifier import NativeNextNVerifier, VerifierMode

QWEN_NEXTN_ADAPTER_ID = "qwen36"


def _qwen_nextn_adapter():
    ensure_adapters_loaded()
    return adapter_registry.get(QWEN_NEXTN_ADAPTER_ID)


def _parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _prompt_ids_for_context(engine: RuntimeEngine, *, context_tokens: int) -> list[int]:
    """Build a deterministic prompt close to `context_tokens` tokens.

    The benchmark is meant to stress prefill/state cost without depending on a
    large external dataset.  Keep the final instruction stable so greedy
    identity checks remain comparable across context lengths.
    """

    suffix = "\n\nQuestion: Summarize the document in two concise sentences.\nAnswer:"
    suffix_ids = engine.encode_prompt(suffix)
    if context_tokens <= len(suffix_ids):
        return suffix_ids[-context_tokens:]
    seed = (
        "LangBurst benchmark context. The document discusses CUDA kernels, "
        "paged KV cache, speculative decoding, native MTP verification, and "
        "stateful serving correctness. "
    )
    seed_ids = engine.encode_prompt(seed)
    if not seed_ids:
        seed_ids = [0]
    body_len = max(0, context_tokens - len(suffix_ids))
    repeats = (body_len + len(seed_ids) - 1) // len(seed_ids)
    body = (seed_ids * repeats)[:body_len]
    return body + suffix_ids


def _safe_cuda_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@torch.no_grad()
def _prefill_for_decode_bench(
    engine: RuntimeEngine,
    prompt_ids: list[int],
) -> tuple[object, torch.Tensor, torch.Tensor, float]:
    features = engine.resolve_plan(engine.features).effective
    state = engine.new_state(features)
    _sync_if_cuda()
    t0 = time.perf_counter()
    logits, raw_hidden = engine._prefill_with_raw_hidden(prompt_ids, state, features)
    _sync_if_cuda()
    return state, logits, raw_hidden, time.perf_counter() - t0


@torch.no_grad()
def _target_decode_from_prefill(
    engine: RuntimeEngine,
    *,
    base_state: object,
    logits: torch.Tensor,
    max_new_tokens: int,
) -> tuple[list[int], float]:
    state = base_state.fork() if hasattr(base_state, "fork") else base_state
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    _sync_if_cuda()
    t0 = time.perf_counter()
    out = engine._continue_ids_greedy_gpu_plain(logits, cfg, state)
    _sync_if_cuda()
    return out, time.perf_counter() - t0


@torch.no_grad()
def _research_verify_nextn_tokens(
    engine: RuntimeEngine,
    token_ids: Iterable[int],
    state: object,
    num_candidates: int,
) -> TargetVerification:
    token_list = [int(t) for t in token_ids]
    if not token_list:
        raise ValueError("verify batch requires at least one token")
    if num_candidates < 0 or num_candidates >= len(token_list):
        raise ValueError("num_candidates must be in [0, len(token_ids) - 1]")
    forward_block = getattr(engine.model, "forward_block")
    result = forward_block(
        token_list,
        state,
        return_logits=True,
        logits_mode="all",
        commit=True,
    )
    if not result.logits or len(result.logits) <= num_candidates:
        raise RuntimeError("research verifier did not return enough logits")
    check_logits = torch.stack([result.logits[i].contiguous() for i in range(num_candidates)], dim=0)
    target_ids = torch.argmax(check_logits, dim=-1).to(device=check_logits.device, dtype=torch.long)
    raw_hiddens = getattr(result, "final_hiddens", None) or getattr(result, "raw_hiddens", None)
    raw_hidden = raw_hiddens[-1] if raw_hiddens else result.logits[-1]
    return TargetVerification(target_ids=target_ids, logits=result.logits[-1], raw_hidden=raw_hidden)


@torch.no_grad()
def _generate_native_nextn_from_prefill(
    engine: RuntimeEngine,
    *,
    base_state: object,
    logits: torch.Tensor,
    raw_hidden: torch.Tensor,
    max_new_tokens: int,
    max_draft: int,
    verifier_mode: VerifierMode | None,
    min_speedup: float,
) -> tuple[SpeculativeDecodeResult, float]:
    if engine.speculative_proposer is None or engine.speculative_proposer.method != "native_mtp1":
        raise RuntimeError("native_mtp1 proposer unavailable")
    state = base_state.fork() if hasattr(base_state, "fork") else base_state
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    policy = RuntimePolicyResolver().speculative_policy(
        max_draft=max_draft,
        verifier_mode=verifier_mode,
        adaptive=False,
        min_speedup=min_speedup,
    )
    stats = SpeculativeDecodeStats(max_draft=policy.max_draft, verifier_mode=policy.verifier_mode)
    out_buf = torch.empty((max_new_tokens,), device=logits.device, dtype=torch.long)
    produced = 0
    live_logits = logits
    live_raw_hidden = raw_hidden
    verifier = NativeNextNVerifier(
        model=engine.model,
        proposer=engine.speculative_proposer,
        sample_next=lambda current_logits: sample_next_tensor(current_logits, cfg),
        max_draft=policy.max_draft,
        mode=policy.verifier_mode,  # type: ignore[arg-type]
        verify_tokens=lambda token_ids, verify_state, num_candidates: _research_verify_nextn_tokens(
            engine,
            token_ids,
            verify_state,
            num_candidates,
        ),
    )
    _sync_if_cuda()
    t0 = time.perf_counter()
    while produced < max_new_tokens:
        step = verifier.step(
            logits=live_logits,
            raw_hidden=live_raw_hidden,
            state=state,
            remaining_tokens=max_new_tokens - produced,
        )
        for token in step.tokens:
            if produced >= max_new_tokens:
                break
            out_buf[produced] = token
            produced += 1
        live_logits = step.logits
        live_raw_hidden = step.raw_hidden
        if step.verified:
            stats.verifier_steps += 1
            stats.accepted_draft_tokens += step.accepted
            stats.verified_draft_tokens += step.verified
            stats.rejected_steps += step.rejected
            stats.rollback_tokens += step.rollback_tokens
    _sync_if_cuda()
    return SpeculativeDecodeResult(out_buf[:produced].detach().cpu().tolist(), stats), time.perf_counter() - t0


def _result_row(
    *,
    context_tokens: int,
    batch_size: int,
    max_draft: int,
    verifier_mode: str,
    result: SpeculativeBenchmarkResult,
) -> dict[str, object]:
    return {
        "context_tokens": context_tokens,
        "batch_size": batch_size,
        "max_draft": max_draft,
        "verifier_mode": verifier_mode,
        "target_tok_s": round(result.target_tok_s, 2),
        "speculative_tok_s": round(result.speculative_tok_s, 2),
        "speedup": round(result.speedup, 3),
        "accept_rate": round(result.accept_rate, 3),
        "rollback_tokens": result.rollback_tokens,
        "fallback_reason": result.fallback_reason,
        "identical": result.identical,
        "keep": result.keep,
        "status": "ok",
    }


@torch.no_grad()
def benchmark_auto_nextn(
    *,
    hf_model: Path,
    qb_model: Path,
    context_values: Iterable[int],
    batch_values: Iterable[int],
    draft_values: Iterable[int],
    max_new_tokens: int,
    recent_window: int,
    verifier_mode: VerifierMode | None,
    min_speedup: float,
    output_json: Path | None = None,
    jsonl: Path | None = None,
) -> dict[str, object]:
    """Sweep native NEXTN policies and persist only speed-positive champions.

    This intentionally refuses to fake batch>1 speculative numbers.  The
    current exact NEXTN generation loop is single-request; multi-request MTP
    should be enabled only after it is wired through the same batch verifier
    contract as target-only serving.
    """

    engine = RuntimeEngine(
        adapter=_qwen_nextn_adapter(),
        hf_model=hf_model,
        qb_model=qb_model,
        device="cuda",
        recent_window=recent_window,
        weight_device="cuda",
    )
    rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    if jsonl is not None:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl_handle = jsonl.open("w") if jsonl is not None else None
    try:
        for context_tokens in context_values:
            prompt_ids: list[int] | None = None
            prefilled: tuple[object, torch.Tensor, torch.Tensor, float] | None = None
            baseline: tuple[list[int], float] | None = None
            for batch_size in batch_values:
                for max_draft in draft_values:
                    base_row = {
                        "context_tokens": int(context_tokens),
                        "batch_size": int(batch_size),
                        "max_draft": int(max_draft),
                        "verifier_mode": verifier_mode or RuntimePolicyResolver().verifier_mode(),
                    }
                    if batch_size != 1:
                        row = {
                            **base_row,
                            "status": "unsupported",
                            "unsupported_reason": "native_nextn_generation_benchmark_batch_gt1_not_wired",
                            "keep": False,
                        }
                        rows.append(row)
                        if jsonl_handle is not None:
                            print(json.dumps(row, ensure_ascii=False), file=jsonl_handle, flush=True)
                        print(json.dumps(row, ensure_ascii=False))
                        continue
                    try:
                        if prompt_ids is None:
                            prompt_ids = _prompt_ids_for_context(engine, context_tokens=int(context_tokens))
                        if prefilled is None:
                            prefilled = _prefill_for_decode_bench(engine, prompt_ids)
                        base_state, base_logits, base_raw_hidden, prefill_s = prefilled
                        if baseline is None:
                            baseline = _target_decode_from_prefill(
                                engine,
                                base_state=base_state,
                                logits=base_logits,
                                max_new_tokens=max_new_tokens,
                            )
                        target_ids, target_s = baseline
                        speculative, spec_s = _generate_native_nextn_from_prefill(
                            engine,
                            base_state=base_state,
                            logits=base_logits,
                            raw_hidden=base_raw_hidden,
                            max_new_tokens=max_new_tokens,
                            max_draft=int(max_draft),
                            verifier_mode=verifier_mode,
                            min_speedup=min_speedup,
                        )
                        result = benchmark_result_from_run(
                            method="native_nextn_auto",
                            tokens=max_new_tokens,
                            target_seconds=target_s,
                            speculative_seconds=spec_s,
                            target_ids=target_ids,
                            speculative=speculative,
                            min_speedup=min_speedup,
                        )
                        row = _result_row(
                            context_tokens=int(context_tokens),
                            batch_size=int(batch_size),
                            max_draft=int(max_draft),
                            verifier_mode=verifier_mode or RuntimePolicyResolver().verifier_mode(),
                            result=result,
                        )
                        row["prefill_s"] = round(prefill_s, 3)
                    except torch.cuda.OutOfMemoryError as exc:
                        _safe_cuda_reset()
                        row = {
                            **base_row,
                            "status": "oom",
                            "error": str(exc).splitlines()[0],
                            "keep": False,
                        }
                    except Exception as exc:  # pragma: no cover - real-model diagnostic path
                        _safe_cuda_reset()
                        row = {
                            **base_row,
                            "status": "error",
                            "error": str(exc).splitlines()[0],
                            "traceback": traceback.format_exc(limit=4),
                            "keep": False,
                        }
                    rows.append(row)
                    if jsonl_handle is not None:
                        print(json.dumps(row, ensure_ascii=False), file=jsonl_handle, flush=True)
                    print(json.dumps(row, ensure_ascii=False))
                    if row.get("keep") and (best is None or float(row.get("speedup", 0.0)) > float(best.get("speedup", 0.0))):
                        best = row
        champion: dict[str, object] = {
            "keep": best is not None,
            "policy": {},
            "best": best,
            "rows": rows,
        }
        if best is not None:
            champion["policy"] = {
                "max_draft": int(best["max_draft"]),
                "verifier_mode": str(best["verifier_mode"]),
                "adaptive": True,
                "min_verified": 1,
                "accept_threshold": 1.0,
                "max_rejections": None,
                "min_speedup": float(min_speedup),
            }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(champion, ensure_ascii=False, indent=2) + "\n")
        print("champion=" + json.dumps(champion, ensure_ascii=False))
        return champion
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()


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
    engine = RuntimeEngine(
        adapter=_qwen_nextn_adapter(),
        hf_model=hf_model,
        qb_model=qb_model,
        device="cuda",
        recent_window=recent_window,
        weight_device="cuda",
    )
    if not isinstance(engine.model, Qwen36Model):
        raise TypeError("native_mtp1 currently supports Qwen36Model only")
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
def generate_native_nextn_timed(
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
    min_speedup: float | None = None,
) -> tuple[SpeculativeDecodeResult, float]:
    if not isinstance(engine.model, Qwen36Model):
        raise TypeError("native_mtp1 currently supports Qwen36Model only")
    if adaptive:
        raise ValueError("research verifier timing does not implement adaptive fallback; use serving batch metrics")
    features = engine.features.with_overrides(speculative_decoding=True)
    state = engine.new_state(features)
    logits, raw_hidden = engine._prefill_with_raw_hidden([int(t) for t in prompt_ids], state, features)
    return _generate_native_nextn_from_prefill(
        engine,
        base_state=state,
        logits=logits,
        raw_hidden=raw_hidden,
        max_new_tokens=max_new_tokens,
        max_draft=max_draft,
        verifier_mode=verifier_mode,
        min_speedup=min_speedup or RuntimePolicyResolver().speculative_policy().min_speedup,
    )


def benchmark_result_from_run(
    *,
    method: str,
    tokens: int,
    target_seconds: float,
    speculative_seconds: float,
    target_ids: list[int],
    speculative: SpeculativeDecodeResult,
    min_speedup: float = 1.03,
) -> SpeculativeBenchmarkResult:
    return SpeculativeBenchmarkResult(
        method=method,
        tokens=tokens,
        target_seconds=target_seconds,
        speculative_seconds=speculative_seconds,
        accepted_second_tokens=speculative.stats.accepted_draft_tokens,
        verified_steps=speculative.stats.verified_draft_tokens,
        identical=target_ids == speculative.ids[:tokens],
        rollback_tokens=speculative.stats.rollback_tokens,
        fallback_reason=speculative.stats.fallback_reason,
        min_speedup=min_speedup,
    )


@torch.no_grad()
def benchmark_native_mtp1_with_engine(
    engine: RuntimeEngine,
    *,
    prompt: str,
    max_new_tokens: int,
    max_draft: int = 1,
    verifier_mode: VerifierMode | None = None,
) -> SpeculativeBenchmarkResult:
    if not isinstance(engine.model, Qwen36Model):
        raise TypeError("native_mtp1 currently supports Qwen36Model only")
    prompt_ids = engine.encode_prompt(prompt)
    target_ids, target_s = _target_only_ids(engine, prompt_ids, max_new_tokens=max_new_tokens)
    speculative, spec_s = generate_native_nextn_timed(
        engine,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        max_draft=max_draft,
        verifier_mode=verifier_mode,
        adaptive=False,
    )
    return benchmark_result_from_run(
        method="native_mtp1",
        tokens=max_new_tokens,
        target_seconds=target_s,
        speculative_seconds=spec_s,
        target_ids=target_ids,
        speculative=speculative,
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
        adapter=_qwen_nextn_adapter(),
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
    ap = argparse.ArgumentParser(description="LangBurst speculative decoding probes")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mtp = sub.add_parser("native-mtp1", help="probe Qwen3.6 native MTP1 acceptance")
    mtp.add_argument("--hf-model", type=Path, required=True)
    mtp.add_argument("--qb-model", type=Path, required=True)
    mtp.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    mtp.add_argument("--steps", type=int, default=16)
    mtp.add_argument("--min-accept-rate", type=float, default=0.55)
    mtp.add_argument("--recent-window", type=int, default=256)
    bench = sub.add_parser("bench-mtp1", help="benchmark exact native MTP1 verifier against target-only")
    bench.add_argument("--hf-model", type=Path, required=True)
    bench.add_argument("--qb-model", type=Path, required=True)
    bench.add_argument("--prompt", default="Explain why the sky is blue in one paragraph.")
    bench.add_argument("--max-new-tokens", type=int, default=64)
    bench.add_argument("--max-draft", type=int, default=1)
    verifier_choices = ("sequential", "transaction_block")
    bench.add_argument("--verifier-mode", choices=verifier_choices, default=RuntimePolicyResolver().verifier_mode())
    bench.add_argument("--recent-window", type=int, default=256)
    bench.add_argument("--adaptive", action="store_true")
    bench.add_argument("--min-verified", type=int, default=1)
    bench.add_argument("--accept-threshold", type=float, default=1.00)
    bench.add_argument("--max-rejections", type=int, default=None)
    suite = sub.add_parser("bench-suite-mtp1", help="benchmark MTP1 over several prompts with one model load")
    suite.add_argument("--hf-model", type=Path, required=True)
    suite.add_argument("--qb-model", type=Path, required=True)
    suite.add_argument("--recent-window", type=int, default=64)
    suite.add_argument("--max-new-tokens", type=int, default=32)
    suite.add_argument("--max-draft", type=int, default=1)
    suite.add_argument("--verifier-mode", choices=verifier_choices, default=RuntimePolicyResolver().verifier_mode())
    suite.add_argument("--adaptive", action="store_true")
    suite.add_argument("--min-verified", type=int, default=1)
    suite.add_argument("--accept-threshold", type=float, default=1.00)
    suite.add_argument("--max-rejections", type=int, default=None)
    sweep = sub.add_parser("bench-policy-sweep-mtp1", help="sweep adaptive MTP1 policy with one model load")
    sweep.add_argument("--hf-model", type=Path, required=True)
    sweep.add_argument("--qb-model", type=Path, required=True)
    sweep.add_argument("--recent-window", type=int, default=64)
    sweep.add_argument("--max-new-tokens", type=int, default=128)
    sweep.add_argument("--max-draft-values", default="1,2,3,4")
    sweep.add_argument("--verifier-modes", default="sequential,transaction_block")
    sweep.add_argument("--min-verified-values", default="1,2,4,8")
    sweep.add_argument("--accept-threshold-values", default="0.70,0.75,0.80,0.85,0.90")
    sweep.add_argument("--max-rejections-values", default="none,1,2")
    auto = sub.add_parser("bench-auto-nextn", help="sweep NEXTN settings and write a speed-positive auto-adopt policy")
    auto.add_argument("--hf-model", type=Path, required=True)
    auto.add_argument("--qb-model", type=Path, required=True)
    auto.add_argument("--recent-window", type=int, default=256)
    auto.add_argument("--max-new-tokens", type=int, default=64)
    auto.add_argument("--draft-values", default="4,6,8,10")
    auto.add_argument("--context-values", default="65536,100000")
    auto.add_argument("--batch-values", default="1,2")
    auto.add_argument("--verifier-mode", choices=verifier_choices, default=RuntimePolicyResolver().verifier_mode())
    auto.add_argument("--min-speedup", type=float, default=1.03)
    auto.add_argument("--output-json", type=Path, default=Path("runs/langburst_nextn_autotune.json"))
    auto.add_argument("--jsonl", type=Path, default=Path("runs/langburst_nextn_autotune.jsonl"))
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
                adapter=_qwen_nextn_adapter(),
                hf_model=args.hf_model,
                qb_model=args.qb_model,
                device="cuda",
                recent_window=args.recent_window,
                weight_device="cuda",
            )
            prompt_ids = engine.encode_prompt(args.prompt)
            target_ids, target_s = _target_only_ids(engine, prompt_ids, max_new_tokens=args.max_new_tokens)
            speculative, spec_s = generate_native_nextn_timed(
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
            result = benchmark_result_from_run(
                method="native_mtp1_adaptive",
                tokens=args.max_new_tokens,
                target_seconds=target_s,
                speculative_seconds=spec_s,
                target_ids=target_ids,
                speculative=speculative,
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
        print(f"rollback_tokens={result.rollback_tokens}")
        print(f"fallback_reason={result.fallback_reason or 'none'}")
        print(f"identical={str(result.identical).lower()}")
        print(f"keep={str(result.keep).lower()}")
    elif args.cmd == "bench-suite-mtp1":
        prompts = [
            ("sky", "Explain why the sky is blue in one paragraph.", args.max_new_tokens),
            ("math", "Solve: if a train travels 120 km in 2 hours, what is its speed? Answer briefly.", args.max_new_tokens),
            ("technical", "Write a concise technical note about quantized LLM inference.", max(args.max_new_tokens, 64)),
        ]
        engine = RuntimeEngine(
            adapter=_qwen_nextn_adapter(),
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
                speculative, spec_s = generate_native_nextn_timed(
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
                result = benchmark_result_from_run(
                    method="native_mtp1_adaptive",
                    tokens=tokens,
                    target_seconds=target_s,
                    speculative_seconds=spec_s,
                    target_ids=target_ids,
                    speculative=speculative,
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
                "rollback_tokens": result.rollback_tokens,
                "fallback_reason": result.fallback_reason,
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
            adapter=_qwen_nextn_adapter(),
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
                                speculative, spec_s = generate_native_nextn_timed(
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
                                result = benchmark_result_from_run(
                                    method="native_mtp1_adaptive",
                                    tokens=tokens,
                                    target_seconds=target_s,
                                    speculative_seconds=spec_s,
                                    target_ids=target_ids,
                                    speculative=speculative,
                                )
                                rows.append({
                                    "name": name,
                                    "speedup": result.speedup,
                                    "accept_rate": result.accept_rate,
                                    "rollback_tokens": result.rollback_tokens,
                                    "fallback_reason": result.fallback_reason,
                                    "identical": result.identical,
                                    "keep": result.keep,
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
                                "keep": all(bool(r["keep"]) for r in rows),
                                "rollback_tokens": sum(int(r["rollback_tokens"]) for r in rows),
                                "rows": [
                                    {
                                        "name": r["name"],
                                        "speedup": round(float(r["speedup"]), 3),
                                        "accept_rate": round(float(r["accept_rate"]), 3),
                                        "rollback_tokens": r["rollback_tokens"],
                                        "fallback_reason": r["fallback_reason"],
                                        "identical": r["identical"],
                                        "keep": r["keep"],
                                    }
                                    for r in rows
                                ],
                            }
                            print(json.dumps(candidate, ensure_ascii=False))
                            if candidate["keep"] and (best is None or avg_speedup > float(best["avg_speedup"])):
                                best = candidate
        print("best=" + json.dumps(best, ensure_ascii=False))
    elif args.cmd == "bench-auto-nextn":
        benchmark_auto_nextn(
            hf_model=args.hf_model,
            qb_model=args.qb_model,
            context_values=_parse_int_list(args.context_values),
            batch_values=_parse_int_list(args.batch_values),
            draft_values=_parse_int_list(args.draft_values),
            max_new_tokens=args.max_new_tokens,
            recent_window=args.recent_window,
            verifier_mode=args.verifier_mode,
            min_speedup=args.min_speedup,
            output_json=args.output_json,
            jsonl=args.jsonl,
        )


if __name__ == "__main__":
    main()
