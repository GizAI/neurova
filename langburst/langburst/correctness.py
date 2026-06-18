from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .cli_features import (
    add_adapter_arg,
    add_model_path_args,
    add_runtime_feature_args,
    create_runtime_engine_from_args,
    runtime_features_from_args,
)
from .core.defaults import serving_recent_window_default
from .core.features import RuntimeFeatures
from .engines.native import BatchedModelRunner, ContinuousBatchScheduler, GenerationConfig, RuntimeEngine, sample_next


@dataclass(frozen=True)
class PathParityResult:
    input_tokens: int
    chunk_size: int
    token_loop_argmax: int
    block_prefill_argmax: int
    argmax_match: bool
    max_abs_logit_diff: float
    mean_abs_logit_diff: float
    pos_match: bool
    kv_len_match: bool
    gdn_state_max_abs_diff: float
    conv_state_max_abs_diff: float
    attention_kv_max_abs_diff: float
    continuation_tokens: int
    continuation_argmax_match: bool
    continuation_max_abs_logit_diff: float
    continuation_mean_abs_logit_diff: float


@dataclass(frozen=True)
class RecallCaseResult:
    name: str
    input_tokens: int
    expected: str
    predicted: str
    passed: bool
    scores: dict[str, float]


@dataclass(frozen=True)
class BatchPathParityResult:
    prompt_tokens: int
    max_new_tokens: int
    direct_target_only: list[int]
    batch_target_only: list[int]
    batch_speculative: list[int]
    batch_prefix_second_hit: list[int]
    target_only_match: bool
    speculative_match: bool
    prefix_cache_match: bool
    prefix_cache_hit_tokens: int


@dataclass(frozen=True)
class CorrectnessReport:
    path_parity: PathParityResult
    batch_path_parity: BatchPathParityResult
    recall: list[RecallCaseResult]
    require_block_prefill_parity: bool = False

    @property
    def ok(self) -> bool:
        block_ok = (
            self.path_parity.argmax_match
            and self.path_parity.pos_match
            and self.path_parity.kv_len_match
            and self.path_parity.gdn_state_max_abs_diff == 0.0
            and self.path_parity.conv_state_max_abs_diff == 0.0
            and self.path_parity.attention_kv_max_abs_diff == 0.0
            and self.path_parity.continuation_argmax_match
            if self.require_block_prefill_parity
            else True
        )
        batch_ok = (
            self.batch_path_parity.target_only_match
            and self.batch_path_parity.speculative_match
            and self.batch_path_parity.prefix_cache_match
        )
        return block_ok and batch_ok and all(case.passed for case in self.recall)


def _filler(repeats: int) -> str:
    sentence = (
        "This filler sentence is deliberately unrelated to the protected secret. "
        "It discusses serving latency, CUDA kernels, cache layout, and benchmark hygiene. "
    )
    return sentence * max(0, repeats)


def recall_prompt(secret: str, *, filler_repeats: int) -> str:
    return (
        "Read the document and answer the final question with only the exact secret code.\n\n"
        f"Document start. The protected secret code is {secret}. "
        + _filler(filler_repeats)
        + "\n\nFinal question: What is the protected secret code? Answer only the code."
    )


def _encode_answer(engine: RuntimeEngine, answer: str) -> list[int]:
    ids = engine.tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError("candidate answer encoded to zero tokens")
    return [int(t) for t in ids]


def _max_abs_dict_diff(a: dict[int, torch.Tensor], b: dict[int, torch.Tensor]) -> float:
    if set(a) != set(b):
        return float("inf")
    max_diff = 0.0
    for key in a:
        diff = (a[key].float() - b[key].float()).abs()
        max_diff = max(max_diff, float(diff.max().detach().cpu()))
    return max_diff


def _attention_kv_max_abs_diff(a, b) -> float:
    if not hasattr(a, "attn_k") or not hasattr(b, "attn_k"):
        return 0.0
    return max(_max_abs_dict_diff(a.attn_k, b.attn_k), _max_abs_dict_diff(a.attn_v, b.attn_v))


@torch.no_grad()
def score_candidate(engine: RuntimeEngine, prompt_ids: Sequence[int], candidate: str, features: RuntimeFeatures) -> float:
    answer_ids = _encode_answer(engine, candidate)
    state = engine.new_state(features)
    logits = engine.prefill(prompt_ids, state, features)
    total = 0.0
    for i, token_id in enumerate(answer_ids):
        logp = F.log_softmax(logits.float(), dim=-1)
        total += float(logp[token_id].detach().cpu())
        if i != len(answer_ids) - 1:
            logits = engine.forward_one(token_id, state, return_logits=True)
    return total


@torch.no_grad()
def run_path_parity(
    engine: RuntimeEngine,
    prompt_ids: Sequence[int],
    *,
    chunk_size: int,
) -> PathParityResult:
    token_features = engine.features.with_overrides(block_prefill=False)
    block_features = engine.features.with_overrides(block_prefill=True, prefill_chunk_size=chunk_size)
    token_state = engine.new_state(token_features)
    block_state = engine.new_state(block_features)
    token_logits = engine.prefill(prompt_ids, token_state, token_features)
    block_logits = engine.prefill(prompt_ids, block_state, block_features)
    diff = (token_logits.float() - block_logits.float()).abs()
    token_argmax = sample_next(token_logits, engine_generation_greedy())
    block_argmax = sample_next(block_logits, engine_generation_greedy())
    pos_match = getattr(token_state, "pos", None) == getattr(block_state, "pos", None)
    kv_len_match = getattr(token_state, "kv_len", None) == getattr(block_state, "kv_len", None)
    gdn_state_max = _max_abs_dict_diff(getattr(token_state, "gdn_states", {}), getattr(block_state, "gdn_states", {}))
    conv_state_max = _max_abs_dict_diff(getattr(token_state, "gdn_conv_states", {}), getattr(block_state, "gdn_conv_states", {}))
    attention_kv_max = _attention_kv_max_abs_diff(token_state, block_state)
    continuation_ids = _encode_answer(engine, "7f3a9c2e-18b4-42a1")[:4]
    continuation_argmax_match = True
    continuation_max = 0.0
    continuation_mean_total = 0.0
    for token_id in continuation_ids:
        token_next = engine.forward_one(token_id, token_state, return_logits=True)
        block_next = engine.forward_one(token_id, block_state, return_logits=True)
        step_diff = (token_next.float() - block_next.float()).abs()
        continuation_argmax_match = continuation_argmax_match and sample_next(token_next, engine_generation_greedy()) == sample_next(
            block_next, engine_generation_greedy()
        )
        continuation_max = max(continuation_max, float(step_diff.max().detach().cpu()))
        continuation_mean_total += float(step_diff.mean().detach().cpu())
    return PathParityResult(
        input_tokens=len(prompt_ids),
        chunk_size=chunk_size,
        token_loop_argmax=token_argmax,
        block_prefill_argmax=block_argmax,
        argmax_match=token_argmax == block_argmax,
        max_abs_logit_diff=float(diff.max().detach().cpu()),
        mean_abs_logit_diff=float(diff.mean().detach().cpu()),
        pos_match=pos_match,
        kv_len_match=kv_len_match,
        gdn_state_max_abs_diff=gdn_state_max,
        conv_state_max_abs_diff=conv_state_max,
        attention_kv_max_abs_diff=attention_kv_max,
        continuation_tokens=len(continuation_ids),
        continuation_argmax_match=continuation_argmax_match,
        continuation_max_abs_logit_diff=continuation_max,
        continuation_mean_abs_logit_diff=continuation_mean_total / max(1, len(continuation_ids)),
    )


def engine_generation_greedy():
    return GenerationConfig.greedy(max_new_tokens=1)


def _batch_generate_ids_once(
    engine: RuntimeEngine,
    prompt_ids: Sequence[int],
    gen_cfg,
    features: RuntimeFeatures,
    *,
    runner: BatchedModelRunner | None = None,
    request_id: str = "batch-parity",
    prompt_cache_key: str | None = None,
) -> tuple[list[int], int]:
    local_runner = runner
    if local_runner is None:
        scheduler = ContinuousBatchScheduler(
            max_num_requests=1,
            max_num_batched_tokens=max(2, int(features.prefill_chunk_size), len(prompt_ids)),
            prefill_chunk_size=max(1, int(features.prefill_chunk_size)),
            kv_window_tokens=engine.recent_window,
        )
        local_runner = BatchedModelRunner(engine=engine, scheduler=scheduler, features=features, max_state_pool_size=0)
    row = local_runner.add_request(
        request_id,
        prompt_ids,
        generation_config=gen_cfg,
        prompt_cache_key=prompt_cache_key,
    )
    out: list[int] = []
    eos = set(int(t) for t in gen_cfg.eos_token_ids)
    stop_ids = set(int(t) for t in gen_cfg.stop_token_ids)
    try:
        while len(out) < int(gen_cfg.max_new_tokens):
            step = local_runner.execute_step(device=engine.device)
            if step is None:
                break
            for token in step.tokens_by_request().get(request_id, []):
                token_id = int(token)
                can_stop = len(out) >= int(gen_cfg.min_new_tokens)
                if can_stop and not gen_cfg.ignore_eos and token_id in eos:
                    return out, int(getattr(row, "prefix_cache_hit_tokens", 0) or 0)
                if can_stop and token_id in stop_ids:
                    return out, int(getattr(row, "prefix_cache_hit_tokens", 0) or 0)
                out.append(token_id)
                if len(out) >= int(gen_cfg.max_new_tokens):
                    return out, int(getattr(row, "prefix_cache_hit_tokens", 0) or 0)
    finally:
        local_runner.finish_request(request_id)
    return out, int(getattr(row, "prefix_cache_hit_tokens", 0) or 0)


@torch.no_grad()
def run_batch_path_parity(
    engine: RuntimeEngine,
    prompt_ids: Sequence[int],
    *,
    features: RuntimeFeatures,
    max_new_tokens: int,
) -> BatchPathParityResult:
    gen_cfg = GenerationConfig.greedy(max_new_tokens=max_new_tokens, eos_token_ids=engine.eos_token_ids())
    target_features = features.with_overrides(speculative_decoding=False, prefix_cache=False)
    direct_target = engine.generate_ids_greedy_gpu(prompt_ids, gen_cfg, features=target_features)
    batch_target, _ = _batch_generate_ids_once(engine, prompt_ids, gen_cfg, target_features, request_id="target")

    speculative_features = features.with_overrides(prefix_cache=False)
    batch_spec, _ = _batch_generate_ids_once(engine, prompt_ids, gen_cfg, speculative_features, request_id="spec")

    prefix_features = features.with_overrides(speculative_decoding=False, prefix_cache=True)
    scheduler = ContinuousBatchScheduler(
        max_num_requests=1,
        max_num_batched_tokens=max(2, int(prefix_features.prefill_chunk_size), len(prompt_ids)),
        prefill_chunk_size=max(1, int(prefix_features.prefill_chunk_size)),
        kv_window_tokens=engine.recent_window,
    )
    runner = BatchedModelRunner(engine=engine, scheduler=scheduler, features=prefix_features, max_state_pool_size=0)
    prefix_key = "batch-path-parity"
    _batch_generate_ids_once(
        engine,
        prompt_ids,
        gen_cfg,
        prefix_features,
        runner=runner,
        request_id="prefix-warm",
        prompt_cache_key=prefix_key,
    )
    batch_prefix, hit_tokens = _batch_generate_ids_once(
        engine,
        prompt_ids,
        gen_cfg,
        prefix_features,
        runner=runner,
        request_id="prefix-hit",
        prompt_cache_key=prefix_key,
    )

    direct = [int(t) for t in direct_target]
    target = [int(t) for t in batch_target]
    spec = [int(t) for t in batch_spec]
    prefix = [int(t) for t in batch_prefix]
    return BatchPathParityResult(
        prompt_tokens=len(prompt_ids),
        max_new_tokens=int(max_new_tokens),
        direct_target_only=direct,
        batch_target_only=target,
        batch_speculative=spec,
        batch_prefix_second_hit=prefix,
        target_only_match=direct == target,
        speculative_match=direct == spec,
        prefix_cache_match=direct == prefix,
        prefix_cache_hit_tokens=int(hit_tokens),
    )


@torch.no_grad()
def run_recall_suite(
    engine: RuntimeEngine,
    *,
    features: RuntimeFeatures,
    filler_repeats: Sequence[int],
) -> list[RecallCaseResult]:
    out: list[RecallCaseResult] = []
    candidate_sets = [
        ("short_secret", "NX-1742-ALPHA", ["NX-1742-ALPHA", "NX-1742-BETA", "NX-9041-ALPHA", "NX-3188-SIGMA"]),
        ("uuid_secret", "7f3a9c2e-18b4-42a1", ["7f3a9c2e-18b4-42a1", "7f3a9c2e-18b4-42b1", "4c19be77-0012-99af", "ALPHA42"]),
    ]
    for repeats in filler_repeats:
        for base_name, expected, candidates in candidate_sets:
            prompt = recall_prompt(expected, filler_repeats=repeats)
            prompt_ids = engine.encode_prompt(prompt)
            scores = {candidate: score_candidate(engine, prompt_ids, candidate, features) for candidate in candidates}
            predicted = max(scores.items(), key=lambda item: item[1])[0]
            out.append(
                RecallCaseResult(
                    name=f"{base_name}_filler{repeats}",
                    input_tokens=len(prompt_ids),
                    expected=expected,
                    predicted=predicted,
                    passed=predicted == expected,
                    scores=scores,
                )
            )
    return out


def run_report(args: argparse.Namespace) -> CorrectnessReport:
    features = runtime_features_from_args(args)
    engine = create_runtime_engine_from_args(args, features=features)
    parity_prompt = recall_prompt("NX-1742-ALPHA", filler_repeats=args.parity_filler_repeats)
    parity_ids = engine.encode_prompt(parity_prompt)
    parity = run_path_parity(engine, parity_ids, chunk_size=args.prefill_chunk_size or features.prefill_chunk_size)
    batch_parity = run_batch_path_parity(
        engine,
        parity_ids,
        features=features,
        max_new_tokens=args.batch_parity_new_tokens,
    )
    recall = run_recall_suite(
        engine,
        features=features,
        filler_repeats=[int(x) for x in args.recall_filler_repeats.split(",") if x.strip()],
    )
    return CorrectnessReport(
        path_parity=parity,
        batch_path_parity=batch_parity,
        recall=recall,
        require_block_prefill_parity=bool(args.require_block_prefill_parity),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LangBurst correctness gates before speed optimization")
    add_model_path_args(parser)
    add_adapter_arg(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recent-window", type=int, default=serving_recent_window_default())
    parser.add_argument("--cpu-embed", action="store_true")
    parser.add_argument("--parity-filler-repeats", type=int, default=8)
    parser.add_argument("--batch-parity-new-tokens", type=int, default=16)
    parser.add_argument("--recall-filler-repeats", default="0,8,32")
    parser.add_argument(
        "--require-block-prefill-parity",
        action="store_true",
        help="fail if experimental block prefill does not match token-loop prefill",
    )
    parser.add_argument("--json", action="store_true")
    add_runtime_feature_args(parser)
    args = parser.parse_args()
    report = run_report(args)
    payload = {
        "ok": report.ok,
        "require_block_prefill_parity": report.require_block_prefill_parity,
        "path_parity": asdict(report.path_parity),
        "batch_path_parity": asdict(report.batch_path_parity),
        "recall": [asdict(case) for case in report.recall],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"ok={payload['ok']}")
        print("path_parity", json.dumps(payload["path_parity"], ensure_ascii=False))
        print("batch_path_parity", json.dumps(payload["batch_path_parity"], ensure_ascii=False))
        for case in payload["recall"]:
            print("recall", json.dumps(case, ensure_ascii=False))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
