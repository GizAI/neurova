from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .adapters import Qwen36Adapter  # noqa: F401 - registers qwen36
from .cli_features import add_runtime_feature_args, runtime_features_from_args
from .core.adapter import adapter_registry
from .core.features import RuntimeFeatures
from .core.runtime import RuntimeEngine, sample_next


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
class CorrectnessReport:
    path_parity: PathParityResult
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
        return block_ok and all(case.passed for case in self.recall)


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
    from .core.runtime import GenerationConfig

    return GenerationConfig(max_new_tokens=1, temperature=0.0, top_k=0, eos_token_ids=())


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
    engine = RuntimeEngine(
        adapter=adapter_registry.get(args.adapter),
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        cpu_embed=args.cpu_embed,
        features=features,
    )
    parity_prompt = recall_prompt("NX-1742-ALPHA", filler_repeats=args.parity_filler_repeats)
    parity = run_path_parity(engine, engine.encode_prompt(parity_prompt), chunk_size=args.prefill_chunk_size or features.prefill_chunk_size)
    recall = run_recall_suite(
        engine,
        features=features,
        filler_repeats=[int(x) for x in args.recall_filler_repeats.split(",") if x.strip()],
    )
    return CorrectnessReport(
        path_parity=parity,
        recall=recall,
        require_block_prefill_parity=bool(args.require_block_prefill_parity),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="QwenBurst correctness gates before speed optimization")
    parser.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    parser.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    parser.add_argument("--adapter", default="qwen36", choices=("qwen36",))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--recent-window", type=int, default=8192)
    parser.add_argument("--cpu-embed", action="store_true")
    parser.add_argument("--parity-filler-repeats", type=int, default=8)
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
        "recall": [asdict(case) for case in report.recall],
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"ok={payload['ok']}")
        print("path_parity", json.dumps(payload["path_parity"], ensure_ascii=False))
        for case in payload["recall"]:
            print("recall", json.dumps(case, ensure_ascii=False))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
