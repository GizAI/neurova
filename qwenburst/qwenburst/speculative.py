from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .adapters import Qwen36Adapter
from .core.adapter import adapter_registry
from .core.runtime import GenerationConfig, RuntimeEngine, sample_next_tensor
from .loader import FP16Tensor, QuantizedStore
from .model import (
    QwenBurstMLP,
    QwenBurstModel,
    WeightResolver,
    apply_rope_single_token,
    embed_lookup,
    linear_any,
    qwen_rmsnorm,
    qwen_rmsnorm_torch,
)


@dataclass(frozen=True)
class SpeculativeProbeResult:
    method: str
    total: int
    accepted: int
    min_accept_rate: float

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def viable(self) -> bool:
        return self.total > 0 and self.accept_rate >= self.min_accept_rate


@dataclass(frozen=True)
class SpeculativeBenchmarkResult:
    method: str
    tokens: int
    target_seconds: float
    speculative_seconds: float
    accepted_second_tokens: int
    verified_steps: int
    identical: bool

    @property
    def target_tok_s(self) -> float:
        return self.tokens / self.target_seconds if self.target_seconds > 0 else 0.0

    @property
    def speculative_tok_s(self) -> float:
        return self.tokens / self.speculative_seconds if self.speculative_seconds > 0 else 0.0

    @property
    def speedup(self) -> float:
        return self.speculative_tok_s / self.target_tok_s if self.target_tok_s > 0 else 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted_second_tokens / self.verified_steps if self.verified_steps else 0.0

    @property
    def keep(self) -> bool:
        return self.identical and self.speedup > 1.03


class QwenNativeMTP1:
    """Qwen3.6 native MTP1 candidate generator.

    This is intentionally research-only until acceptance is high enough.  It
    follows the vLLM Qwen3Next MTP dataflow: normalize next-token embedding and
    previous hidden state, concatenate `[embedding, hidden]`, project through
    `mtp.fc`, run the single MTP full-attention layer, then project with the
    target lm_head.
    """

    def __init__(self, model: QwenBurstModel, store: QuantizedStore):
        self.model = model
        self.cfg = model.cfg
        self.device = model.device
        wr = WeightResolver(store)
        self.pre_fc_norm_embedding = wr.fp16("mtp.pre_fc_norm_embedding.weight").to(self.device, dtype=torch.float16).contiguous()
        self.pre_fc_norm_hidden = wr.fp16("mtp.pre_fc_norm_hidden.weight").to(self.device, dtype=torch.float16).contiguous()
        self.fc = wr.get("mtp.fc.weight")
        if not isinstance(self.fc, FP16Tensor):
            raise TypeError("mtp.fc.weight must be fp16_raw for native MTP probing")
        self.input_norm = wr.fp16("mtp.layers.0.input_layernorm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.post_norm = wr.fp16("mtp.layers.0.post_attention_layernorm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.qkv_proj = wr.any_linear("mtp.layers.0.self_attn.qkv_proj.weight")
        self.o_proj = wr.any_linear("mtp.layers.0.self_attn.o_proj.weight")
        self.q_norm = wr.fp16("mtp.layers.0.self_attn.q_norm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.k_norm = wr.fp16("mtp.layers.0.self_attn.k_norm.weight").to(self.device, dtype=torch.float16).contiguous()
        self.mlp = QwenBurstMLP(self.cfg, wr, prefix="mtp.layers.0")
        self.norm = wr.fp16("mtp.norm.weight").to(self.device, dtype=torch.float16).contiguous()
        kv_rows = self.cfg.num_key_value_heads * self.cfg.attention_head_dim
        self.qkv_q_rows = self.cfg.num_attention_heads * self.cfg.attention_head_dim * 2
        self.qkv_split = (self.qkv_q_rows, kv_rows, kv_rows)

    def _single_token_decoder_layer(self, x: torch.Tensor, *, pos: int) -> torch.Tensor:
        residual = x
        h = qwen_rmsnorm(x.contiguous(), self.input_norm, self.cfg.rms_norm_eps)
        qkv_all = linear_any(self.qkv_proj, h)
        q_all, k_all, v_all = torch.split(qkv_all, self.qkv_split, dim=0)
        q_heads = q_all.view(self.cfg.num_attention_heads, self.cfg.attention_head_dim * 2)
        q, gate = torch.chunk(q_heads, 2, dim=-1)
        k = k_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        v = v_all.view(self.cfg.num_key_value_heads, self.cfg.attention_head_dim)
        q = qwen_rmsnorm_torch(q.contiguous(), self.q_norm, self.cfg.rms_norm_eps)
        k = qwen_rmsnorm_torch(k.contiguous(), self.k_norm, self.cfg.rms_norm_eps)
        q, k = apply_rope_single_token(q, k, pos=pos, rope_dim=self.cfg.rope_dim, rope_theta=self.cfg.rope_theta)

        # Research approximation: no MTP KV cache is committed here.  The target
        # model remains the only verifier/committer.
        ratio = self.cfg.num_attention_heads // self.cfg.num_key_value_heads
        att = v.repeat_interleave(ratio, dim=0)
        att_flat = (att.reshape(-1) * torch.sigmoid(gate.reshape(-1).to(att.dtype))).contiguous()
        h = residual + linear_any(self.o_proj, att_flat)
        residual = h
        h = qwen_rmsnorm(h.contiguous(), self.post_norm, self.cfg.rms_norm_eps)
        return residual + self.mlp(h)

    @torch.no_grad()
    def logits_for_second_token(self, raw_hidden: torch.Tensor, first_token: torch.Tensor, *, pos: int) -> torch.Tensor:
        emb = embed_lookup(self.model.embed, first_token).to(self.device, dtype=torch.float16).reshape(-1).contiguous()
        h_norm = qwen_rmsnorm(raw_hidden.to(self.device, dtype=torch.float16).contiguous(), self.pre_fc_norm_hidden, self.cfg.rms_norm_eps)
        e_norm = qwen_rmsnorm(emb, self.pre_fc_norm_embedding, self.cfg.rms_norm_eps)
        x = torch.cat([e_norm, h_norm], dim=0)
        x = linear_any(self.fc, x).to(self.device, dtype=torch.float16).contiguous()
        x = self._single_token_decoder_layer(x, pos=pos)
        x = qwen_rmsnorm(x.contiguous(), self.norm, self.cfg.rms_norm_eps)
        return linear_any(self.model.lm_head, x)


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
    mtp = QwenNativeMTP1(engine.model, engine.model.store)
    state = engine.new_state()
    prompt_ids = engine.encode_prompt(prompt)
    logits: torch.Tensor | None = None
    raw_hidden: torch.Tensor | None = None
    for i, tid in enumerate(prompt_ids):
        if i == len(prompt_ids) - 1:
            logits, raw_hidden = engine.model.forward_one(tid, state, return_hidden=True, return_raw_hidden=True)
        else:
            engine.model.forward_one(tid, state, return_logits=False)
    assert logits is not None and raw_hidden is not None
    gen_cfg = GenerationConfig(max_new_tokens=steps)
    accepted = 0
    total = 0
    for _ in range(steps):
        first = sample_next_tensor(logits, gen_cfg)
        mtp_logits = mtp.logits_for_second_token(raw_hidden, first, pos=state.pos)
        candidate_second = sample_next_tensor(mtp_logits, gen_cfg)

        logits, raw_hidden = engine.model.forward_one(first, state, return_hidden=True, return_raw_hidden=True)
        target_second = sample_next_tensor(logits, gen_cfg)
        accepted += int(torch.equal(candidate_second.reshape(()).cpu(), target_second.reshape(()).cpu()))
        total += 1
    return SpeculativeProbeResult("native_mtp1", total=total, accepted=accepted, min_accept_rate=min_accept_rate)


def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _target_only_ids(engine: RuntimeEngine, prompt_ids: list[int], *, max_new_tokens: int) -> tuple[list[int], float]:
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    _sync_if_cuda()
    t0 = time.perf_counter()
    out = engine.generate_ids_greedy_gpu(prompt_ids, cfg)
    _sync_if_cuda()
    return out, time.perf_counter() - t0


@torch.no_grad()
def generate_native_mtp1_ids(
    engine: RuntimeEngine,
    prompt_ids: list[int],
    *,
    max_new_tokens: int,
    adaptive: bool = False,
    min_verified: int = 8,
    accept_threshold: float = 0.80,
) -> tuple[list[int], float, int, int]:
    if not isinstance(engine.model, QwenBurstModel):
        raise TypeError("native_mtp1 currently supports QwenBurstModel only")
    mtp = QwenNativeMTP1(engine.model, engine.model.store)
    state = engine.new_state()
    _sync_if_cuda()
    t0 = time.perf_counter()
    logits: torch.Tensor | None = None
    raw_hidden: torch.Tensor | None = None
    for i, tid in enumerate(prompt_ids):
        if i == len(prompt_ids) - 1:
            logits, raw_hidden = engine.model.forward_one(tid, state, return_hidden=True, return_raw_hidden=True)
        else:
            engine.model.forward_one(tid, state, return_logits=False)
    assert logits is not None and raw_hidden is not None
    cfg = GenerationConfig(max_new_tokens=max_new_tokens)
    out: list[int] = []
    accepted = 0
    verified = 0
    while len(out) < max_new_tokens:
        first = sample_next_tensor(logits, cfg)
        first_id = int(first.detach().cpu().item())
        candidate_logits = mtp.logits_for_second_token(raw_hidden, first, pos=state.pos)
        candidate = sample_next_tensor(candidate_logits, cfg)
        candidate_id = int(candidate.detach().cpu().item())

        branch = state.fork()
        verified_block = engine.model.forward_block([first_id, candidate_id], branch, return_logits=True, commit=True)
        target_second = sample_next_tensor(verified_block.logits[0], cfg)
        target_second_id = int(target_second.detach().cpu().item())
        verified += 1
        if candidate_id == target_second_id and len(out) + 2 <= max_new_tokens:
            state = branch
            out.extend([first_id, candidate_id])
            logits = verified_block.logits[1]
            raw_hidden = verified_block.raw_hiddens[1]
            accepted += 1
        else:
            logits, raw_hidden = engine.model.forward_one(first_id, state, return_hidden=True, return_raw_hidden=True)
            out.append(first_id)
        if adaptive and verified >= min_verified and (accepted / verified) < accept_threshold:
            while len(out) < max_new_tokens:
                next_token = sample_next_tensor(logits, cfg)
                next_id = int(next_token.detach().cpu().item())
                out.append(next_id)
                if len(out) >= max_new_tokens:
                    break
                logits = engine.model.forward_one(next_id, state, return_logits=True)
            break
    _sync_if_cuda()
    return out[:max_new_tokens], time.perf_counter() - t0, accepted, verified


@torch.no_grad()
def benchmark_native_mtp1_with_engine(
    engine: RuntimeEngine,
    *,
    prompt: str,
    max_new_tokens: int,
) -> SpeculativeBenchmarkResult:
    if not isinstance(engine.model, QwenBurstModel):
        raise TypeError("native_mtp1 currently supports QwenBurstModel only")
    prompt_ids = engine.encode_prompt(prompt)
    target_ids, target_s = _target_only_ids(engine, prompt_ids, max_new_tokens=max_new_tokens)
    speculative_ids, spec_s, accepted, verified = generate_native_mtp1_ids(
        engine,
        prompt_ids,
        max_new_tokens=max_new_tokens,
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
    return benchmark_native_mtp1_with_engine(engine, prompt=prompt, max_new_tokens=max_new_tokens)


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
    bench.add_argument("--recent-window", type=int, default=256)
    bench.add_argument("--adaptive", action="store_true")
    bench.add_argument("--min-verified", type=int, default=8)
    bench.add_argument("--accept-threshold", type=float, default=0.80)
    suite = sub.add_parser("bench-suite-mtp1", help="benchmark MTP1 over several prompts with one model load")
    suite.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    suite.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    suite.add_argument("--recent-window", type=int, default=64)
    suite.add_argument("--max-new-tokens", type=int, default=32)
    suite.add_argument("--adaptive", action="store_true")
    suite.add_argument("--min-verified", type=int, default=8)
    suite.add_argument("--accept-threshold", type=float, default=0.80)
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
                adaptive=True,
                min_verified=args.min_verified,
                accept_threshold=args.accept_threshold,
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
                    adaptive=True,
                    min_verified=args.min_verified,
                    accept_threshold=args.accept_threshold,
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
                result = benchmark_native_mtp1_with_engine(engine, prompt=prompt, max_new_tokens=tokens)
            row = {
                "name": name,
                "tokens": result.tokens,
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


if __name__ == "__main__":
    main()
