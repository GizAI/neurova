from __future__ import annotations

import json
import threading
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch

from .adapter import ModelAdapter
from .features import RuntimeFeatures, RuntimePlan, resolve_runtime_plan
from .platform import resolve_index_file
from .policy import ExecutionPolicy
from .text_stream import StreamingTextDecoder
from ..speculative_batch import DecodeBatchPlan
from ..ops import cuda_ops
from ..speculation import SpeculativeDecodePolicy, SpeculativeDecodeResult, SpeculativeDecodeStats, SpeculativeProposer
from ..speculative_verifier import NativeNextNVerifier, TargetVerification


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    min_new_tokens: int = 0
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    logit_bias: dict[int, float] | None = None
    bad_token_ids: tuple[int, ...] = ()
    suppress_tokens: tuple[int, ...] = ()
    seed: int | None = None
    eos_token_ids: tuple[int, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    ignore_eos: bool = False

    @classmethod
    def greedy(cls, *, max_new_tokens: int, eos_token_ids: Sequence[int] = ()) -> "GenerationConfig":
        return cls(
            max_new_tokens=int(max_new_tokens),
            temperature=0.0,
            top_k=0,
            eos_token_ids=tuple(int(t) for t in eos_token_ids),
        )


def _apply_generation_constraints(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    *,
    history: Sequence[int] = (),
    generated: Sequence[int] = (),
) -> torch.Tensor:
    if not _has_generation_constraints(cfg, generated=generated):
        return logits
    scores = logits.float().clone()
    if cfg.logit_bias:
        for token, bias in cfg.logit_bias.items():
            idx = int(token)
            if 0 <= idx < scores.numel():
                scores[idx] += float(bias)
    for token in tuple(cfg.bad_token_ids) + tuple(cfg.suppress_tokens):
        idx = int(token)
        if 0 <= idx < scores.numel():
            scores[idx] = -torch.inf
    if generated:
        counts: dict[int, int] = {}
        for token in generated:
            counts[int(token)] = counts.get(int(token), 0) + 1
        for token, count in counts.items():
            if 0 <= token < scores.numel():
                if cfg.repetition_penalty and cfg.repetition_penalty != 1.0:
                    if scores[token] < 0:
                        scores[token] *= float(cfg.repetition_penalty)
                    else:
                        scores[token] /= float(cfg.repetition_penalty)
                if cfg.presence_penalty:
                    scores[token] -= float(cfg.presence_penalty)
                if cfg.frequency_penalty:
                    scores[token] -= float(cfg.frequency_penalty) * int(count)
    if cfg.no_repeat_ngram_size and cfg.no_repeat_ngram_size > 1:
        n = int(cfg.no_repeat_ngram_size)
        seq = [int(t) for t in history]
        if len(seq) >= n - 1:
            prefix = tuple(seq[-(n - 1) :])
            banned: set[int] = set()
            for i in range(0, len(seq) - n + 1):
                if tuple(seq[i : i + n - 1]) == prefix:
                    banned.add(seq[i + n - 1])
            for token in banned:
                if 0 <= token < scores.numel():
                    scores[token] = -torch.inf
    return scores


def _has_generation_constraints(cfg: GenerationConfig, *, generated: Sequence[int] = ()) -> bool:
    return any(
        (
            bool(cfg.logit_bias),
            bool(cfg.bad_token_ids),
            bool(cfg.suppress_tokens),
            bool(generated)
            and (
                (cfg.repetition_penalty and cfg.repetition_penalty != 1.0)
                or bool(cfg.presence_penalty)
                or bool(cfg.frequency_penalty)
            ),
            bool(cfg.no_repeat_ngram_size and cfg.no_repeat_ngram_size > 1),
        )
    )


def _filter_top_p_min_p(scores: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
    filtered = scores
    if cfg.top_k and cfg.top_k > 0 and cfg.top_k < filtered.numel():
        vals, idx = torch.topk(filtered, int(cfg.top_k))
        next_scores = torch.full_like(filtered, -torch.inf)
        next_scores[idx] = vals
        filtered = next_scores
    if cfg.min_p and cfg.min_p > 0:
        probs = torch.softmax(filtered, dim=-1)
        max_prob = torch.max(probs)
        filtered = torch.where(probs >= max_prob * float(cfg.min_p), filtered, torch.full_like(filtered, -torch.inf))
    if cfg.top_p and 0 < cfg.top_p < 1.0:
        sorted_scores, sorted_idx = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > float(cfg.top_p)
        if remove.numel() > 0:
            remove[0] = False
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)
        next_scores = torch.full_like(filtered, -torch.inf)
        next_scores[sorted_idx] = sorted_scores
        filtered = next_scores
    return filtered


def sample_next(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    *,
    history: Sequence[int] = (),
    generated: Sequence[int] = (),
    sample_index: int = 0,
) -> int:
    scores = _apply_generation_constraints(logits, cfg, history=history, generated=generated)
    if cfg.temperature <= 0:
        return int(torch.argmax(scores, dim=-1).item())
    scores = _filter_top_p_min_p(scores / max(cfg.temperature, 1e-6), cfg)
    probs = torch.softmax(scores, dim=-1)
    generator = None
    if cfg.seed is not None:
        generator = torch.Generator(device=probs.device)
        generator.manual_seed(int(cfg.seed) + int(sample_index))
    return int(torch.multinomial(probs, 1, generator=generator).item())


def sample_next_tensor(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    *,
    history: Sequence[int] = (),
    generated: Sequence[int] = (),
    sample_index: int = 0,
) -> torch.Tensor:
    if cfg.temperature <= 0:
        scores = _apply_generation_constraints(logits, cfg, history=history, generated=generated)
        if logits.device.type == "cuda" and scores.data_ptr() == logits.data_ptr():
            return cuda_ops().argmax(scores.contiguous()).reshape(()).to(device=logits.device, dtype=torch.long)
        return torch.argmax(scores, dim=-1).reshape(()).to(device=logits.device, dtype=torch.long)
    token = sample_next(logits, cfg, history=history, generated=generated, sample_index=sample_index)
    return torch.tensor(token, device=logits.device, dtype=torch.long)


class RuntimeEngine:
    """Model-independent single-user runtime.

    Adapters own architecture-specific config, weight mapping, chat template,
    and state allocation.  The runtime owns the common prefill/decode/server
    contract so new model families do not fork the serving loop.
    """

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        hf_model: Path,
        qb_model: Path,
        device: str,
        recent_window: int,
        weight_device: str,
        cpu_embed: bool = False,
        model_name: str | None = None,
        features: RuntimeFeatures | None = None,
        max_state_pool_size: int = 0,
    ) -> None:
        self.adapter = adapter
        self.hf_model = Path(hf_model)
        self.qb_model = Path(qb_model)
        self.device = device
        self.recent_window = int(recent_window)
        self.weight_device = weight_device
        self.cpu_embed = bool(cpu_embed)
        self.features = features or RuntimeFeatures.from_profile("stateful")
        if max_state_pool_size < 0:
            raise ValueError("max_state_pool_size must be >= 0")
        self.max_state_pool_size = int(max_state_pool_size)
        self.model_name = model_name or adapter.descriptor.default_model_name
        self.cfg = adapter.load_config(self.hf_model)
        self.tokenizer = adapter.load_tokenizer(self.hf_model)
        self.model = adapter.create_model(
            qb_model=self.qb_model,
            cfg=self.cfg,
            device=self.device,
            weight_device=self.weight_device,
            cpu_embed=self.cpu_embed,
        )
        self.lock = threading.Lock()
        self._state_pool: dict[tuple[tuple[str, object], ...], list[Any]] = {}
        forward_block = getattr(self.model, "forward_block", None)
        self._forward_block = forward_block
        self._forward_block_supports_logits_mode = (
            forward_block is not None and "logits_mode" in inspect.signature(forward_block).parameters
        )
        create_proposer = getattr(adapter, "create_speculative_proposer", None)
        self.speculative_proposer: SpeculativeProposer | None = (
            create_proposer(self.model) if callable(create_proposer) else None
        )

    def resolve_plan(self, features: RuntimeFeatures | None = None) -> RuntimePlan:
        return resolve_runtime_plan(features or self.features, self.adapter.descriptor.capabilities)

    def resolve_policy(
        self,
        features: RuntimeFeatures | None = None,
        *,
        speculative: SpeculativeDecodePolicy | None = None,
    ) -> ExecutionPolicy:
        return ExecutionPolicy.from_plan(self.resolve_plan(features), speculative=speculative)

    def _state_pool_key(self, features: RuntimeFeatures) -> tuple[tuple[str, object], ...]:
        plan = self.resolve_plan(features)
        return tuple(sorted(plan.effective.summary().items()))

    def new_state(self, features: RuntimeFeatures | None = None) -> Any:
        plan = self.resolve_plan(features)
        return self.adapter.allocate_state(
            self.cfg,
            recent_window=self.recent_window,
            device=self.device,
            features=plan.effective,
        )

    def create_state_arena(
        self,
        *,
        features: RuntimeFeatures,
        max_slots: int,
        kv_num_blocks: int | None = None,
        kv_block_size: int | None = None,
    ) -> Any | None:
        create_arena = getattr(self.adapter, "create_state_arena", None)
        if not callable(create_arena):
            return None
        return create_arena(
            self.cfg,
            max_seq_len=self.recent_window,
            num_slots=max_slots,
            device=self.device,
            features=features,
            kv_num_blocks=kv_num_blocks,
            kv_block_size=kv_block_size,
        )

    def estimated_state_bytes(self) -> int:
        estimate = getattr(self.adapter, "estimate_state_bytes", None)
        if callable(estimate):
            features = self.resolve_plan(self.features).effective
            try:
                return int(estimate(self.cfg, recent_window=self.recent_window, features=features))
            except TypeError:
                return int(estimate(self.cfg, recent_window=self.recent_window))
        return 0

    def estimated_weight_bytes(self) -> int:
        try:
            index = json.loads(resolve_index_file(self.qb_model).read_text(encoding="utf-8"))
        except Exception:
            return 0
        total = 0
        for meta in index.get("tensors", {}).values():
            for field_name in ("qweight", "scales", "path"):
                rel = meta.get(field_name)
                if not rel:
                    continue
                try:
                    total += (self.qb_model / rel).stat().st_size
                except FileNotFoundError:
                    pass
        return total

    @contextmanager
    def pooled_state(self, features: RuntimeFeatures | None = None):
        plan = self.resolve_plan(features)
        if not plan.effective.state_pool:
            state = self.new_state(plan.effective)
            try:
                yield state
            finally:
                reset = getattr(state, "reset", None)
                if callable(reset):
                    reset(reset_attention=True)
            return
        key = self._state_pool_key(plan.effective)
        pool = self._state_pool.setdefault(key, [])
        state = pool.pop() if pool else self.new_state(plan.effective)
        reset = getattr(state, "reset", None)
        if callable(reset):
            reset(reset_attention=True)
        try:
            yield state
        finally:
            reset = getattr(state, "reset", None)
            if callable(reset):
                reset(reset_attention=True)
            if len(pool) < self.max_state_pool_size:
                pool.append(state)

    def clear_state_pool(self) -> None:
        self._state_pool.clear()

    def state_pool_summary(self) -> dict[str, object]:
        return {
            "max_state_pool_size": self.max_state_pool_size,
            "keys": len(self._state_pool),
            "pooled_states": sum(len(states) for states in self._state_pool.values()),
        }

    def encode_prompt(self, prompt: str, system: str | None = None) -> list[int]:
        return self.adapter.encode_prompt(self.tokenizer, prompt, system)

    def encode_messages(self, messages: Sequence[dict[str, Any]]) -> list[int]:
        return self.adapter.encode_messages(self.tokenizer, messages)

    def eos_token_ids(self) -> tuple[int, ...]:
        return self.adapter.eos_token_ids(self.tokenizer)

    def forward_one(self, token: int | torch.Tensor, state: Any, *, return_logits: bool = True) -> torch.Tensor:
        try:
            return self.model.forward_one(token, state, use_mtp=False, return_logits=return_logits)
        except TypeError:
            return self.model.forward_one(int(token), state, use_mtp=False)

    @torch.no_grad()
    def forward_batch(
        self,
        plan: DecodeBatchPlan,
        states: Sequence[Any],
        *,
        return_logits: bool = True,
    ) -> list[torch.Tensor | None]:
        """Execute a continuous-serving decode batch plan.

        If the adapter/model exposes a native `forward_batch(plan, states)`, use
        it.  Otherwise this correctness-preserving fallback executes each row's
        scheduled token span against its own DecodeState.  The public contract is
        the same, so server/scheduler code does not fork when a true batched CUDA
        implementation lands.
        """

        if len(states) != plan.num_requests:
            raise ValueError("states length must match plan.num_requests")
        native = getattr(self.model, "forward_batch", None)
        if callable(native):
            return native(plan, states, return_logits=return_logits)

        outputs: list[torch.Tensor | None] = []
        forward_block = getattr(self.model, "forward_block", None)
        for row, state in enumerate(states):
            start, end = plan.row_spans[row]
            row_tokens = plan.input_ids[start:end].detach().cpu().tolist()
            row_logits: torch.Tensor | None = None
            if callable(forward_block) and len(row_tokens) > 1:
                result = forward_block(
                    row_tokens,
                    state,
                    return_logits=return_logits,
                    logits_mode="last",
                    commit=True,
                )
                row_logits = result.logits[-1] if return_logits and result.logits else None
            else:
                for i, token in enumerate(row_tokens):
                    want_logits = return_logits and (i == len(row_tokens) - 1)
                    result = self.forward_one(int(token), state, return_logits=want_logits)
                    if want_logits:
                        row_logits = result
            outputs.append(row_logits)
        return outputs

    @torch.no_grad()
    def forward_batch_logits(
        self,
        plan: DecodeBatchPlan,
        states: Sequence[Any],
    ) -> list[list[torch.Tensor]]:
        """Execute a batch and return every row's scheduled logits.

        Rows without draft tokens return only the final logit row. Rows with
        draft tokens return one logit row per scheduled token, matching the reference runtime's
        rejection-sampler input shape.
        """

        if len(states) != plan.num_requests:
            raise ValueError("states length must match plan.num_requests")
        native = getattr(self.model, "forward_batch_logits", None)
        if callable(native):
            return native(plan, states)

        outputs: list[list[torch.Tensor]] = []
        forward_block = getattr(self.model, "forward_block", None)
        for row, state in enumerate(states):
            start, end = plan.row_spans[row]
            row_tokens = plan.input_ids[start:end].detach().cpu().tolist()
            wants_all = plan.num_draft_tokens_per_request[row] > 0
            if callable(forward_block) and len(row_tokens) > 1:
                result = forward_block(
                    row_tokens,
                    state,
                    return_logits=True,
                    logits_mode="all" if wants_all else "last",
                    commit=True,
                )
                outputs.append([logit for logit in result.logits])
                continue
            row_logits: list[torch.Tensor] = []
            for i, token in enumerate(row_tokens):
                want_logits = wants_all or i == len(row_tokens) - 1
                result = self.forward_one(int(token), state, return_logits=want_logits)
                if want_logits:
                    row_logits.append(result)
            outputs.append(row_logits)
        return outputs

    @torch.no_grad()
    def _prefill_token_loop(self, ids: list[int], state: Any) -> torch.Tensor:
        logits: torch.Tensor | None = None
        for i, tid in enumerate(ids):
            logits = self.forward_one(tid, state, return_logits=(i == len(ids) - 1))
        assert logits is not None
        return logits

    def _prefill_block(self, ids: list[int], state: Any, features: RuntimeFeatures) -> torch.Tensor:
        forward_block = self._forward_block
        if forward_block is None:
            return self._prefill_token_loop(ids, state)
        logits: torch.Tensor | None = None
        chunk_size = max(1, int(features.prefill_chunk_size))
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            is_last = start + chunk_size >= len(ids)
            if self._forward_block_supports_logits_mode:
                result = forward_block(
                    chunk,
                    state,
                    return_logits=is_last,
                    logits_mode="last",
                    commit=True,
                )
            else:
                result = forward_block(chunk, state, return_logits=is_last, commit=True)
            if is_last:
                if not result.logits:
                    raise RuntimeError("block prefill did not return final logits")
                logits = result.logits[-1]
        assert logits is not None
        return logits

    @torch.no_grad()
    def prefill(
        self,
        input_ids: Iterable[int],
        state: Any,
        features: RuntimeFeatures | None = None,
    ) -> torch.Tensor:
        ids = [int(tid) for tid in input_ids]
        if not ids:
            raise ValueError("prefill requires at least one token")
        features = features or self.features
        features = self.resolve_plan(features).effective
        if features.block_prefill:
            return self._prefill_block(ids, state, features)
        return self._prefill_token_loop(ids, state)

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
    ) -> Iterator[int]:
        state = self.new_state(features) if state is None else state
        features = self.resolve_plan(features).effective
        prompt_list = [int(t) for t in prompt_ids]
        logits = self.prefill(prompt_list, state, features)
        generated: list[int] = []
        next_id = sample_next(logits, gen_cfg, history=prompt_list, generated=generated)
        for i in range(gen_cfg.max_new_tokens):
            can_stop = len(generated) >= int(gen_cfg.min_new_tokens)
            if can_stop and not gen_cfg.ignore_eos and gen_cfg.eos_token_ids and next_id in gen_cfg.eos_token_ids:
                break
            if can_stop and gen_cfg.stop_token_ids and next_id in gen_cfg.stop_token_ids:
                break
            yield next_id
            generated.append(int(next_id))
            if i == gen_cfg.max_new_tokens - 1:
                break
            logits = self.forward_one(next_id, state, return_logits=True)
            next_id = sample_next(
                logits,
                gen_cfg,
                history=prompt_list + generated,
                generated=generated,
                sample_index=len(generated),
            )

    @torch.no_grad()
    def generate_ids_greedy_gpu(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
    ) -> list[int]:
        return self.generate_decode_result(prompt_ids, gen_cfg, state=state, features=features).ids

    @torch.no_grad()
    def generate_decode_result(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
        *,
        policy: ExecutionPolicy | None = None,
    ) -> SpeculativeDecodeResult:
        """Generate tokens through the single resolved decode policy.

        The returned stats are meaningful for native NEXTN speculation and
        still give callers one result object for plain greedy or sampling
        fallback paths.
        """

        stats = SpeculativeDecodeStats(method="greedy")
        if gen_cfg.max_new_tokens <= 0:
            return SpeculativeDecodeResult([], stats)
        policy = policy or self.resolve_policy(features)
        features = policy.features
        prompt_list = [int(t) for t in prompt_ids]
        if gen_cfg.temperature > 0 or gen_cfg.top_k > 0 or not features.gpu_sampling:
            stats.fallback_reason = "sampling_or_cpu_path"
            return SpeculativeDecodeResult(list(self.generate_ids(prompt_list, gen_cfg, state, features)), stats)
        has_speculative_proposer = self.speculative_proposer is not None
        if features.speculative_decoding and has_speculative_proposer:
            return self.generate_native_nextn_result(
                prompt_list,
                gen_cfg,
                state=state,
                features=features,
                policy=policy.speculative,
            )
        return SpeculativeDecodeResult(
            self._generate_ids_greedy_gpu_plain(prompt_list, gen_cfg, state=state, features=features),
            stats,
        )

    @torch.no_grad()
    def _generate_ids_greedy_gpu_plain(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
    ) -> list[int]:
        features = self.resolve_plan(features).effective
        state = self.new_state(features) if state is None else state
        logits = self.prefill(prompt_ids, state, features)
        return self._continue_ids_greedy_gpu_plain(logits, gen_cfg, state)

    @torch.no_grad()
    def _continue_ids_greedy_gpu_plain(
        self,
        logits: torch.Tensor,
        gen_cfg: GenerationConfig,
        state: Any,
        *,
        first_token: torch.Tensor | None = None,
        prefix: Sequence[int] = (),
    ) -> list[int]:
        """GPU-resident greedy continuation used by both normal and fallback decode."""
        if len(prefix) >= gen_cfg.max_new_tokens:
            return [int(t) for t in prefix[: gen_cfg.max_new_tokens]]
        remaining = gen_cfg.max_new_tokens - len(prefix)
        prompt_history = [int(t) for t in prefix]
        next_token = (
            sample_next_tensor(logits, gen_cfg, history=prompt_history, generated=prompt_history)
            if first_token is None
            else first_token
        )
        out = torch.empty((remaining,), device=next_token.device, dtype=torch.long)
        produced = 0
        for i in range(remaining):
            next_id = int(next_token.detach().cpu().item())
            can_stop = len(prefix) + produced >= int(gen_cfg.min_new_tokens)
            if can_stop and not gen_cfg.ignore_eos and gen_cfg.eos_token_ids and next_id in set(gen_cfg.eos_token_ids):
                break
            if can_stop and gen_cfg.stop_token_ids and next_id in set(gen_cfg.stop_token_ids):
                break
            out[i] = next_token
            produced += 1
            if i == remaining - 1:
                break
            logits = self.forward_one(next_token, state, return_logits=True)
            generated_so_far = [int(t) for t in prefix]
            generated_so_far.extend(int(t) for t in out[:produced].detach().cpu().tolist())
            next_token = sample_next_tensor(
                logits,
                gen_cfg,
                history=generated_so_far,
                generated=generated_so_far,
                sample_index=len(generated_so_far),
            )
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        ids = [int(t) for t in prefix]
        ids.extend(out[:produced].detach().cpu().tolist())
        if gen_cfg.eos_token_ids:
            eos = set(gen_cfg.eos_token_ids)
            for i, tid in enumerate(ids):
                if tid in eos:
                    return [int(t) for t in ids[:i]]
        return [int(t) for t in ids]

    @torch.no_grad()
    def _prefill_with_raw_hidden(
        self,
        ids: list[int],
        state: Any,
        features: RuntimeFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_block = self._forward_block
        if forward_block is not None and self._forward_block_supports_logits_mode:
            logits: torch.Tensor | None = None
            raw_hidden: torch.Tensor | None = None
            chunk_size = max(1, int(features.prefill_chunk_size))
            for start in range(0, len(ids), chunk_size):
                chunk = ids[start : start + chunk_size]
                is_last = start + chunk_size >= len(ids)
                result = forward_block(
                    chunk,
                    state,
                    return_logits=is_last,
                    logits_mode="last",
                    commit=True,
                )
                if is_last:
                    if not result.logits or not result.raw_hiddens:
                        raise RuntimeError("block prefill did not return final logits/raw hidden")
                    logits = result.logits[-1]
                    if result.final_hiddens:
                        raw_hidden = result.final_hiddens[-1]
                    else:
                        raw_hidden = result.raw_hiddens[-1]
            assert logits is not None and raw_hidden is not None
            return logits, raw_hidden
        logits: torch.Tensor | None = None
        raw_hidden: torch.Tensor | None = None
        for i, tid in enumerate(ids):
            if i == len(ids) - 1:
                logits, raw_hidden = self.model.forward_one(
                    tid,
                    state,
                    return_hidden=True,
                    return_raw_hidden=False,
                )
            else:
                self.forward_one(tid, state, return_logits=False)
        assert logits is not None and raw_hidden is not None
        return logits, raw_hidden

    @torch.no_grad()
    def _verify_block_sequential_with_raw_hidden(
        self,
        tokens: Sequence[int],
        state: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        logits_out: list[torch.Tensor] = []
        raw_hiddens: list[torch.Tensor] = []
        for token in tokens:
            logits, raw_hidden = self.model.forward_one(
                int(token),
                state,
                return_hidden=True,
                return_raw_hidden=True,
            )
            logits_out.append(logits.contiguous().clone())
            raw_hiddens.append(raw_hidden.contiguous().clone())
        return logits_out, raw_hiddens

    def _target_ids_for_candidate_logits(self, logits: Sequence[torch.Tensor], num_candidates: int) -> torch.Tensor:
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if len(logits) < num_candidates:
            raise RuntimeError("target verifier did not return enough logits")
        check_logits = torch.stack([logits[i].contiguous() for i in range(num_candidates)], dim=0)
        if check_logits.device.type == "cuda":
            return torch.argmax(check_logits, dim=-1).to(device=check_logits.device, dtype=torch.long)
        return torch.argmax(check_logits, dim=-1).to(dtype=torch.long)

    def _target_verification_from_block_result(self, result: Any, num_candidates: int) -> TargetVerification:
        if not result.logits:
            raise RuntimeError("target verifier did not return logits")
        target_ids = self._target_ids_for_candidate_logits(result.logits, int(num_candidates))
        final_hiddens = getattr(result, "final_hiddens", None)
        raw_hiddens = getattr(result, "raw_hiddens", None)
        if final_hiddens:
            raw_hidden = final_hiddens[-1]
        elif raw_hiddens:
            raw_hidden = raw_hiddens[-1]
        else:
            raw_hidden = result.logits[-1]
        return TargetVerification(target_ids=target_ids, logits=result.logits[-1], raw_hidden=raw_hidden)

    @torch.no_grad()
    def verify_nextn_tokens(
        self,
        token_ids: Sequence[int],
        state: Any,
        num_candidates: int,
    ) -> TargetVerification:
        """Verify a sampled token plus native MTP/NEXTN drafts in one target call.

        This is the runtime-owned verifier boundary.  LangBurst routes it
        through a continuous-serving batch plan first, then falls back to the older
        single-request block/scalar paths only when an adapter has not exposed
        `forward_verify_batch`.
        """

        verify_batch = getattr(self.model, "forward_verify_batch", None)
        if callable(verify_batch):
            plan = self._build_single_verify_batch_plan(token_ids, state, num_candidates)
            result = verify_batch(plan, [state])[0]
            return TargetVerification(target_ids=result.target_ids, logits=result.logits, raw_hidden=result.hidden)

        verify_block = getattr(self.model, "forward_verify_block", None)
        if callable(verify_block):
            result = verify_block(token_ids, state, num_candidates=int(num_candidates))
            return TargetVerification(target_ids=result.target_ids, logits=result.logits, raw_hidden=result.hidden)

        forward_block = getattr(self.model, "forward_block", None)
        if callable(forward_block):
            result = forward_block(
                token_ids,
                state,
                return_logits=True,
                logits_mode="all",
                commit=True,
            )
            return self._target_verification_from_block_result(result, int(num_candidates))

        logits, raw_hiddens = self._verify_block_sequential_with_raw_hidden(token_ids, state)
        target_ids = self._target_ids_for_candidate_logits(logits, int(num_candidates))
        return TargetVerification(target_ids=target_ids, logits=logits[-1], raw_hidden=raw_hiddens[-1])

    def _build_single_verify_batch_plan(
        self,
        token_ids: Sequence[int],
        state: Any,
        num_candidates: int,
    ) -> DecodeBatchPlan:
        token_list = [int(t) for t in token_ids]
        if not token_list:
            raise ValueError("verify batch requires at least one token")
        if num_candidates < 0 or num_candidates >= len(token_list):
            raise ValueError("num_candidates must be in [0, len(token_ids) - 1]")
        device = torch.device(self.device)
        pos = int(getattr(state, "pos", 0))
        state_index = int(getattr(state, "arena_slot", 0) or 0)
        n = len(token_list)
        return DecodeBatchPlan(
            request_ids=["native-nextn-verify"],
            state_indices=torch.tensor([state_index], dtype=torch.int32, device=device),
            input_ids=torch.tensor(token_list, dtype=torch.long, device=device),
            positions=torch.arange(pos, pos + n, dtype=torch.long, device=device),
            query_start_loc=torch.tensor([0, n], dtype=torch.int32, device=device),
            seq_lens=torch.tensor([pos + n], dtype=torch.int32, device=device),
            logits_indices=torch.arange(0, n, dtype=torch.long, device=device),
            cu_num_logits=torch.tensor([0, n], dtype=torch.int32, device=device),
            row_spans=((0, n),),
            num_scheduled_tokens=[n],
            num_draft_tokens_per_request=[int(num_candidates)],
            is_prefill=[False],
        )

    @torch.no_grad()
    def generate_native_nextn_result(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
        *,
        policy: SpeculativeDecodePolicy | None = None,
    ) -> SpeculativeDecodeResult:
        resolved = self.resolve_policy(features, speculative=policy)
        policy = resolved.speculative
        stats = SpeculativeDecodeStats(max_draft=policy.max_draft, verifier_mode=policy.verifier_mode)
        if gen_cfg.max_new_tokens <= 0:
            return SpeculativeDecodeResult([], stats)
        if gen_cfg.temperature > 0 or gen_cfg.top_k > 0:
            stats.fallback_reason = "non_greedy_sampling"
            return SpeculativeDecodeResult(list(self.generate_ids(prompt_ids, gen_cfg, state, resolved.features)), stats)
        features = resolved.features
        if self.speculative_proposer is None:
            stats.fallback_reason = "speculative_proposer_unavailable"
            return SpeculativeDecodeResult(
                self._generate_ids_greedy_gpu_plain(prompt_ids, gen_cfg, state=state, features=features),
                stats,
            )
        ids = [int(t) for t in prompt_ids]
        state = self.new_state(features) if state is None else state
        logits, raw_hidden = self._prefill_with_raw_hidden(ids, state, features)
        out_buf = torch.empty((gen_cfg.max_new_tokens,), device=logits.device, dtype=torch.long)
        produced = 0
        recent_accepts: list[int] = []
        verifier = NativeNextNVerifier(
            model=self.model,
            proposer=self.speculative_proposer,
            sample_next=lambda current_logits: sample_next_tensor(current_logits, gen_cfg),
            max_draft=policy.max_draft,
            mode=policy.verifier_mode,  # type: ignore[arg-type]
            verify_tokens=self.verify_nextn_tokens,
        )
        while produced < gen_cfg.max_new_tokens:
            step = verifier.step(
                logits=logits,
                raw_hidden=raw_hidden,
                state=state,
                remaining_tokens=gen_cfg.max_new_tokens - produced,
            )
            for token in step.tokens:
                if produced >= gen_cfg.max_new_tokens:
                    break
                out_buf[produced] = token
                produced += 1
            logits = step.logits
            raw_hidden = step.raw_hidden
            if step.verified:
                recent_accepts.extend([1] * step.accepted)
                if step.rejected:
                    recent_accepts.append(0)
                stats.verifier_steps += 1
                stats.accepted_draft_tokens += step.accepted
                stats.verified_draft_tokens += step.verified
                stats.rejected_steps += step.rejected
                stats.rollback_tokens += step.rollback_tokens

            if len(recent_accepts) > policy.min_verified:
                del recent_accepts[: len(recent_accepts) - policy.min_verified]
            recent_ready = len(recent_accepts) >= policy.min_verified
            recent_rate = sum(recent_accepts) / len(recent_accepts) if recent_accepts else 0.0
            too_many_rejections = policy.max_rejections is not None and verifier.rejected >= policy.max_rejections
            poor_recent_acceptance = (
                policy.adaptive
                and verifier.verified >= policy.min_verified
                and recent_ready
                and recent_rate < policy.accept_threshold
            )
            if policy.adaptive and (too_many_rejections or poor_recent_acceptance):
                stats.fallback_reason = "max_rejections" if too_many_rejections else "accept_rate"
                prefix = out_buf[:produced].detach().cpu().tolist()
                return SpeculativeDecodeResult(
                    self._continue_ids_greedy_gpu_plain(logits, gen_cfg, state, prefix=prefix),
                    stats,
                )
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        out = out_buf[:produced].detach().cpu().tolist()
        if gen_cfg.eos_token_ids:
            eos = set(gen_cfg.eos_token_ids)
            for i, tid in enumerate(out):
                if int(tid) in eos:
                    return SpeculativeDecodeResult([int(t) for t in out[:i]], stats)
        return SpeculativeDecodeResult([int(t) for t in out[: gen_cfg.max_new_tokens]], stats)

    def completion_tokens(
        self,
        messages: Sequence[dict[str, Any]],
        gen_cfg: GenerationConfig,
        features: RuntimeFeatures | None = None,
    ):
        yield from self.completion_tokens_from_ids(self.encode_messages(messages), gen_cfg, features)

    def completion_tokens_from_ids(
        self,
        prompt_ids: Sequence[int],
        gen_cfg: GenerationConfig,
        features: RuntimeFeatures | None = None,
    ):
        with self.lock, torch.no_grad():
            with self.pooled_state(features) as state:
                decoder = StreamingTextDecoder(self.tokenizer, skip_special_tokens=False)
                for tid in self.generate_ids(prompt_ids, gen_cfg, state=state, features=features):
                    text = decoder.push(tid)
                    if text:
                        yield tid, text
                text = decoder.flush()
                if text:
                    yield -1, text

    def completion_ids_greedy_gpu(
        self,
        messages: Sequence[dict[str, Any]],
        gen_cfg: GenerationConfig,
        features: RuntimeFeatures | None = None,
    ) -> list[int]:
        return self.completion_ids_greedy_gpu_from_ids(self.encode_messages(messages), gen_cfg, features)

    def completion_ids_greedy_gpu_from_ids(
        self,
        prompt_ids: Sequence[int],
        gen_cfg: GenerationConfig,
        features: RuntimeFeatures | None = None,
    ) -> list[int]:
        with self.lock, torch.no_grad():
            with self.pooled_state(features) as state:
                return self.generate_ids_greedy_gpu(prompt_ids, gen_cfg, state=state, features=features)
