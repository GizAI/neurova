from __future__ import annotations

import threading
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch

from .adapter import ModelAdapter
from .features import RuntimeFeatures, RuntimePlan, resolve_runtime_plan
from ..speculative_batch import DecodeBatchPlan
from ..ops import cuda_ops
from ..qwen_mtp import native_mtp1_proposer_for_model
from ..speculation import SpeculativeProposer
from ..speculative_verifier import NativeNextNVerifier
from ..tuning import speculative_verifier_mode


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 0
    eos_token_ids: tuple[int, ...] = ()


def sample_next(logits: torch.Tensor, cfg: GenerationConfig) -> int:
    if cfg.temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scores = logits.float() / max(cfg.temperature, 1e-6)
    if cfg.top_k and cfg.top_k > 0:
        vals, idx = torch.topk(scores, min(cfg.top_k, scores.numel()))
        probs = torch.softmax(vals, dim=-1)
        return int(idx[torch.multinomial(probs, 1)].item())
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def sample_next_tensor(logits: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
    if cfg.temperature <= 0:
        if logits.device.type == "cuda":
            return cuda_ops().argmax(logits.contiguous()).reshape(()).to(device=logits.device, dtype=torch.long)
        return torch.argmax(logits, dim=-1).reshape(()).to(dtype=torch.long)
    token = sample_next(logits, cfg)
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
        max_state_pool_size: int = 1,
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
        self.speculative_proposer: SpeculativeProposer | None = native_mtp1_proposer_for_model(self.model)

    def resolve_plan(self, features: RuntimeFeatures | None = None) -> RuntimePlan:
        return resolve_runtime_plan(features or self.features, self.adapter.descriptor.capabilities)

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
        """Execute a vLLM-style decode batch plan.

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
        draft tokens return one logit row per scheduled token, matching vLLM's
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
        logits = self.prefill(prompt_ids, state, features)
        next_id = sample_next(logits, gen_cfg)
        for i in range(gen_cfg.max_new_tokens):
            if gen_cfg.eos_token_ids and next_id in gen_cfg.eos_token_ids:
                break
            yield next_id
            if i == gen_cfg.max_new_tokens - 1:
                break
            logits = self.forward_one(next_id, state, return_logits=True)
            next_id = sample_next(logits, gen_cfg)

    @torch.no_grad()
    def generate_ids_greedy_gpu(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
    ) -> list[int]:
        if gen_cfg.max_new_tokens <= 0:
            return []
        features = self.resolve_plan(features).effective
        if gen_cfg.temperature > 0 or gen_cfg.top_k > 0 or not features.gpu_sampling:
            return list(self.generate_ids(prompt_ids, gen_cfg, state, features))
        if features.speculative_decoding:
            return self.generate_ids_native_mtp1_speculative(prompt_ids, gen_cfg, state=state, features=features)
        return self._generate_ids_greedy_gpu_plain(prompt_ids, gen_cfg, state=state, features=features)

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
        next_token = sample_next_tensor(logits, gen_cfg) if first_token is None else first_token
        out = torch.empty((remaining,), device=next_token.device, dtype=torch.long)
        produced = 0
        for i in range(remaining):
            out[i] = next_token
            produced += 1
            if i == remaining - 1:
                break
            logits = self.forward_one(next_token, state, return_logits=True)
            next_token = sample_next_tensor(logits, gen_cfg)
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

    @torch.no_grad()
    def generate_ids_native_mtp1_speculative(
        self,
        prompt_ids: Iterable[int],
        gen_cfg: GenerationConfig,
        state: Any | None = None,
        features: RuntimeFeatures | None = None,
        *,
        max_draft: int = 1,
        min_verified: int = 1,
        min_accept_rate: float = 1.00,
    ) -> list[int]:
        if gen_cfg.max_new_tokens <= 0:
            return []
        if gen_cfg.temperature > 0 or gen_cfg.top_k > 0:
            return list(self.generate_ids(prompt_ids, gen_cfg, state, features))
        features = self.resolve_plan(features).effective
        if self.speculative_proposer is None or self.speculative_proposer.method != "native_mtp1":
            return self._generate_ids_greedy_gpu_plain(prompt_ids, gen_cfg, state=state, features=features)
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
            max_draft=max_draft,
            mode=speculative_verifier_mode(),  # type: ignore[arg-type]
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

            if len(recent_accepts) > min_verified:
                del recent_accepts[: len(recent_accepts) - min_verified]
            recent_ready = len(recent_accepts) >= min_verified
            recent_rate = sum(recent_accepts) / len(recent_accepts) if recent_accepts else 0.0
            if verifier.verified >= min_verified and recent_ready and recent_rate < min_accept_rate:
                prefix = out_buf[:produced].detach().cpu().tolist()
                return self._continue_ids_greedy_gpu_plain(logits, gen_cfg, state, prefix=prefix)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        out = out_buf[:produced].detach().cpu().tolist()
        if gen_cfg.eos_token_ids:
            eos = set(gen_cfg.eos_token_ids)
            for i, tid in enumerate(out):
                if int(tid) in eos:
                    return [int(t) for t in out[:i]]
        return [int(t) for t in out[: gen_cfg.max_new_tokens]]

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
                for tid in self.generate_ids(prompt_ids, gen_cfg, state=state, features=features):
                    yield tid, self.tokenizer.decode([tid], skip_special_tokens=False)

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
