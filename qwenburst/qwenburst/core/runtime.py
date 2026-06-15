from __future__ import annotations

import threading
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch

from .adapter import ModelAdapter
from .features import RuntimeFeatures
from ..ops import cuda_ops


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
    ) -> None:
        self.adapter = adapter
        self.hf_model = Path(hf_model)
        self.qb_model = Path(qb_model)
        self.device = device
        self.recent_window = int(recent_window)
        self.weight_device = weight_device
        self.cpu_embed = bool(cpu_embed)
        self.features = features or RuntimeFeatures.from_profile("stateful")
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

    def new_state(self, features: RuntimeFeatures | None = None) -> Any:
        features = features or self.features
        return self.adapter.allocate_state(
            self.cfg,
            recent_window=self.recent_window,
            device=self.device,
            features=features,
        )

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
    def _prefill_token_loop(self, ids: list[int], state: Any) -> torch.Tensor:
        logits: torch.Tensor | None = None
        for i, tid in enumerate(ids):
            logits = self.forward_one(tid, state, return_logits=(i == len(ids) - 1))
        assert logits is not None
        return logits

    def _prefill_block(self, ids: list[int], state: Any, features: RuntimeFeatures) -> torch.Tensor:
        forward_block = getattr(self.model, "forward_block", None)
        if forward_block is None:
            return self._prefill_token_loop(ids, state)
        logits: torch.Tensor | None = None
        chunk_size = max(1, int(features.prefill_chunk_size))
        supports_logits_mode = "logits_mode" in inspect.signature(forward_block).parameters
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start : start + chunk_size]
            is_last = start + chunk_size >= len(ids)
            if supports_logits_mode:
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
        features = features or self.features
        logits = self.prefill(prompt_ids, state, features)
        next_id = sample_next(logits, gen_cfg)
        for _ in range(gen_cfg.max_new_tokens):
            if gen_cfg.eos_token_ids and next_id in gen_cfg.eos_token_ids:
                break
            yield next_id
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
        if gen_cfg.temperature > 0 or gen_cfg.top_k > 0:
            return list(self.generate_ids(prompt_ids, gen_cfg, state, features))
        features = features or self.features
        state = self.new_state(features) if state is None else state
        logits = self.prefill(prompt_ids, state, features)
        next_token = sample_next_tensor(logits, gen_cfg)
        out = torch.empty((gen_cfg.max_new_tokens,), device=next_token.device, dtype=torch.long)
        produced = 0
        for i in range(gen_cfg.max_new_tokens):
            out[i] = next_token
            produced += 1
            logits = self.forward_one(next_token, state, return_logits=True)
            next_token = sample_next_tensor(logits, gen_cfg)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        ids = out[:produced].detach().cpu().tolist()
        if gen_cfg.eos_token_ids:
            eos = set(gen_cfg.eos_token_ids)
            for i, tid in enumerate(ids):
                if tid in eos:
                    return [int(t) for t in ids[:i]]
        return [int(t) for t in ids]

    def completion_tokens(
        self,
        messages: Sequence[dict[str, Any]],
        gen_cfg: GenerationConfig,
        features: RuntimeFeatures | None = None,
    ):
        prompt_ids = self.encode_messages(messages)
        with self.lock, torch.no_grad():
            for tid in self.generate_ids(prompt_ids, gen_cfg, features=features):
                yield tid, self.tokenizer.decode([tid], skip_special_tokens=False)

    def completion_ids_greedy_gpu(
        self,
        messages: Sequence[dict[str, Any]],
        gen_cfg: GenerationConfig,
        features: RuntimeFeatures | None = None,
    ) -> list[int]:
        with self.lock, torch.no_grad():
            return self.generate_ids_greedy_gpu(self.encode_messages(messages), gen_cfg, features=features)
