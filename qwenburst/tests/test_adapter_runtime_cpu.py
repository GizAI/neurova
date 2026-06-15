from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import torch

import qwenburst.adapters  # noqa: F401 - registers built-in adapters
from qwenburst.core.adapter import AdapterDescriptor, adapter_registry
from qwenburst.core.features import RuntimeFeatures
from qwenburst.core.runtime import GenerationConfig, RuntimeEngine


@dataclass
class ToyState:
    pos: int = 0
    profile: str = "stateful"


class ToyTokenizer:
    eos_token_id = 7
    pad_token_id = 7

    def encode(self, text: str):
        return [ord(c) % 8 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(int(i)) for i in ids)


class ToyModel:
    def __init__(self):
        self.table = [1, 2, 3, 4]
        self.block_calls = 0
        self.one_calls = 0

    def forward_one(self, token, state, *, use_mtp=False, return_logits=True):
        self.one_calls += 1
        idx = min(state.pos, len(self.table) - 1)
        logits = torch.full((8,), -1000.0)
        logits[self.table[idx]] = 1000.0
        state.pos += 1
        return logits

    def forward_block(self, tokens, state, *, return_logits=True, logits_mode="all", commit=True):
        self.block_calls += 1
        logits = []
        for i, _ in enumerate(tokens):
            idx = min(state.pos + i, len(self.table) - 1)
            row = torch.full((8,), -1000.0)
            row[self.table[idx]] = 1000.0
            if return_logits and (logits_mode == "all" or i == len(tokens) - 1):
                logits.append(row)
        state.pos += len(tokens)
        return SimpleNamespace(logits=logits, state=state, hidden_taps=[], raw_hiddens=[])


class ToyAdapter:
    descriptor = AdapterDescriptor(
        adapter_id="toy",
        family="toy-decoder",
        default_model_name="toy-model",
        supports_state=True,
    )

    def load_config(self, hf_model: Path):
        return {"vocab": 8}

    def load_tokenizer(self, hf_model: Path):
        return ToyTokenizer()

    def create_model(self, *, qb_model: Path, cfg, device: str, weight_device: str, cpu_embed: bool = False):
        return ToyModel()

    def allocate_state(self, cfg, *, recent_window: int, device: str, features):
        return ToyState(profile=features.profile)

    def encode_messages(self, tokenizer, messages):
        return tokenizer.encode("\n".join(str(m["content"]) for m in messages))

    def encode_prompt(self, tokenizer, prompt: str, system: str | None = None):
        return tokenizer.encode((system + "\n" if system else "") + prompt)

    def eos_token_ids(self, tokenizer):
        return (tokenizer.eos_token_id,)


def test_builtin_qwen_adapter_is_registered():
    assert any(d.adapter_id == "qwen36" for d in adapter_registry.list())


def test_runtime_engine_uses_adapter_boundary(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    ids = engine.encode_prompt("ab")
    out = engine.generate_ids_greedy_gpu(ids, GenerationConfig(max_new_tokens=3, eos_token_ids=()))
    assert out == [2, 3, 4]
    assert engine.completion_ids_greedy_gpu(
        [{"role": "user", "content": "ab"}],
        GenerationConfig(max_new_tokens=2, eos_token_ids=()),
    ) == [2, 3]


def test_runtime_engine_accepts_per_request_features(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("stateful"),
    )
    state = engine.new_state(RuntimeFeatures.from_profile("original"))
    assert state.profile == "original"


def test_runtime_engine_prefill_uses_block_path_by_default(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("original").with_overrides(prefill_chunk_size=2),
    )
    state = engine.new_state()
    logits = engine.prefill([1, 2, 3], state)
    assert int(torch.argmax(logits).item()) == 3
    assert state.pos == 3
    assert engine.model.block_calls == 2
    assert engine.model.one_calls == 0


def test_runtime_engine_prefill_can_disable_block_path(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("original").with_overrides(block_prefill=False),
    )
    state = engine.new_state()
    logits = engine.prefill([1, 2, 3], state)
    assert int(torch.argmax(logits).item()) == 3
    assert state.pos == 3
    assert engine.model.block_calls == 0
    assert engine.model.one_calls == 3
