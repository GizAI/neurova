from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from pathlib import Path

import torch

import qwenburst.adapters  # noqa: F401 - registers built-in adapters
from qwenburst.core.adapter import AdapterDescriptor, adapter_registry
from qwenburst.core.features import RuntimeFeatures
from qwenburst.core.runtime import GenerationConfig, RuntimeEngine
from qwenburst.model import QwenBurstModel
from qwenburst.speculative_batch import DecodeRequestState, build_decode_batch_plan


@dataclass
class ToyState:
    pos: int = 0
    profile: str = "stateful"

    def reset(self, *, reset_attention: bool = True) -> None:
        self.pos = 0


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


class CountingToyAdapter(ToyAdapter):
    def __init__(self):
        self.allocations = 0

    def allocate_state(self, cfg, *, recent_window: int, device: str, features):
        self.allocations += 1
        return super().allocate_state(cfg, recent_window=recent_window, device=device, features=features)


class NativeBatchToyModel(ToyModel):
    def __init__(self):
        super().__init__()
        self.batch_calls = 0

    def forward_batch(self, plan, states, *, return_logits=True):
        self.batch_calls += 1
        return [torch.full((8,), 1000.0 + i) for i, _ in enumerate(states)]

    def forward_batch_logits(self, plan, states):
        self.batch_calls += 1
        return [[torch.full((8,), 1000.0 + i)] for i, _ in enumerate(states)]


class NativeBatchToyAdapter(ToyAdapter):
    def create_model(self, *, qb_model: Path, cfg, device: str, weight_device: str, cpu_embed: bool = False):
        return NativeBatchToyModel()


class BatchRouteProbe:
    forward_batch = QwenBurstModel.forward_batch
    forward_batch_logits = QwenBurstModel.forward_batch_logits

    def __init__(self):
        self.calls = 0

    def _forward_single_token_batch(self, input_ids, row_spans, states):
        self.calls += 1
        for state in states:
            state.pos += 1
        return [torch.full((8,), 1000.0 + i) for i, _ in enumerate(states)]


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


def test_generation_does_not_compute_unused_final_next_logits(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    out = engine.generate_ids_greedy_gpu(engine.encode_prompt("ab"), GenerationConfig(max_new_tokens=3, eos_token_ids=()))
    assert out == [2, 3, 4]
    assert engine.model.one_calls == 2


def test_completion_reuses_pooled_state_with_reset(tmp_path: Path):
    adapter = CountingToyAdapter()
    engine = RuntimeEngine(
        adapter=adapter,
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    cfg = GenerationConfig(max_new_tokens=2, eos_token_ids=())
    first = engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], cfg)
    second = engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], cfg)
    assert first == [2, 3]
    assert second == [2, 3]
    assert adapter.allocations == 1


def test_state_pool_size_zero_disables_pool_retention(tmp_path: Path):
    adapter = CountingToyAdapter()
    engine = RuntimeEngine(
        adapter=adapter,
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        max_state_pool_size=0,
    )
    cfg = GenerationConfig(max_new_tokens=2, eos_token_ids=())
    engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], cfg)
    engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], cfg)
    assert adapter.allocations == 2
    assert engine.state_pool_summary()["pooled_states"] == 0


def test_completion_can_disable_state_pool(tmp_path: Path):
    adapter = CountingToyAdapter()
    engine = RuntimeEngine(
        adapter=adapter,
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("stateful").with_overrides(state_pool=False),
    )
    cfg = GenerationConfig(max_new_tokens=2, eos_token_ids=())
    first = engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], cfg)
    second = engine.completion_ids_greedy_gpu([{"role": "user", "content": "ab"}], cfg)
    assert first == [2, 3]
    assert second == [2, 3]
    assert adapter.allocations == 2


def test_gpu_sampling_flag_can_use_reference_sampling_path(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
        features=RuntimeFeatures.from_profile("stateful").with_overrides(gpu_sampling=False),
    )
    out = engine.completion_ids_greedy_gpu(
        [{"role": "user", "content": "ab"}],
        GenerationConfig(max_new_tokens=2, eos_token_ids=()),
    )
    assert out == [2, 3]


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


def test_runtime_engine_forward_batch_uses_decode_batch_plan(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    states = [engine.new_state(), engine.new_state()]
    rows = [
        DecodeRequestState("a", 0, [1, 2], computed_tokens=0),
        DecodeRequestState("b", 1, [3], computed_tokens=0),
    ]
    plan = build_decode_batch_plan(rows, scheduled_tokens_per_request=[2, 1])

    logits = engine.forward_batch(plan, states)

    assert len(logits) == 2
    assert [state.pos for state in states] == [2, 1]
    assert [int(torch.argmax(row).item()) for row in logits if row is not None] == [2, 1]
    assert engine.model.block_calls == 1
    assert engine.model.one_calls == 1


def test_runtime_engine_forward_batch_prefers_native_model_batch(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=NativeBatchToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    states = [engine.new_state(), engine.new_state()]
    rows = [
        DecodeRequestState("a", 0, [1], computed_tokens=0),
        DecodeRequestState("b", 1, [2], computed_tokens=0),
    ]
    plan = build_decode_batch_plan(rows)

    logits = engine.forward_batch(plan, states)

    assert engine.model.batch_calls == 1
    assert len(logits) == 2
    assert torch.equal(logits[0], torch.full((8,), 1000.0))
    assert torch.equal(logits[1], torch.full((8,), 1001.0))


def test_runtime_engine_forward_batch_logits_returns_all_draft_rows(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    state = engine.new_state()
    row = DecodeRequestState("a", 0, [1, 2], computed_tokens=2, last_sampled_token=9, draft_token_ids=[3])
    plan = build_decode_batch_plan([row])

    logits_rows = engine.forward_batch_logits(plan, [state])

    assert len(logits_rows) == 1
    assert len(logits_rows[0]) == 2
    assert [int(torch.argmax(logit).item()) for logit in logits_rows[0]] == [1, 2]


def test_native_model_forward_batch_logits_prefers_row_spans_over_query_tensor(tmp_path: Path):
    engine = RuntimeEngine(
        adapter=ToyAdapter(),
        hf_model=tmp_path,
        qb_model=tmp_path,
        device="cpu",
        recent_window=16,
        weight_device="cpu",
    )
    rows = [
        DecodeRequestState("a", 0, [1, 2], computed_tokens=0),
        DecodeRequestState("b", 1, [3], computed_tokens=0),
    ]
    plan = build_decode_batch_plan(rows, scheduled_tokens_per_request=[2, 1])
    bad_plan = replace(plan, query_start_loc=torch.tensor([99, 99, 99], dtype=torch.int32))
    states = [engine.new_state(), engine.new_state()]

    logits_rows = engine.forward_batch_logits(bad_plan, states)

    assert len(logits_rows) == 2
    assert [state.pos for state in states] == [2, 1]
    assert [int(torch.argmax(row[-1]).item()) for row in logits_rows] == [2, 1]


def test_qwen_model_forward_batch_routes_single_token_rows_to_cross_request_batch():
    model = BatchRouteProbe()
    states = [ToyState(), ToyState()]
    rows = [
        DecodeRequestState("a", 0, [1], computed_tokens=0),
        DecodeRequestState("b", 1, [2], computed_tokens=0),
    ]
    plan = build_decode_batch_plan(rows)

    logits = model.forward_batch(plan, states)
    logits_rows = model.forward_batch_logits(plan, states)

    assert model.calls == 2
    assert [state.pos for state in states] == [2, 2]
    assert [float(row.max().item()) for row in logits if row is not None] == [1000.0, 1001.0]
    assert [[float(row.max().item()) for row in rows] for rows in logits_rows] == [[1000.0], [1001.0]]
