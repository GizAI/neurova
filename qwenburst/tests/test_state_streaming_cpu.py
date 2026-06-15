from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

from qwenburst.config import Qwen36_27B_TextConfig
from qwenburst.state import DecodeState
from qwenburst.state_delta import DecodeStateDelta
from qwenburst.streaming import InfiniteStreamingRuntime, InfiniteStreamPolicy


def tiny_cfg() -> Qwen36_27B_TextConfig:
    return Qwen36_27B_TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_layers=4,
        vocab_size_padded=128,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=4,
        rope_dim=4,
    )


def test_decode_state_fork_decay_and_snapshot(tmp_path):
    cfg = tiny_cfg()
    st = DecodeState.allocate(cfg, max_seq_len=4, device="cpu", dtype=torch.float16, kv_window_policy="shift")
    layer = cfg.gdn_layers[0]
    st.gdn_states[layer].fill_(2.0)
    st.gdn_conv_states[layer].fill_(4.0)
    st.pos = 7
    st.kv_len = 3

    branch = st.fork()
    branch.gdn_states[layer].fill_(9.0)
    assert float(st.gdn_states[layer][0, 0, 0]) == 2.0

    st.decay_gdn_(0.25)
    assert float(st.gdn_states[layer][0, 0, 0]) == 0.5
    assert float(st.gdn_conv_states[layer][0, 0]) == 1.0

    path = tmp_path / "state.qbstate.pt"
    st.save_snapshot(path, include_attention_kv=False)
    loaded = DecodeState.load_snapshot(path, cfg, device="cpu", dtype=torch.float16)
    assert loaded.pos == 7
    assert loaded.kv_len == 3
    assert torch.equal(loaded.gdn_states[layer], st.gdn_states[layer])
    assert torch.equal(loaded.gdn_conv_states[layer], st.gdn_conv_states[layer])


def test_attention_shift_window_keeps_recent_tokens():
    cfg = tiny_cfg()
    st = DecodeState.allocate(cfg, max_seq_len=2, device="cpu", dtype=torch.float16, kv_window_policy="shift")
    layer = cfg.attention_layers[0]
    for i in range(3):
        k = torch.full((cfg.num_key_value_heads, cfg.attention_head_dim), float(i + 1), dtype=torch.float16)
        v = torch.full_like(k, float((i + 1) * 10))
        st.append_attention_kv(layer, k, v)
        st.finish_token()
    assert st.kv_len == 2
    assert st.pos == 3
    assert torch.equal(st.attn_k[layer][0, :, 0], torch.tensor([2.0, 3.0], dtype=torch.float16))
    assert torch.equal(st.attn_v[layer][0, :, 0], torch.tensor([20.0, 30.0], dtype=torch.float16))


class DummyStreamingModel:
    def __init__(self, cfg: Qwen36_27B_TextConfig):
        self.cfg = cfg

    def forward_one(self, token: torch.Tensor, state: DecodeState, *, use_mtp: bool = False) -> torch.Tensor:
        layer = self.cfg.gdn_layers[0]
        state.gdn_states[layer][0, 0, 0].add_(token.float().item())
        state.finish_token()
        return torch.zeros(self.cfg.vocab_size_padded)


class DeterministicStateModel:
    def __init__(self, cfg: Qwen36_27B_TextConfig):
        self.cfg = cfg

    def forward_one(self, token: torch.Tensor | int, state: DecodeState, *, use_mtp: bool = False) -> torch.Tensor:
        t = float(int(token))
        gdn_layer = self.cfg.gdn_layers[0]
        state.gdn_states[gdn_layer].add_(t * 0.01)
        state.gdn_conv_states[gdn_layer][:, :-1] = state.gdn_conv_states[gdn_layer][:, 1:].clone()
        state.gdn_conv_states[gdn_layer][:, -1] = t
        if self.cfg.attention_layers:
            attn_layer = self.cfg.attention_layers[0]
            k = torch.full(
                (self.cfg.num_key_value_heads, self.cfg.attention_head_dim),
                t,
                dtype=state.dtype,
                device=state.device,
            )
            v = torch.full_like(k, t + 1000.0)
            state.append_attention_kv(attn_layer, k, v)
        state.finish_token()
        logits = torch.arange(self.cfg.vocab_size_padded, dtype=torch.float32, device=state.device)
        return logits + state.gdn_states[gdn_layer].sum().float() * 1e-4


def assert_state_equal(a: DecodeState, b: DecodeState) -> None:
    assert a.pos == b.pos
    assert a.kv_len == b.kv_len
    for layer in a.gdn_states:
        assert torch.equal(a.gdn_states[layer], b.gdn_states[layer])
        assert torch.equal(a.gdn_conv_states[layer], b.gdn_conv_states[layer])
    for layer in a.attn_k:
        assert torch.equal(a.attn_k[layer], b.attn_k[layer])
        assert torch.equal(a.attn_v[layer], b.attn_v[layer])


def test_infinite_stream_runtime_ingest_snapshot(tmp_path):
    cfg = tiny_cfg()
    rt = InfiniteStreamingRuntime.create(
        DummyStreamingModel(cfg),
        device="cpu",
        dtype=torch.float16,
        policy=InfiniteStreamPolicy(recent_window_tokens=4, kv_window_policy="shift", snapshot_dir=tmp_path),
    )
    rt.ingest_tokens([1, 2, 3], boundary=True)
    assert rt.stats.tokens_ingested == 3
    assert rt.state.pos == 3
    layer = cfg.gdn_layers[0]
    assert float(rt.state.gdn_states[layer][0, 0, 0]) == 6.0
    path = rt.write_snapshot(include_attention_kv=False)
    assert path.exists()


def test_chunked_ingest_matches_one_shot_state():
    cfg = tiny_cfg()
    tokens = list(range(1, 12))
    one = InfiniteStreamingRuntime.create(
        DeterministicStateModel(cfg),
        device="cpu",
        dtype=torch.float16,
        policy=InfiniteStreamPolicy(recent_window_tokens=8, kv_window_policy="ring"),
    )
    chunked = InfiniteStreamingRuntime.create(
        DeterministicStateModel(cfg),
        device="cpu",
        dtype=torch.float16,
        policy=InfiniteStreamPolicy(recent_window_tokens=8, kv_window_policy="ring"),
    )
    one.ingest_tokens(tokens)
    for part in (tokens[:3], tokens[3:7], tokens[7:]):
        chunked.ingest_tokens(part)
    assert_state_equal(one.state, chunked.state)


def test_decode_state_carry_changes_next_logits():
    cfg = tiny_cfg()
    model = DeterministicStateModel(cfg)
    state0 = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    state1 = state0.fork()
    logits0 = model.forward_one(7, state0)
    for tok in [1, 2, 3, 4, 5]:
        model.forward_one(tok, state1)
    logits1 = model.forward_one(7, state1)
    assert not torch.equal(logits0, logits1)


def test_ring_kv_boundaries_63_64_65_and_snapshot_roundtrip(tmp_path):
    cfg = Qwen36_27B_TextConfig(num_layers=4, num_key_value_heads=1, attention_head_dim=2)
    layer = cfg.attention_layers[0]
    for n in (63, 64, 65, 200):
        st = DecodeState.allocate(cfg, max_seq_len=64, device="cpu", dtype=torch.float16, kv_window_policy="ring")
        for i in range(n):
            k = torch.full((1, 2), float(i), dtype=torch.float16)
            v = torch.full((1, 2), float(i + 1000), dtype=torch.float16)
            st.append_attention_kv(layer, k, v)
            st.finish_token()
        assert st.kv_len == min(n, 64)
        assert st.pos == n
        if n >= 64:
            assert st.attention_write_index() == n % 64
        path = tmp_path / f"ring_{n}.pt"
        st.save_snapshot(path, include_attention_kv=True)
        loaded = DecodeState.load_snapshot(path, cfg, device="cpu", dtype=torch.float16)
        assert_state_equal(st, loaded)


def test_state_fork_branch_and_parent_isolation():
    cfg = tiny_cfg()
    base = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    model = DeterministicStateModel(cfg)
    model.forward_one(1, base)
    branch_a = base.fork()
    branch_b = base.fork()
    model.forward_one(10, branch_a)
    model.forward_one(20, branch_b)
    assert base.pos == 1
    assert branch_a.pos == 2
    assert branch_b.pos == 2
    assert not torch.equal(branch_a.gdn_states[cfg.gdn_layers[0]], branch_b.gdn_states[cfg.gdn_layers[0]])


def test_decay_reset_and_state_delta_apply():
    cfg = tiny_cfg()
    base = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    direct = base.fork()
    model = DeterministicStateModel(cfg)
    before = direct.fork()
    for tok in [3, 4, 5]:
        model.forward_one(tok, direct)
    delta = DecodeStateDelta.between(before, direct)
    applied = before.fork()
    delta.apply_to(applied)
    assert_state_equal(direct, applied)

    layer = cfg.gdn_layers[0]
    norm = direct.gdn_states[layer].norm().item()
    direct.decay_gdn_(0.5)
    assert abs(direct.gdn_states[layer].norm().item() - norm * 0.5) < 1e-3
    direct.reset()
    assert direct.gdn_states[layer].norm().item() == 0.0
    assert direct.gdn_conv_states[layer].norm().item() == 0.0
    assert direct.pos == 0
    assert direct.kv_len == 0


def test_state_only_warm_boot_matches_continue(tmp_path):
    cfg = tiny_cfg()
    model = DeterministicStateModel(cfg)
    st = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    for tok in range(20):
        model.forward_one(tok, st)
    saved = st.fork()
    path = tmp_path / "warm.pt"
    st.save_snapshot(path, include_attention_kv=True)
    loaded = DecodeState.load_snapshot(path, cfg, device="cpu", dtype=torch.float16)
    assert_state_equal(saved, loaded)
    logits_a = model.forward_one(99, saved)
    logits_b = model.forward_one(99, loaded)
    assert torch.equal(logits_a, logits_b)
    assert_state_equal(saved, loaded)


def test_simulated_speculative_reject_rolls_back_state():
    cfg = tiny_cfg()
    model = DeterministicStateModel(cfg)
    base = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    for tok in [1, 2]:
        model.forward_one(tok, base)
    proposed = [3, 4, 5, 6]
    branch = base.fork()
    for tok in proposed:
        model.forward_one(tok, branch)
    accepted = base.fork()
    for tok in proposed[:2]:
        model.forward_one(tok, accepted)
    base.copy_from_(accepted)
    assert base.pos == 4
    assert base.kv_len == 4
    assert not torch.equal(base.gdn_states[cfg.gdn_layers[0]], branch.gdn_states[cfg.gdn_layers[0]])


def test_stateful_multi_turn_chat_uses_same_decode_state():
    cfg = tiny_cfg()
    model = DeterministicStateModel(cfg)
    state = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    codename = [65, 76, 80, 72, 65, 42]
    for tok in codename:
        model.forward_one(tok, state)
    remembered = state.fork()
    logits_with_history = model.forward_one(7, state)

    fresh = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    logits_without_history = model.forward_one(7, fresh)

    assert remembered.pos == len(codename)
    assert not torch.equal(logits_with_history, logits_without_history)


def test_long_streaming_memory_budget_is_bounded():
    cfg = tiny_cfg()
    model = DeterministicStateModel(cfg)
    rt = InfiniteStreamingRuntime.create(
        model,
        device="cpu",
        dtype=torch.float16,
        policy=InfiniteStreamPolicy(recent_window_tokens=64, kv_window_policy="ring"),
    )
    initial_bytes = rt.state.total_bytes
    for total in (1_000, 10_000, 100_000):
        start = rt.stats.tokens_ingested
        rt.ingest_tokens((i % 127 for i in range(total - start)))
        assert rt.stats.tokens_ingested == total
        assert rt.state.pos == total
        assert rt.state.kv_len == 64
        assert rt.state.total_bytes == initial_bytes
    # A million-token stream must not require a bigger state allocation.  The
    # exact million-token throughput benchmark is kept out of the unit suite.
    rt.state.pos = 1_000_000
    rt.state.kv_len = 64
    assert rt.state.total_bytes == initial_bytes


def test_cuda_graph_state_reuse_contract_is_not_enabled_yet():
    import pytest

    pytest.skip("CUDA Graph decode is not implemented; add eager-vs-graph state parity when the graph path lands")


def test_process_level_persistence_load_continue_matches(tmp_path):
    snapshot = tmp_path / "state.pt"
    env = os.environ.copy()
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    script = f"""
import json
import torch
from qwenburst.config import Qwen36_27B_TextConfig
from qwenburst.state import DecodeState

cfg = Qwen36_27B_TextConfig(
    hidden_size=32,
    intermediate_size=64,
    num_layers=4,
    vocab_size_padded=128,
    linear_num_value_heads=2,
    linear_num_key_heads=1,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=3,
    num_attention_heads=2,
    num_key_value_heads=1,
    attention_head_dim=4,
    rope_dim=4,
)

def step(tok, state):
    t = float(int(tok))
    layer = cfg.gdn_layers[0]
    state.gdn_states[layer].add_(t * 0.01)
    state.gdn_conv_states[layer][:, :-1] = state.gdn_conv_states[layer][:, 1:].clone()
    state.gdn_conv_states[layer][:, -1] = t
    attn = cfg.attention_layers[0]
    k = torch.full((cfg.num_key_value_heads, cfg.attention_head_dim), t, dtype=state.dtype)
    v = torch.full_like(k, t + 1000.0)
    state.append_attention_kv(attn, k, v)
    state.finish_token()
    return float(state.gdn_states[layer].sum().item())

state = DecodeState.allocate(cfg, max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
for tok in range(20):
    step(tok, state)
state.save_snapshot(r"{snapshot}", include_attention_kv=True)
direct = state.fork()
direct_logit = step(99, direct)
print(json.dumps({{"direct": direct_logit, "pos": direct.pos, "kv_len": direct.kv_len}}))
"""
    proc_a = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True, env=env)
    expected = json.loads(proc_a.stdout.strip())

    load_script = f"""
import json
import torch
from qwenburst.config import Qwen36_27B_TextConfig
from qwenburst.state import DecodeState

cfg = Qwen36_27B_TextConfig(
    hidden_size=32,
    intermediate_size=64,
    num_layers=4,
    vocab_size_padded=128,
    linear_num_value_heads=2,
    linear_num_key_heads=1,
    linear_key_head_dim=4,
    linear_value_head_dim=4,
    linear_conv_kernel_dim=3,
    num_attention_heads=2,
    num_key_value_heads=1,
    attention_head_dim=4,
    rope_dim=4,
)

def step(tok, state):
    t = float(int(tok))
    layer = cfg.gdn_layers[0]
    state.gdn_states[layer].add_(t * 0.01)
    state.gdn_conv_states[layer][:, :-1] = state.gdn_conv_states[layer][:, 1:].clone()
    state.gdn_conv_states[layer][:, -1] = t
    attn = cfg.attention_layers[0]
    k = torch.full((cfg.num_key_value_heads, cfg.attention_head_dim), t, dtype=state.dtype)
    v = torch.full_like(k, t + 1000.0)
    state.append_attention_kv(attn, k, v)
    state.finish_token()
    return float(state.gdn_states[layer].sum().item())

loaded = DecodeState.load_snapshot(r"{snapshot}", cfg, device="cpu", dtype=torch.float16)
continued = step(99, loaded)
print(json.dumps({{"direct": continued, "pos": loaded.pos, "kv_len": loaded.kv_len}}))
"""
    proc_b = subprocess.run([sys.executable, "-c", load_script], check=True, capture_output=True, text=True, env=env)
    actual = json.loads(proc_b.stdout.strip())
    assert actual == expected
