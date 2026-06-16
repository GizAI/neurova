from __future__ import annotations

import torch

from langburst.adapters.qwen36_impl.config import Qwen36_27B_TextConfig
from langburst.core.runtime import GenerationConfig, sample_next_tensor
from langburst.adapters.qwen36_tools.graph import inspect_decode1_graph_safety, verify_graph_safe_argmax
from langburst.adapters.qwen36_impl.state import DecodeState


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


def test_decode1_graph_audit_blocks_current_python_state_contract():
    state = DecodeState.allocate(tiny_cfg(), max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    report = inspect_decode1_graph_safety(model=None, state=state, gen_cfg=GenerationConfig(max_new_tokens=1))
    assert not report.graph_ready
    assert report.greedy_argmax_device_safe
    assert not report.device_position_counters
    assert not report.ring_kv_device_indexing
    assert any("Python counters" in blocker for blocker in report.blockers)
    assert any("ring KV" in blocker for blocker in report.blockers)


def test_decode1_graph_audit_requires_greedy_first():
    state = DecodeState.allocate(tiny_cfg(), max_seq_len=8, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    report = inspect_decode1_graph_safety(
        model=None,
        state=state,
        gen_cfg=GenerationConfig(max_new_tokens=1, temperature=0.7, top_k=16),
    )
    assert not report.greedy_argmax_device_safe
    assert any("greedy-only" in blocker for blocker in report.blockers)


def test_sample_next_tensor_keeps_greedy_token_as_tensor():
    logits = torch.tensor([0.0, 4.0, 1.0], dtype=torch.float32)
    token = sample_next_tensor(logits, GenerationConfig(max_new_tokens=1, temperature=0.0, top_k=0))
    assert torch.is_tensor(token)
    assert token.shape == torch.Size([])
    assert int(token) == 1
    assert verify_graph_safe_argmax()
