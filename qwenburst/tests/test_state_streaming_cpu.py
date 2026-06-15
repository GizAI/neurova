from __future__ import annotations

import torch

from qwenburst.config import Qwen36_27B_TextConfig
from qwenburst.state import DecodeState
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
