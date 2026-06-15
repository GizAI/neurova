from __future__ import annotations

import torch

from qwenburst.episodic_memory import EpisodicMemory
from qwenburst.config import Qwen36_27B_TextConfig
from qwenburst.state import DecodeState
from qwenburst.ttt_sidecar import TTTSidecarConfig, TTTSidecarMemory


def test_episodic_memory_search_and_roundtrip(tmp_path):
    mem = EpisodicMemory()
    mem.add_text("Gated DeltaNet recurrent state stores compressed long context", record_id="gdn")
    mem.add_text("Paged KV cache stores recent exact attention tokens", record_id="kv")
    hits = mem.search("compressed recurrent context", top_k=1)
    assert hits[0].record_id == "gdn"
    path = tmp_path / "memory.json"
    mem.save(path)
    loaded = EpisodicMemory.load(path)
    assert loaded.search("recent exact attention", top_k=1)[0].record_id == "kv"


def test_episodic_memory_exact_fact_topk():
    mem = EpisodicMemory()
    for i in range(100):
        mem.add_text(f"user_{i:03d} UUID=abc-{i:03d} phone=555-01{i:02d}", record_id=f"fact-{i:03d}")
    hits = mem.search("what is UUID abc 042 phone", top_k=5)
    assert any(h.record_id == "fact-042" for h in hits)
    assert hits[0].record_id == "fact-042"


def test_ttt_sidecar_update_read_and_state_dict():
    torch.manual_seed(0)
    cfg = TTTSidecarConfig(hidden_size=32, memory_rank=8, dtype=torch.float16)
    mem = TTTSidecarMemory(cfg, device="cpu")
    hidden = torch.randn(5, 32, dtype=torch.float16)
    before = mem.read(hidden[-1]).clone()
    mem.update(hidden)
    after = mem.read(hidden[-1])
    assert mem.updates == 4
    assert after.shape == (32,)
    assert not torch.equal(before, after)

    state = mem.state_dict()
    mem2 = TTTSidecarMemory(cfg, device="cpu")
    mem2.load_state_dict(state)
    assert torch.allclose(mem.read(hidden[-1]).float(), mem2.read(hidden[-1]).float(), atol=1e-5)


def test_ttt_sidecar_does_not_mutate_decode_state():
    cfg_model = Qwen36_27B_TextConfig(
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
    st_off = DecodeState.allocate(cfg_model, max_seq_len=4, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    st_on = st_off.fork()
    layer = cfg_model.gdn_layers[0]
    for tok in [1, 2, 3]:
        st_off.gdn_states[layer].add_(tok * 0.1)
        st_on.gdn_states[layer].add_(tok * 0.1)

    ttt = TTTSidecarMemory(TTTSidecarConfig(hidden_size=32, memory_rank=8, dtype=torch.float16), device="cpu")
    before = ttt.memory.clone()
    ttt.update(torch.randn(6, 32, dtype=torch.float16))
    assert torch.equal(st_off.gdn_states[layer], st_on.gdn_states[layer])
    assert not torch.equal(before, ttt.memory)
