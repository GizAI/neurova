from __future__ import annotations

import torch

from qwenburst.episodic_memory import EpisodicMemory
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
