from __future__ import annotations

from types import SimpleNamespace

import torch

from langburst.adapters.qwen36_impl.config import Qwen36_27B_TextConfig
from langburst.adapters.qwen36 import Qwen36Adapter
from langburst.core.features import RuntimeFeatures
from langburst.core.state_store import BatchStateStore
from langburst.adapters.qwen36_impl.state import DecodeStateArena


def tiny_qwen_cfg() -> Qwen36_27B_TextConfig:
    return Qwen36_27B_TextConfig(
        hidden_size=8,
        intermediate_size=16,
        num_layers=2,
        layer_types=("gdn", "attn"),
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=4,
        rope_dim=4,
        vocab_size_padded=16,
    )


def test_decode_state_arena_returns_slot_views_and_recycles():
    arena = DecodeStateArena(cfg=tiny_qwen_cfg(), max_seq_len=4, num_slots=2, device="cpu")

    slot, state = arena.allocate()
    state.gdn_states[0].fill_(3)
    state.attn_k[1].fill_(5)

    assert state.arena is arena
    assert state.arena_slot == slot
    assert arena.gdn_states[0][slot].sum().item() > 0
    assert arena.attn_k[1][slot].sum().item() > 0

    arena.release(slot)

    assert arena.summary() == {
        "num_slots": 2,
        "active_slots": 0,
        "free_slots": 2,
        "max_seq_len": 4,
        "kv_num_blocks": 0,
        "kv_block_size": 0,
    }
    assert torch.count_nonzero(arena.gdn_states[0][slot]).item() == 0
    assert torch.count_nonzero(arena.attn_k[1][slot]).item() == 0


def test_batch_state_store_uses_arena_for_qwen_like_engine():
    def fail_new_state(_features):
        raise AssertionError("arena-backed store should not allocate per-request state")

    engine = SimpleNamespace(
        cfg=tiny_qwen_cfg(),
        recent_window=4,
        device="cpu",
        new_state=fail_new_state,
        create_state_arena=lambda *, features, max_slots, kv_num_blocks=None, kv_block_size=None: Qwen36Adapter().create_state_arena(
            tiny_qwen_cfg(),
            max_seq_len=4,
            num_slots=max_slots,
            device="cpu",
            features=features,
            kv_num_blocks=kv_num_blocks,
            kv_block_size=kv_block_size,
        ),
    )
    store = BatchStateStore(engine=engine, features=RuntimeFeatures.from_profile("stateful"), max_slots=2)

    first = store.allocate(10)
    second = store.allocate(11)

    assert store.arena_summary() == {
        "num_slots": 2,
        "active_slots": 2,
        "free_slots": 0,
        "max_seq_len": 4,
        "kv_num_blocks": 0,
        "kv_block_size": 0,
    }
    first.gdn_states[0].fill_(7)
    assert torch.equal(store.get(10).gdn_states[0], first.gdn_states[0])
    assert not torch.equal(store.get(10).gdn_states[0], second.gdn_states[0])
    store.release(10)
    assert store.arena_summary() == {
        "num_slots": 2,
        "active_slots": 1,
        "free_slots": 1,
        "max_seq_len": 4,
        "kv_num_blocks": 0,
        "kv_block_size": 0,
    }


def test_decode_state_arena_can_allocate_paged_kv_buffers():
    arena = DecodeStateArena(
        cfg=tiny_qwen_cfg(),
        max_seq_len=4,
        num_slots=2,
        kv_num_blocks=8,
        kv_block_size=4,
        device="cpu",
    )

    assert arena.paged_attn_k is not None
    assert arena.paged_attn_v is not None
    assert arena.paged_attn_k[1].shape == (8, 1, 4, 4)
    assert arena.summary()["kv_num_blocks"] == 8
    assert arena.summary()["kv_block_size"] == 4
