from __future__ import annotations

from types import SimpleNamespace

import torch

from langburst.adapters.qwen36_impl.config import Qwen36_27B_TextConfig
from langburst.adapters.qwen36 import Qwen36Adapter
from langburst.core.features import RuntimeFeatures
from langburst.core.kv_cache import KVCacheLayout, KVCacheSpec, allocate_kv_cache_tensors, hadamard_transform, pack_int4_rows, unpack_int4_rows
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


def test_hadamard_transform_keeps_original_shape_for_batched_bdr_rows():
    x = torch.arange(1024, dtype=torch.float16).reshape(4, 2, 64, 2)

    y = hadamard_transform(x, 2)
    z = hadamard_transform(hadamard_transform(x, 2), 2)

    assert y.shape == x.shape
    assert z.shape == x.shape
    assert torch.allclose(z.float(), x.float(), atol=1e-3, rtol=1e-3)


def test_hadamard_transform_order4_no_dimension_accumulation_regression():
    x = torch.randn(4, 2, 64, 4, dtype=torch.float16)

    y = hadamard_transform(x, 4)
    z = hadamard_transform(hadamard_transform(x, 4), 4)

    assert y.shape == x.shape
    assert z.shape == x.shape
    assert torch.allclose(z.float(), x.float(), atol=2e-3, rtol=2e-3)


def test_int4_bdr_pack_roundtrip_accepts_multi_request_rows():
    x = torch.randn(4, 2, 256, dtype=torch.float16)
    rotated = hadamard_transform(x, 128)
    packed, scale, zero = pack_int4_rows(rotated)

    restored = unpack_int4_rows(packed, scale, zero, head_dim=256)
    unrotated = hadamard_transform(restored, 128)

    assert packed.shape == (4, 2, 128)
    assert scale.shape == (4, 2)
    assert zero.shape == (4, 2)
    assert unrotated.shape == x.shape
    assert torch.isfinite(unrotated.float()).all()


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
        "kv_cache_dtype": "fp16",
        "kv_storage_head_dim": 4,
        "paged_kv_enabled": False,
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
        "kv_cache_dtype": "fp16",
        "kv_storage_head_dim": 4,
        "paged_kv_enabled": False,
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
        "kv_cache_dtype": "fp16",
        "kv_storage_head_dim": 4,
        "paged_kv_enabled": False,
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
    assert arena.summary()["kv_cache_dtype"] == "fp16"
    assert arena.summary()["kv_storage_head_dim"] == 4
    assert arena.summary()["paged_kv_enabled"] is True
    assert arena.summary()["paged_kv_mirror"] is True
    assert arena.attn_k[1].size(2) == 4


def test_decode_state_arena_can_allocate_fp8_paged_kv_buffers():
    if not hasattr(torch, "float8_e4m3fn"):
        return
    arena = DecodeStateArena(
        cfg=tiny_qwen_cfg(),
        max_seq_len=4,
        num_slots=2,
        kv_num_blocks=8,
        kv_block_size=4,
        device="cpu",
        kv_cache_spec=KVCacheSpec.resolve("fp8_e4m3"),
    )

    assert arena.attn_k[1].size(2) == 0
    assert arena.attn_k[1].dtype == torch.float8_e4m3fn
    assert arena.paged_attn_k is not None
    assert arena.paged_attn_k[1].dtype == torch.float8_e4m3fn
    assert arena.summary()["kv_cache_dtype"] == "fp8_e4m3"
    assert arena.summary()["kv_storage_head_dim"] == 4
    assert arena.summary()["paged_kv_enabled"] is True


def test_qwen_state_estimate_accounts_for_fp8_kv():
    cfg = tiny_qwen_cfg()
    adapter = Qwen36Adapter()
    fp16 = adapter.estimate_state_bytes(cfg, recent_window=16, features=RuntimeFeatures(kv_cache_dtype="fp16"))
    fp8 = adapter.estimate_state_bytes(cfg, recent_window=16, features=RuntimeFeatures(kv_cache_dtype="fp8_e4m3"))

    assert fp8 < fp16
    assert fp16 - fp8 == len(cfg.attention_layers) * 2 * cfg.num_key_value_heads * 16 * cfg.attention_head_dim


def test_decode_state_arena_can_allocate_int4_bdr_paged_kv_buffers():
    arena = DecodeStateArena(
        cfg=tiny_qwen_cfg(),
        max_seq_len=4,
        num_slots=2,
        kv_num_blocks=8,
        kv_block_size=4,
        device="cpu",
        kv_cache_spec=KVCacheSpec.resolve("int4_bdr", hadamard_order=4),
    )

    assert arena.attn_k[1].size(2) == 0
    assert arena.paged_attn_k is not None
    assert arena.paged_attn_k_scale is not None
    assert arena.paged_attn_v_scale is not None
    assert arena.paged_attn_k_zero is not None
    assert arena.paged_attn_v_zero is not None
    assert arena.paged_attn_k[1].dtype == torch.uint8
    assert arena.paged_attn_k[1].shape == (8, 1, 4, 2)
    assert arena.paged_attn_k_scale[1].shape == (8, 1, 4)
    assert arena.paged_attn_k_zero[1].shape == (8, 1, 4)
    assert arena.summary()["kv_cache_dtype"] == "int4_bdr"
    assert arena.summary()["kv_storage_head_dim"] == 2
    assert arena.summary()["paged_kv_enabled"] is True


def test_qwen_state_estimate_accounts_for_int4_scale_overhead():
    cfg = tiny_qwen_cfg()
    adapter = Qwen36Adapter()
    fp16 = adapter.estimate_state_bytes(cfg, recent_window=16, features=RuntimeFeatures(kv_cache_dtype="fp16"))
    int4 = adapter.estimate_state_bytes(cfg, recent_window=16, features=RuntimeFeatures(kv_cache_dtype="int4"))

    assert int4 < fp16
    expected_kv = int(len(cfg.attention_layers) * 2 * cfg.num_key_value_heads * 16 * cfg.attention_head_dim * 0.5)
    expected_scales = len(cfg.attention_layers) * 2 * 2 * cfg.num_key_value_heads * 16 * 2
    fp16_kv = len(cfg.attention_layers) * 2 * cfg.num_key_value_heads * 16 * cfg.attention_head_dim * 2
    assert fp16 - int4 == fp16_kv - expected_kv - expected_scales


def test_core_kv_layout_allocates_model_agnostic_int4_buffers():
    layout = KVCacheLayout.from_parts(layers=(3, 7), num_kv_heads=2, head_dim=8)
    spec = KVCacheSpec.resolve("int4_bdr", hadamard_order=4)
    kv = allocate_kv_cache_tensors(layout, spec, seq_len=5, device=torch.device("cpu"), leading_shape=(4,))

    assert kv.k[3].shape == (4, 2, 5, 4)
    assert kv.v[7].shape == (4, 2, 5, 4)
    assert kv.k_scale is not None
    assert kv.k_zero is not None
    assert kv.k_scale[3].shape == (4, 2, 5)
    assert layout.total_bytes(spec, 5, leading_count=4) == (
        len(layout.layers) * 2 * 4 * 2 * 5 * 8 // 2
        + len(layout.layers) * 2 * 2 * 4 * 2 * 5 * 2
    )
