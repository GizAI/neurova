from __future__ import annotations

import json

import pytest
import torch

from qwenburst.dflash import DFlashConfig, DFlashDraftAdapter, convert_dflash_lowbit, inspect_dflash

safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_fake_dflash(root):
    cfg = {
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 32,
        "block_size": 4,
        "num_target_layers": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 16,
        "dflash_config": {"target_layer_ids": [1, 5], "mask_token_id": 0},
    }
    root.mkdir()
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    safetensors_torch.save_file(
        ({
            "fc.weight": torch.randn(8, 16, dtype=torch.float16),
            "hidden_norm.weight": torch.ones(8, dtype=torch.float16),
            "norm.weight": torch.ones(8, dtype=torch.float16),
            "layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.float16),
            "layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float16),
            "layers.0.self_attn.q_norm.weight": torch.ones(4, dtype=torch.float16),
            "layers.0.self_attn.k_norm.weight": torch.ones(4, dtype=torch.float16),
            "layers.0.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.float16),
            "layers.0.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.float16),
            "layers.0.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.float16),
            "layers.0.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.float16),
            "layers.0.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.float16),
            "layers.0.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.float16),
            "layers.0.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.float16),
            "layers.1.input_layernorm.weight": torch.ones(8, dtype=torch.float16),
            "layers.1.post_attention_layernorm.weight": torch.ones(8, dtype=torch.float16),
            "layers.1.self_attn.q_norm.weight": torch.ones(4, dtype=torch.float16),
            "layers.1.self_attn.k_norm.weight": torch.ones(4, dtype=torch.float16),
            "layers.1.self_attn.q_proj.weight": torch.randn(8, 8, dtype=torch.float16),
            "layers.1.self_attn.k_proj.weight": torch.randn(4, 8, dtype=torch.float16),
            "layers.1.self_attn.v_proj.weight": torch.randn(4, 8, dtype=torch.float16),
            "layers.1.self_attn.o_proj.weight": torch.randn(8, 8, dtype=torch.float16),
            "layers.1.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.float16),
            "layers.1.mlp.up_proj.weight": torch.randn(16, 8, dtype=torch.float16),
            "layers.1.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.float16),
        }),
        root / "model.safetensors",
    )


def test_dflash_config_and_inspect(tmp_path):
    src = tmp_path / "draft"
    _write_fake_dflash(src)
    cfg = DFlashConfig.from_path(src)
    assert cfg.target_layer_ids == (1, 5)
    assert cfg.block_size == 4
    info = inspect_dflash(src)
    assert info["tensor_count"] == 25
    assert info["config"]["target_layer_ids"] == [1, 5]


def test_dflash_lowbit_conversion_loads_adapter(tmp_path):
    src = tmp_path / "draft"
    out = tmp_path / "draft-qb3"
    _write_fake_dflash(src)
    convert_dflash_lowbit(src, out, bits=3, group_size=4)
    index = json.loads((out / "qwenburst_index.json").read_text(encoding="utf-8"))
    assert index["format"] == "qwenburst-dflash-q3-v1"
    assert index["tensors"]["fc.weight"]["kind"] == "lowbit_symmetric_groupwise"
    assert index["tensors"]["layers.0.input_layernorm.weight"]["kind"] == "fp16_raw"
    adapter = DFlashDraftAdapter.from_lowbit_dir(out, device="cpu")
    assert adapter.target_layer_ids == (1, 5)
