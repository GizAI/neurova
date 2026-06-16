from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from langburst.adapters.qwen36_impl.config import Qwen36_27B_TextConfig
from langburst.adapters.qwen36_impl.model import split_gdn_qkv
from langburst.adapters.qwen36_tools.quantize import should_quantize, quantize_symmetric_lowbit
from langburst.loader import LowBitTensor
from langburst.ops import CPUFallbackOps


def test_hf_layer_types_override_cycle(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "num_hidden_layers": 4,
        "hidden_size": 16,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    }))
    cfg = Qwen36_27B_TextConfig.from_hf_config(cfg_path)
    assert cfg.layer_type(0) == "gdn"
    assert cfg.layer_type(3) == "attn"


def test_split_gdn_qkv_shapes():
    cfg = Qwen36_27B_TextConfig(linear_num_key_heads=2, linear_num_value_heads=6, linear_key_head_dim=4, linear_value_head_dim=4)
    out = torch.arange(2 * (2 * 4 + 3 * 4), dtype=torch.float16)
    q, k, v = split_gdn_qkv(cfg, out)
    assert q.shape == (2, 4)
    assert k.shape == (2, 4)
    assert v.shape == (6, 4)


def test_lowbit_row_dequant_cpu_matches_manual():
    w = torch.tensor([[0.0, 1.0, -2.0, 3.0, 4.0]], dtype=torch.float16)
    packed, scales, meta = quantize_symmetric_lowbit(w, group_size=4, bits=4)
    lowbit = LowBitTensor(
        name="toy",
        qweight=torch.from_numpy(packed),
        scales=torch.from_numpy(scales),
        cols=meta["cols"],
        group_size=meta["group_size"],
    )
    row = lowbit.row_dequant(0).float()
    assert row.shape == (5,)
    # Low-bit quantization is lossy but should approximately recover the scale of the input.
    assert torch.allclose(row[:4], w[0, :4].float(), atol=0.25, rtol=0.25)


def test_quantizer_includes_split_gdn_and_embed():
    assert should_quantize("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
    assert should_quantize("model.language_model.layers.0.linear_attn.in_proj_z.weight")
    assert should_quantize("model.language_model.layers.0.linear_attn.in_proj_a.weight")
    assert should_quantize("model.language_model.layers.0.linear_attn.in_proj_b.weight")
    assert should_quantize("model.language_model.embed_tokens.weight")
    assert not should_quantize("model.language_model.embed_tokens.weight", fp16_embed=True)
