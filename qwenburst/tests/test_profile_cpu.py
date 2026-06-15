from __future__ import annotations

from qwenburst.profile import projection_category


def test_projection_category_keeps_hot_paths_stable():
    assert projection_category("model.layers.0.mlp.gate_up_proj.weight") == "mlp_gate_up"
    assert projection_category("model.layers.0.mlp.down_proj.weight") == "mlp_down"
    assert projection_category("model.layers.0.linear_attn.in_proj_qkvz.weight") == "gdn_qkvz"
    assert projection_category("model.layers.0.linear_attention.out_proj.weight") == "gdn_out"
    assert projection_category("model.layers.3.self_attn.qkv_proj.weight") == "attn_qkv"
    assert projection_category("model.layers.3.self_attn.o_proj.weight") == "attn_o"
    assert projection_category("lm_head.weight") == "lm_head"
