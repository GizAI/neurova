from __future__ import annotations

import torch

from qwenburst.gdn_reference import gdn_recurrent_reference
from qwenburst.model import apply_rope_single_token
from qwenburst.ops import CPUFallbackOps
from qwenburst.quantize import should_quantize


def test_embedding_is_lowbit_by_default():
    assert should_quantize("model.embed_tokens.weight")
    assert should_quantize("model.layers.0.mlp.up_proj.weight")
    assert not should_quantize("model.embed_tokens.weight", fp16_embed=True)
    assert should_quantize("model.layers.0.linear_attn.in_proj_qkv.weight")
    assert should_quantize("model.layers.0.linear_attn.in_proj_z.weight")
    assert should_quantize("model.layers.0.linear_attn.in_proj_a.weight")
    assert should_quantize("model.layers.0.linear_attn.in_proj_b.weight")


def test_rope_pos_zero_identity_and_pass_dims_preserved():
    q = torch.randn(2, 8, dtype=torch.float32)
    k = torch.randn(1, 8, dtype=torch.float32)
    q0, k0 = apply_rope_single_token(q, k, pos=0, rope_dim=4, rope_theta=10000.0)
    assert torch.allclose(q0, q)
    assert torch.allclose(k0, k)

    q1, k1 = apply_rope_single_token(q, k, pos=3, rope_dim=4, rope_theta=10000.0)
    assert torch.allclose(q1[:, 4:], q[:, 4:])
    assert torch.allclose(k1[:, 4:], k[:, 4:])
    assert not torch.allclose(q1[:, :4], q[:, :4])


def test_cpu_fallback_gdn_matches_reference():
    torch.manual_seed(3)
    q = torch.randn(2, 128, dtype=torch.float16) * 0.1
    k = torch.randn(2, 128, dtype=torch.float16) * 0.1
    v = torch.randn(6, 128, dtype=torch.float16) * 0.1
    g = torch.randn(6, dtype=torch.float32) * 0.01 - 0.1
    beta = torch.sigmoid(torch.randn(6, dtype=torch.float32)).to(torch.float16)
    s0 = torch.randn(6, 128, 128, dtype=torch.float16) * 0.01
    s_ref = s0.clone()
    s_cpu = s0.clone()
    out_ref, _ = gdn_recurrent_reference(q, k, v, g, beta, s_ref, inplace=True)
    out_cpu = CPUFallbackOps.gdn_recurrent(q, k, v, g, beta, s_cpu)
    assert torch.allclose(out_ref, out_cpu, atol=2e-5, rtol=0)
    assert torch.allclose(s_ref, s_cpu, atol=2e-5, rtol=0)


from qwenburst.config import Qwen36_27B_TextConfig
from qwenburst.state import DecodeState


def test_ring_kv_window_preserves_logical_order_cpu():
    cfg = Qwen36_27B_TextConfig(num_layers=4, num_key_value_heads=1, attention_head_dim=2)
    st = DecodeState.allocate(cfg, max_seq_len=3, device="cpu", dtype=torch.float16, kv_window_policy="ring")
    layer = cfg.attention_layers[0]
    for i in range(5):
        k = torch.full((1, 2), float(i), dtype=torch.float16)
        v = torch.full((1, 2), float(i + 10), dtype=torch.float16)
        st.append_attention_kv(layer, k, v)
        st.finish_token()
    # Before next append, live logical window is tokens 2,3,4.  attention_kv_view
    # is normally called before finish_token, so emulate by backing pos to last token.
    st.pos -= 1
    k_view, v_view, live = st.attention_kv_view(layer)
    assert live == 3
    assert k_view[0, :, 0].tolist() == [2.0, 3.0, 4.0]
    assert v_view[0, :, 0].tolist() == [12.0, 13.0, 14.0]


def test_qwen_rmsnorm_uses_plus_one_weight():
    x = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float16)
    w = torch.tensor([0.0, 1.0, -0.5], dtype=torch.float16)
    y = CPUFallbackOps.rmsnorm_qwen(x, w, 1e-6)
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean() + 1e-6) * (1.0 + w.float())
    assert torch.allclose(y.float(), ref, atol=1e-3, rtol=1e-3)
