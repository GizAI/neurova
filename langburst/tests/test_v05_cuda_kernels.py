from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_qwen_rmsnorm_cuda_plus_one():
    import langburst_cuda
    x = torch.randn(2, 512, device="cuda", dtype=torch.float16)
    w = torch.randn(512, device="cuda", dtype=torch.float16) * 0.01
    y = langburst_cuda.rmsnorm_qwen(x, w, 1e-6)
    x32 = x.float()
    ref = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * (1.0 + w.float())
    assert torch.allclose(y.float(), ref, atol=3e-3, rtol=3e-3)


@pytest.mark.parametrize(
    ("bits", "packed", "expected"),
    [
        (4, [0x80, 0x97, 0x0F], [-8.0, 0.0, -1.0, 1.0, 7.0]),
        (3, [0xE0, 0x01], [-4.0, 0.0, 3.0, -4.0]),
    ],
)
def test_lowbit_row_dequant_cuda_matches_cpu(bits, packed, expected):
    import langburst_cuda
    qweight = torch.tensor([packed], device="cuda", dtype=torch.uint8)
    scales = torch.ones((1, 1), device="cuda", dtype=torch.float16)
    row = torch.tensor(0, device="cuda", dtype=torch.long)
    out = langburst_cuda.lowbit_row_dequant(qweight, scales, row, len(expected), 128, bits)
    assert out.cpu().tolist() == expected


def test_depthwise_conv_update_scan_matches_single_token_loop():
    import langburst_cuda

    torch.manual_seed(0)
    state_loop = torch.randn(32, 3, device="cuda", dtype=torch.float16)
    state_scan = state_loop.clone()
    x = torch.randn(7, 32, device="cuda", dtype=torch.float16)
    weight = torch.randn(32, 4, device="cuda", dtype=torch.float16)
    bias = torch.randn(32, device="cuda", dtype=torch.float16)

    loop_rows = []
    for row in x:
        loop_rows.append(langburst_cuda.depthwise_conv_update(state_loop, row.contiguous(), weight, bias))
    y_loop = torch.stack(loop_rows, dim=0)
    y_scan = langburst_cuda.depthwise_conv_update_scan(state_scan, x.contiguous(), weight, bias)
    torch.cuda.synchronize()

    assert torch.equal(y_scan, y_loop)
    assert torch.equal(state_scan, state_loop)


def test_depthwise_conv_update_batch_matches_single_token_loop():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 4
    batch = 3
    channels = 32
    history = 3
    state_batch = torch.randn(slots, channels, history, device="cuda", dtype=torch.float16)
    state_loop = state_batch.clone()
    state_indices = torch.tensor([2, 0, 3], device="cuda", dtype=torch.long)
    x = torch.randn(batch, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(channels, history + 1, device="cuda", dtype=torch.float16)
    bias = torch.randn(channels, device="cuda", dtype=torch.float16)

    loop_rows = []
    for row in range(batch):
        slot = int(state_indices[row].item())
        loop_rows.append(langburst_cuda.depthwise_conv_update(state_loop[slot], x[row].contiguous(), weight, bias))
    y_loop = torch.stack(loop_rows, dim=0)
    y_batch = langburst_cuda.depthwise_conv_update_batch(state_batch, state_indices, x.contiguous(), weight, bias)
    torch.cuda.synchronize()

    assert torch.equal(y_batch, y_loop)
    assert torch.equal(state_batch, state_loop)


def test_depthwise_conv_update_spec_commits_prefix_only():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 4
    batch = 3
    tokens = 5
    channels = 32
    history = 3
    state_spec = torch.randn(slots, channels, history, device="cuda", dtype=torch.float16)
    state_ref_full = state_spec.clone()
    state_ref_commit = state_spec.clone()
    state_indices = torch.tensor([2, 0, 3], device="cuda", dtype=torch.long)
    commit_tokens = torch.tensor([5, 2, 0], device="cuda", dtype=torch.int32)
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(channels, history + 1, device="cuda", dtype=torch.float16)
    bias = torch.randn(channels, device="cuda", dtype=torch.float16)

    full_rows = []
    for row in range(batch):
        slot = int(state_indices[row].item())
        row_out = []
        for step in range(tokens):
            row_out.append(langburst_cuda.depthwise_conv_update(state_ref_full[slot], x[row, step].contiguous(), weight, bias))
        full_rows.append(torch.stack(row_out, dim=0))
        commit_n = int(commit_tokens[row].item())
        if commit_n > 0:
            for step in range(commit_n):
                langburst_cuda.depthwise_conv_update(state_ref_commit[slot], x[row, step].contiguous(), weight, bias)
    y_ref = torch.stack(full_rows, dim=0)
    y_spec = langburst_cuda.depthwise_conv_update_spec(state_spec, state_indices, commit_tokens, x.contiguous(), weight, bias)
    torch.cuda.synchronize()

    assert torch.equal(y_spec, y_ref)
    assert torch.equal(state_spec, state_ref_commit)


def test_depthwise_conv_update_spec_trajectory_out_matches_allocating_api():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 3
    batch = 2
    tokens = 4
    channels = 16
    history = 3
    state = torch.randn(slots, channels, history, device="cuda", dtype=torch.float16)
    state_indices = torch.tensor([1, 2], device="cuda", dtype=torch.long)
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.float16)
    weight = torch.randn(channels, history + 1, device="cuda", dtype=torch.float16)
    bias = torch.randn(channels, device="cuda", dtype=torch.float16)

    y_ref, traj_ref = langburst_cuda.depthwise_conv_update_spec_trajectory(
        state,
        state_indices,
        x.contiguous(),
        weight,
        bias,
    )
    y_out = torch.empty_like(y_ref)
    traj_out = torch.empty_like(traj_ref)
    langburst_cuda.depthwise_conv_update_spec_trajectory_out(
        state,
        state_indices,
        x.contiguous(),
        weight,
        bias,
        y_out,
        traj_out,
    )
    torch.cuda.synchronize()

    assert torch.equal(y_out, y_ref)
    assert torch.equal(traj_out, traj_ref)


def test_gdn_recurrent_ab_scan_matches_single_token_loop():
    import langburst_cuda

    torch.manual_seed(0)
    tokens = 5
    kv_heads = 4
    v_heads = 8
    head_dim = 128
    q = (torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(tokens, v_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a = (torch.randn(tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    b = (torch.randn(tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a_log = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 1.0).contiguous()
    dt_bias = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    state_loop = (torch.randn(v_heads, head_dim, head_dim, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_scan = state_loop.clone()

    loop_rows = []
    for t in range(tokens):
        loop_rows.append(
            langburst_cuda.gdn_recurrent_ab(
                q[t],
                k[t],
                v[t],
                a[t],
                b[t],
                a_log,
                dt_bias,
                state_loop,
            )
        )
    y_loop = torch.stack(loop_rows, dim=0)
    y_scan = langburst_cuda.gdn_recurrent_ab_scan(q, k, v, a, b, a_log, dt_bias, state_scan)
    torch.cuda.synchronize()

    assert torch.equal(y_scan, y_loop)
    assert torch.equal(state_scan, state_loop)


def test_gdn_recurrent_ab_batch_matches_single_token_loop():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 4
    batch = 3
    kv_heads = 4
    v_heads = 8
    head_dim = 128
    q = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(batch, v_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a = (torch.randn(batch, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    b = (torch.randn(batch, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a_log = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 1.0).contiguous()
    dt_bias = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    state_batch = (torch.randn(slots, v_heads, head_dim, head_dim, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_loop = state_batch.clone()
    state_indices = torch.tensor([2, 0, 3], device="cuda", dtype=torch.long)

    loop_rows = []
    for row in range(batch):
        slot = int(state_indices[row].item())
        loop_rows.append(
            langburst_cuda.gdn_recurrent_ab(
                q[row],
                k[row],
                v[row],
                a[row],
                b[row],
                a_log,
                dt_bias,
                state_loop[slot],
            )
        )
    y_loop = torch.stack(loop_rows, dim=0)
    y_batch = langburst_cuda.gdn_recurrent_ab_batch(q, k, v, a, b, a_log, dt_bias, state_batch, state_indices)
    torch.cuda.synchronize()

    assert torch.equal(y_batch, y_loop)
    assert torch.equal(state_batch, state_loop)


def test_gdn_recurrent_ab_batch_matches_qwen_shape_single_token_loop():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 3
    batch = 2
    kv_heads = 16
    v_heads = 48
    head_dim = 128
    q = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(batch, v_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a = (torch.randn(batch, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    b = (torch.randn(batch, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a_log = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 1.0).contiguous()
    dt_bias = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    state_batch = (torch.randn(slots, v_heads, head_dim, head_dim, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_loop = state_batch.clone()
    state_indices = torch.tensor([2, 0], device="cuda", dtype=torch.long)

    loop_rows = []
    for row in range(batch):
        slot = int(state_indices[row].item())
        loop_rows.append(
            langburst_cuda.gdn_recurrent_ab(
                q[row],
                k[row],
                v[row],
                a[row],
                b[row],
                a_log,
                dt_bias,
                state_loop[slot],
            )
        )
    y_loop = torch.stack(loop_rows, dim=0)
    y_batch = langburst_cuda.gdn_recurrent_ab_batch(q, k, v, a, b, a_log, dt_bias, state_batch, state_indices)
    torch.cuda.synchronize()

    assert torch.equal(state_batch, state_loop)
    assert torch.allclose(y_batch, y_loop, atol=2e-4, rtol=2e-4)


def test_gdn_recurrent_ab_spec_commits_prefix_only():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 4
    batch = 3
    tokens = 4
    kv_heads = 4
    v_heads = 8
    head_dim = 128
    q = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(batch, tokens, v_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a = (torch.randn(batch, tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    b = (torch.randn(batch, tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a_log = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 1.0).contiguous()
    dt_bias = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    state_spec = (torch.randn(slots, v_heads, head_dim, head_dim, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_ref_full = state_spec.clone()
    state_ref_commit = state_spec.clone()
    state_indices = torch.tensor([2, 0, 3], device="cuda", dtype=torch.long)
    commit_tokens = torch.tensor([4, 2, 0], device="cuda", dtype=torch.int32)

    full_rows = []
    for row in range(batch):
        slot = int(state_indices[row].item())
        row_out = []
        for step in range(tokens):
            row_out.append(
                langburst_cuda.gdn_recurrent_ab(
                    q[row, step],
                    k[row, step],
                    v[row, step],
                    a[row, step],
                    b[row, step],
                    a_log,
                    dt_bias,
                    state_ref_full[slot],
                )
            )
        full_rows.append(torch.stack(row_out, dim=0))
        commit_n = int(commit_tokens[row].item())
        if commit_n > 0:
            for step in range(commit_n):
                langburst_cuda.gdn_recurrent_ab(
                    q[row, step],
                    k[row, step],
                    v[row, step],
                    a[row, step],
                    b[row, step],
                    a_log,
                    dt_bias,
                    state_ref_commit[slot],
                )
    y_ref = torch.stack(full_rows, dim=0)
    y_spec = langburst_cuda.gdn_recurrent_ab_spec(q, k, v, a, b, a_log, dt_bias, state_spec, state_indices, commit_tokens)
    torch.cuda.synchronize()

    assert torch.allclose(y_spec, y_ref, atol=2e-4, rtol=2e-4)
    assert torch.allclose(state_spec, state_ref_commit, atol=2e-4, rtol=2e-4)


def test_gdn_recurrent_ab_spec_trajectory_out_matches_allocating_api():
    import langburst_cuda

    torch.manual_seed(0)
    slots = 3
    batch = 2
    tokens = 3
    kv_heads = 2
    v_heads = 4
    head_dim = 128
    q = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v = (torch.randn(batch, tokens, v_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a = (torch.randn(batch, tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    b = (torch.randn(batch, tokens, v_heads, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    a_log = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01 - 1.0).contiguous()
    dt_bias = (torch.randn(v_heads, device="cuda", dtype=torch.float32) * 0.01).contiguous()
    state = (torch.randn(slots, v_heads, head_dim, head_dim, device="cuda", dtype=torch.float16) * 0.01).contiguous()
    state_indices = torch.tensor([1, 2], device="cuda", dtype=torch.long)

    y_ref, traj_ref = langburst_cuda.gdn_recurrent_ab_spec_trajectory(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_indices,
    )
    y_out = torch.empty_like(y_ref)
    traj_out = torch.empty_like(traj_ref)
    langburst_cuda.gdn_recurrent_ab_spec_trajectory_out(
        q,
        k,
        v,
        a,
        b,
        a_log,
        dt_bias,
        state,
        state_indices,
        y_out,
        traj_out,
    )
    torch.cuda.synchronize()

    assert torch.allclose(y_out, y_ref, atol=2e-4, rtol=2e-4)
    assert torch.allclose(traj_out, traj_ref, atol=2e-4, rtol=2e-4)


def test_silu_mul_cuda_matches_torch():
    import langburst_cuda

    torch.manual_seed(0)
    gate = torch.randn(4096, device="cuda", dtype=torch.float16)
    up = torch.randn(4096, device="cuda", dtype=torch.float16)
    y = langburst_cuda.silu_mul(gate.contiguous(), up.contiguous())
    ref = (torch.nn.functional.silu(gate.float()) * up.float()).half()
    torch.cuda.synchronize()
    assert torch.allclose(y, ref, atol=2e-3, rtol=2e-3)


def test_attention_decode_paged_matches_contiguous_cache_loop():
    import langburst_cuda

    torch.manual_seed(0)
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 256
    block_size = 4
    num_blocks = 4
    q = (torch.randn(batch, q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k_pages = (torch.randn(num_blocks, kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    v_pages = (torch.randn(num_blocks, kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16) * 0.1).contiguous()
    k_ref_pages = k_pages.clone()
    v_ref_pages = v_pages.clone()
    block_tables = torch.tensor([[0, 3], [1, 2]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    slot_mapping = torch.tensor([2, 8], device="cuda", dtype=torch.long)

    y_paged = langburst_cuda.attention_decode_paged_fp16(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        slot_mapping,
        block_tables,
        seq_lens,
        block_size,
        head_dim ** -0.5,
    )

    ref_rows = []
    for row in range(batch):
        slot = int(slot_mapping[row].item())
        block = slot // block_size
        offset = slot % block_size
        k_ref_pages[block, :, offset, :] = k_new[row]
        v_ref_pages[block, :, offset, :] = v_new[row]
        k_rows = []
        v_rows = []
        for pos in range(int(seq_lens[row].item())):
            block_idx = pos // block_size
            block_offset = pos % block_size
            block_id = int(block_tables[row, block_idx].item())
            k_rows.append(k_ref_pages[block_id, :, block_offset, :])
            v_rows.append(v_ref_pages[block_id, :, block_offset, :])
        k_cache = torch.stack(k_rows, dim=1).contiguous()
        v_cache = torch.stack(v_rows, dim=1).contiguous()
        ref_rows.append(langburst_cuda.attention_decode_fp16(q[row], k_cache, v_cache, int(seq_lens[row].item()), head_dim ** -0.5))
    y_ref = torch.stack(ref_rows, dim=0).contiguous()
    torch.cuda.synchronize()

    assert torch.equal(k_pages, k_ref_pages)
    assert torch.equal(v_pages, v_ref_pages)
    assert torch.equal(y_paged, y_ref)


def test_attention_decode_paged_fp8_is_finite_and_updates_pages():
    import langburst_cuda

    if not hasattr(torch, "float8_e4m3fn"):
        return
    torch.manual_seed(1)
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 256
    block_size = 4
    num_blocks = 4
    q = (torch.randn(batch, q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    k_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    v_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    k_pages = torch.empty((num_blocks, kv_heads, block_size, head_dim), device="cuda", dtype=torch.float8_e4m3fn)
    v_pages = torch.empty_like(k_pages)
    seed_k = (torch.randn(num_blocks, kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16) * 0.05)
    seed_v = (torch.randn(num_blocks, kv_heads, block_size, head_dim, device="cuda", dtype=torch.float16) * 0.05)
    k_pages.copy_(seed_k)
    v_pages.copy_(seed_v)
    before_k = k_pages.clone()
    before_v = v_pages.clone()
    block_tables = torch.tensor([[0, 3], [1, 2]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    slot_mapping = torch.tensor([2, 8], device="cuda", dtype=torch.long)

    y = langburst_cuda.attention_decode_paged_fp8_e4m3(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        slot_mapping,
        block_tables,
        seq_lens,
        block_size,
        head_dim ** -0.5,
        1.0,
        1.0,
    )
    torch.cuda.synchronize()

    assert y.shape == (batch, q_heads, head_dim)
    assert y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()
    for slot in slot_mapping.detach().cpu().tolist():
        block = int(slot) // block_size
        offset = int(slot) % block_size
        assert not torch.equal(k_pages[block, :, offset, :], before_k[block, :, offset, :])
        assert not torch.equal(v_pages[block, :, offset, :], before_v[block, :, offset, :])


def test_attention_decode_paged_int4_bdr_is_finite_and_updates_pages():
    import langburst_cuda

    torch.manual_seed(2)
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 256
    block_size = 4
    num_blocks = 4
    packed_dim = head_dim // 2
    q = (torch.randn(batch, q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    k_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    v_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.05).contiguous()
    k_pages = torch.zeros((num_blocks, kv_heads, block_size, packed_dim), device="cuda", dtype=torch.uint8)
    v_pages = torch.zeros_like(k_pages)
    k_scales = torch.ones((num_blocks, kv_heads, block_size), device="cuda", dtype=torch.float16)
    v_scales = torch.ones_like(k_scales)
    k_zeros = torch.zeros_like(k_scales)
    v_zeros = torch.zeros_like(k_scales)
    before_k = k_pages.clone()
    before_v = v_pages.clone()
    block_tables = torch.tensor([[0, 3], [1, 2]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    slot_mapping = torch.tensor([2, 8], device="cuda", dtype=torch.long)

    y = langburst_cuda.attention_decode_paged_int4(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scales,
        v_scales,
        k_zeros,
        v_zeros,
        slot_mapping,
        block_tables,
        seq_lens,
        block_size,
        head_dim ** -0.5,
        128,
        True,
        False,
    )
    torch.cuda.synchronize()

    assert y.shape == (batch, q_heads, head_dim)
    assert y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()
    for slot in slot_mapping.detach().cpu().tolist():
        block = int(slot) // block_size
        offset = int(slot) % block_size
        assert not torch.equal(k_pages[block, :, offset, :], before_k[block, :, offset, :])
        assert not torch.equal(v_pages[block, :, offset, :], before_v[block, :, offset, :])


def test_attention_decode_paged_int4_tiled_layout_is_finite_and_updates_pages():
    import langburst_cuda

    torch.manual_seed(4)
    batch = 2
    q_heads = 4
    kv_heads = 2
    head_dim = 256
    block_size = 16
    num_blocks = 4
    packed_dim = head_dim // 2
    q = (torch.randn(batch, q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.04).contiguous()
    k_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.04).contiguous()
    v_new = (torch.randn(batch, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.04).contiguous()
    k_pages = torch.zeros((num_blocks, kv_heads, packed_dim, block_size), device="cuda", dtype=torch.uint8)
    v_pages = torch.zeros_like(k_pages)
    before_k = k_pages.clone()
    before_v = v_pages.clone()
    k_scales = torch.ones((num_blocks, kv_heads, block_size), device="cuda", dtype=torch.float16)
    v_scales = torch.ones_like(k_scales)
    k_zeros = torch.zeros_like(k_scales)
    v_zeros = torch.zeros_like(k_scales)
    block_tables = torch.tensor([[0, 3], [1, 2]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([12, 21], device="cuda", dtype=torch.int32)
    slot_mapping = torch.tensor([11, 20], device="cuda", dtype=torch.long)

    y = langburst_cuda.attention_decode_paged_int4(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scales,
        v_scales,
        k_zeros,
        v_zeros,
        slot_mapping,
        block_tables,
        seq_lens,
        block_size,
        head_dim ** -0.5,
        128,
        False,
        False,
        True,
    )
    torch.cuda.synchronize()

    assert y.shape == (batch, q_heads, head_dim)
    assert y.dtype == torch.float16
    assert torch.isfinite(y.float()).all()
    for slot in slot_mapping.detach().cpu().tolist():
        block = int(slot) // block_size
        offset = int(slot) % block_size
        assert not torch.equal(k_pages[block, :, :, offset], before_k[block, :, :, offset])
        assert not torch.equal(v_pages[block, :, :, offset], before_v[block, :, :, offset])


def test_attention_append_paged_int4_spec_commits_prefix_only():
    import langburst_cuda

    torch.manual_seed(5)
    batch = 3
    tokens = 4
    kv_heads = 2
    head_dim = 256
    block_size = 8
    num_blocks = 4
    packed_dim = head_dim // 2
    k_new = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.03).contiguous()
    v_new = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.03).contiguous()
    k_pages = torch.zeros((num_blocks, kv_heads, block_size, packed_dim), device="cuda", dtype=torch.uint8)
    v_pages = torch.zeros_like(k_pages)
    k_ref = k_pages.clone()
    v_ref = v_pages.clone()
    k_scales = torch.ones((num_blocks, kv_heads, block_size), device="cuda", dtype=torch.float16)
    v_scales = torch.ones_like(k_scales)
    k_zeros = torch.zeros_like(k_scales)
    v_zeros = torch.zeros_like(k_scales)
    k_ref_scales = k_scales.clone()
    v_ref_scales = v_scales.clone()
    k_ref_zeros = k_zeros.clone()
    v_ref_zeros = v_zeros.clone()
    slot_mapping = torch.tensor(
        [
            [0, 1, 2, 3],
            [8, 9, 10, 11],
            [16, 17, 18, 19],
        ],
        device="cuda",
        dtype=torch.long,
    )
    commit_tokens = torch.tensor([4, 2, 0], device="cuda", dtype=torch.int32)

    langburst_cuda.attention_append_paged_int4_spec(
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scales,
        v_scales,
        k_zeros,
        v_zeros,
        slot_mapping,
        commit_tokens,
        block_size,
        128,
        True,
        False,
    )
    for row in range(batch):
        commit_n = int(commit_tokens[row].item())
        if commit_n == 0:
            continue
        langburst_cuda.attention_append_paged_int4(
            k_new[row, :commit_n].contiguous(),
            v_new[row, :commit_n].contiguous(),
            k_ref,
            v_ref,
            k_ref_scales,
            v_ref_scales,
            k_ref_zeros,
            v_ref_zeros,
            slot_mapping[row, :commit_n].contiguous(),
            block_size,
            128,
            True,
            False,
        )
    torch.cuda.synchronize()

    assert torch.equal(k_pages, k_ref)
    assert torch.equal(v_pages, v_ref)
    assert torch.equal(k_scales, k_ref_scales)
    assert torch.equal(v_scales, v_ref_scales)
    assert torch.equal(k_zeros, k_ref_zeros)
    assert torch.equal(v_zeros, v_ref_zeros)


def test_attention_spec_decode_paged_int4_matches_candidate_causal_reference():
    import langburst_cuda

    torch.manual_seed(6)
    batch = 2
    tokens = 3
    q_heads = 4
    kv_heads = 2
    head_dim = 256
    block_size = 8
    num_blocks = 2
    packed_dim = head_dim // 2
    scale = head_dim ** -0.5
    q = (torch.randn(batch, tokens, q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.03).contiguous()
    k_new = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.03).contiguous()
    v_new = (torch.randn(batch, tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.03).contiguous()
    k_pages = torch.zeros((num_blocks, kv_heads, block_size, packed_dim), device="cuda", dtype=torch.uint8)
    v_pages = torch.zeros_like(k_pages)
    k_scales = torch.ones((num_blocks, kv_heads, block_size), device="cuda", dtype=torch.float16)
    v_scales = torch.ones_like(k_scales)
    k_zeros = torch.zeros_like(k_scales)
    v_zeros = torch.zeros_like(k_scales)
    block_tables = torch.zeros((batch, 1), device="cuda", dtype=torch.int32)
    base_seq_lens = torch.zeros((batch,), device="cuda", dtype=torch.int32)

    y = langburst_cuda.attention_spec_decode_paged_int4(
        q,
        k_new,
        v_new,
        k_pages,
        v_pages,
        k_scales,
        v_scales,
        k_zeros,
        v_zeros,
        block_tables,
        base_seq_lens,
        block_size,
        scale,
        128,
        False,
        False,
    )
    ref = torch.empty_like(y)
    ratio = q_heads // kv_heads
    for row in range(batch):
        for step in range(tokens):
            for qh in range(q_heads):
                kvh = qh // ratio
                scores = torch.matmul(k_new[row, : step + 1, kvh].float(), q[row, step, qh].float()) * scale
                probs = torch.softmax(scores, dim=0)
                ref[row, step, qh] = torch.matmul(probs, v_new[row, : step + 1, kvh].float()).to(torch.float16)
    torch.cuda.synchronize()

    assert y.shape == (batch, tokens, q_heads, head_dim)
    assert torch.allclose(y.float(), ref.float(), atol=2e-3, rtol=2e-3)
