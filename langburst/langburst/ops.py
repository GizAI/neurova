from __future__ import annotations

import importlib
import os
from functools import lru_cache

import torch


class CPUFallbackOps:
    """Correctness-first CPU/PyTorch fallback.

    It is deliberately slow.  The 100 tok/s path requires langburst_cuda.
    """

    @staticmethod
    def lowbit_gemv(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        cols: int,
        group_size: int,
        bits: int,
        rows_per_cta: int = 8,
    ) -> torch.Tensor:
        rows = qweight.shape[0]
        device = x.device
        out = torch.empty((rows,), device=device, dtype=torch.float32)
        x32 = x.to(torch.float32)
        qmask = (1 << bits) - 1
        zero = 1 << (bits - 1)
        for r in range(rows):
            acc = torch.zeros((), device=device, dtype=torch.float32)
            for c in range(cols):
                bit_pos = c * bits
                byte_i = bit_pos // 8
                shift = bit_pos % 8
                word = int(qweight[r, byte_i].item())
                if shift + bits > 8 and byte_i + 1 < qweight.shape[1]:
                    word |= int(qweight[r, byte_i + 1].item()) << 8
                q = ((word >> shift) & qmask) - zero
                acc = acc + (q * float(scales[r, c // group_size].item())) * x32[c]
            out[r] = acc
        return out.to(x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.float32)

    @staticmethod
    def lowbit_row_dequant(qweight: torch.Tensor, scales: torch.Tensor, row: int | torch.Tensor, cols: int, group_size: int, bits: int) -> torch.Tensor:
        if torch.is_tensor(row):
            row = int(row.item())
        qrow = qweight[row]
        srow = scales[row]
        vals = torch.empty((cols,), device=qweight.device, dtype=torch.float32)
        qmask = (1 << bits) - 1
        zero = 1 << (bits - 1)
        for c in range(cols):
            bit_pos = c * bits
            byte_i = bit_pos // 8
            shift = bit_pos % 8
            word = int(qrow[byte_i].item())
            if shift + bits > 8 and byte_i + 1 < qrow.numel():
                word |= int(qrow[byte_i + 1].item()) << 8
            vals[c] = (((word >> shift) & qmask) - zero) * float(srow[c // group_size].item())
        return vals.to(torch.float16)

    @staticmethod
    def lowbit_marlin_gemm(qweight: torch.Tensor, scales: torch.Tensor, x: torch.Tensor, cols: int, group_size: int) -> torch.Tensor:
        raise RuntimeError("lowbit_marlin_gemm requires langburst_cuda")

    @staticmethod
    def lowbit_marlin_gemm_out(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        cols: int,
        group_size: int,
    ) -> None:
        raise RuntimeError("lowbit_marlin_gemm_out requires langburst_cuda")

    @staticmethod
    def lowbit_marlin_mlp_streaming_out(
        gate_up_qweight: torch.Tensor,
        gate_up_scales: torch.Tensor,
        down_qweight: torch.Tensor,
        down_scales: torch.Tensor,
        x: torch.Tensor,
        out: torch.Tensor,
        accum: torch.Tensor,
        sync: torch.Tensor,
        epoch: int,
        hidden: int,
        intermediate: int,
        gate_group_size: int,
        down_group_size: int,
    ) -> None:
        gate_up_scratch = torch.empty((x.size(0), int(intermediate) * 2), device=x.device, dtype=x.dtype)
        down_workspace = torch.empty((max(1, int(hidden) // 128 * 16),), device=x.device, dtype=torch.int32)
        CPUFallbackOps.lowbit_marlin_gemm_out(gate_up_qweight, gate_up_scales, x, gate_up_scratch, down_workspace, int(hidden), int(gate_group_size))
        act = CPUFallbackOps.silu_mul_packed(gate_up_scratch, int(intermediate)).contiguous()
        CPUFallbackOps.lowbit_marlin_gemm_out(down_qweight, down_scales, act, out, down_workspace, int(intermediate), int(down_group_size))

    @staticmethod
    def lowbit_marlin_gemm_silu_packed_out(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        mixed: torch.Tensor,
        out: torch.Tensor,
        workspace: torch.Tensor,
        cols: int,
        group_size: int,
    ) -> None:
        act = CPUFallbackOps.silu_mul_packed(mixed, int(cols)).contiguous()
        CPUFallbackOps.lowbit_marlin_gemm_out(qweight, scales, act, out, workspace, cols, group_size)

    @staticmethod
    def lowbit_marlin_gemm_argmax_out(
        qweight: torch.Tensor,
        scales: torch.Tensor,
        x: torch.Tensor,
        scratch_out: torch.Tensor,
        workspace: torch.Tensor,
        argmax_state: torch.Tensor,
        argmax_out: torch.Tensor,
        argmax_sync: torch.Tensor,
        argmax_epoch: int,
        cols: int,
        group_size: int,
    ) -> None:
        # CPU fallback preserves the public contract.  The CUDA implementation
        # reduces logits inside the Marlin write path and writes argmax_out
        # directly; scratch_out is only needed by this fallback path.
        CPUFallbackOps.lowbit_marlin_gemm_out(qweight, scales, x, scratch_out, workspace, cols, group_size)
        CPUFallbackOps.argmax_many_out(scratch_out, argmax_out)

    @staticmethod
    def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        x32 = x.to(torch.float32)
        inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x32 * inv * weight.to(device=x.device, dtype=torch.float32)).to(x.dtype)

    @staticmethod
    def rmsnorm_qwen(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # Qwen3.5/Qwen3.6 HF RMSNorm uses (1 + weight), not weight.
        x32 = x.to(torch.float32)
        inv = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (x32 * inv * (1.0 + weight.to(device=x.device, dtype=torch.float32))).to(x.dtype)

    @staticmethod
    def rmsnorm_silu_gate(x: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
        y = CPUFallbackOps.rmsnorm(x, weight, eps).to(torch.float32)
        return (y * torch.nn.functional.silu(z.to(torch.float32))).to(x.dtype)

    @staticmethod
    def rmsnorm_qwen_silu_gate(x: torch.Tensor, weight: torch.Tensor, z: torch.Tensor, eps: float) -> torch.Tensor:
        y = CPUFallbackOps.rmsnorm_qwen(x, weight, eps).to(torch.float32)
        return (y * torch.nn.functional.silu(z.to(torch.float32))).to(x.dtype)

    @staticmethod
    def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)

    @staticmethod
    def silu_mul_packed(mixed: torch.Tensor, hidden: int) -> torch.Tensor:
        gate, up = torch.split(mixed, [int(hidden), int(hidden)], dim=-1)
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(mixed.dtype).contiguous()

    @staticmethod
    def sigmoid_mul(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.sigmoid(gate.float())).to(x.dtype).contiguous()

    @staticmethod
    def sigmoid_mul_repeat_kv(v: torch.Tensor, gate: torch.Tensor, ratio: int) -> torch.Tensor:
        return (v.repeat_interleave(int(ratio), dim=1).reshape(gate.size(0), -1).float() * torch.sigmoid(gate.reshape(gate.size(0), -1).float())).to(v.dtype).contiguous()

    @staticmethod
    def rmsnorm_qwen_pair_cat(x0: torch.Tensor, w0: torch.Tensor, x1: torch.Tensor, w1: torch.Tensor, eps: float) -> torch.Tensor:
        y0 = CPUFallbackOps.rmsnorm_qwen(x0, w0, eps)
        y1 = CPUFallbackOps.rmsnorm_qwen(x1, w1, eps)
        return torch.cat([y0, y1], dim=-1).contiguous()

    @staticmethod
    def rmsnorm_qwen_rope(x: torch.Tensor, weight: torch.Tensor, pos: int, rope_dim: int, rope_theta: float, eps: float) -> torch.Tensor:
        from langburst.adapters.qwen36_impl.model import apply_rope_single_tensor
        y = CPUFallbackOps.rmsnorm_qwen(x.reshape(-1, x.shape[-1]), weight, eps).reshape_as(x)
        return apply_rope_single_tensor(y, pos=int(pos), rope_dim=int(rope_dim), rope_theta=float(rope_theta)).contiguous()

    @staticmethod
    def gdn_recurrent(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        qf = qf * torch.rsqrt((qf * qf).sum(dim=-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt((kf * kf).sum(dim=-1, keepdim=True) + 1e-6)
        qf = qf * (qf.shape[-1] ** -0.5)
        vf = v.to(torch.float32)
        gf = g.to(torch.float32)
        bf = beta.to(torch.float32)
        state_f = state.to(torch.float32)
        out = torch.empty_like(vf)
        kv_ratio = vf.size(0) // kf.size(0)
        for h in range(vf.size(0)):
            kh = h // kv_ratio
            S = state_f[h]
            decay = torch.exp(gf[h])
            pred = torch.matmul(kf[kh], S)
            delta = (vf[h] - pred) * bf[h]
            S_new = S * decay + torch.outer(kf[kh], delta)
            state[h].copy_(S_new.to(state.dtype))
            out[h] = torch.matmul(qf[kh], state[h].to(torch.float32))
        return out.to(v.dtype)

    @staticmethod
    def gdn_recurrent_ab(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        beta = torch.sigmoid(b).to(torch.float16).contiguous()
        g = (-torch.exp(A_log.to(a.device)) * torch.nn.functional.softplus(a.float() + dt_bias.to(a.device))).contiguous()
        return CPUFallbackOps.gdn_recurrent(q, k, v, g, beta, state)

    @staticmethod
    def gdn_recurrent_ab_scan(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        outs = []
        for t in range(q.size(0)):
            outs.append(
                CPUFallbackOps.gdn_recurrent_ab(
                    q[t],
                    k[t],
                    v[t],
                    a[t],
                    b[t],
                    A_log,
                    dt_bias,
                    state,
                )
            )
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def gdn_recurrent_ab_batch(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> torch.Tensor:
        outs = []
        for row in range(q.size(0)):
            slot = int(state_indices[row].item())
            outs.append(
                CPUFallbackOps.gdn_recurrent_ab(
                    q[row],
                    k[row],
                    v[row],
                    a[row],
                    b[row],
                    A_log,
                    dt_bias,
                    state_arena[slot],
                )
            )
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def gdn_recurrent_ab_batch_norm_gate(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        norm_w: torch.Tensor,
        z: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        core = CPUFallbackOps.gdn_recurrent_ab_batch(q, k, v, a, b, A_log, dt_bias, state_arena, state_indices)
        return CPUFallbackOps.rmsnorm_silu_gate(
            core.reshape(core.size(0), -1).contiguous(),
            norm_w.reshape(-1).to(device=core.device, dtype=core.dtype).contiguous(),
            z.reshape(core.size(0), -1).to(device=core.device, dtype=core.dtype).contiguous(),
            eps,
        ).reshape_as(core)

    @staticmethod
    def gdn_recurrent_ab_spec(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        commit_tokens: torch.Tensor,
    ) -> torch.Tensor:
        outs = []
        for row in range(q.size(0)):
            slot = int(state_indices[row].item())
            commit_n = max(0, min(int(commit_tokens[row].item()), int(q.size(1))))
            scratch = state_arena[slot].clone()
            row_out = []
            committed = None
            for step in range(q.size(1)):
                row_out.append(
                    CPUFallbackOps.gdn_recurrent_ab(
                        q[row, step],
                        k[row, step],
                        v[row, step],
                        a[row, step],
                        b[row, step],
                        A_log,
                        dt_bias,
                        scratch,
                    )
                )
                if step + 1 == commit_n:
                    committed = scratch.clone()
            if committed is not None:
                state_arena[slot].copy_(committed)
            outs.append(torch.stack(row_out, dim=0).contiguous())
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def gdn_recurrent_ab_spec_trajectory(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outs = []
        trajectories = []
        for row in range(q.size(0)):
            slot = int(state_indices[row].item())
            scratch = state_arena[slot].clone()
            row_out = []
            row_traj = []
            for step in range(q.size(1)):
                row_out.append(
                    CPUFallbackOps.gdn_recurrent_ab(
                        q[row, step],
                        k[row, step],
                        v[row, step],
                        a[row, step],
                        b[row, step],
                        A_log,
                        dt_bias,
                        scratch,
                    )
                )
                row_traj.append(scratch.clone())
            outs.append(torch.stack(row_out, dim=0).contiguous())
            trajectories.append(torch.stack(row_traj, dim=0).contiguous())
        return torch.stack(outs, dim=0).contiguous(), torch.stack(trajectories, dim=0).contiguous()

    @staticmethod
    def gdn_recurrent_ab_spec_trajectory_out(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        out: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> None:
        out_ref, traj_ref = CPUFallbackOps.gdn_recurrent_ab_spec_trajectory(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state_arena,
            state_indices,
        )
        out.copy_(out_ref)
        trajectory.copy_(traj_ref)

    @staticmethod
    def copy_selected_trajectory_out(
        trajectory: torch.Tensor,
        dest: torch.Tensor,
        state_indices: torch.Tensor,
        commit_tokens: torch.Tensor,
    ) -> None:
        rows = int(trajectory.size(0))
        tokens = int(trajectory.size(1))
        for row in range(rows):
            slot = int(state_indices[row].item())
            commit_n = max(1, min(int(commit_tokens[row].item()), tokens))
            dest[slot].copy_(trajectory[row, commit_n - 1])

    @staticmethod
    def depthwise_conv_update(state: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 3:
            w = weight[:, 0, :]
        else:
            w = weight
        window = torch.cat([state, x[:, None]], dim=1)
        y = (window * w.to(device=x.device, dtype=x.dtype)).sum(dim=1)
        if bias.numel() > 0:
            y = y + bias.to(device=x.device, dtype=x.dtype)
        if state.numel() > 0:
            state[:, :-1] = state[:, 1:].clone()
            state[:, -1] = x
        return torch.nn.functional.silu(y)

    @staticmethod
    def depthwise_conv_update_scan(state: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        outs = []
        for row in x:
            outs.append(CPUFallbackOps.depthwise_conv_update(state, row.contiguous(), weight, bias))
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def depthwise_conv_update_batch(
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        outs = []
        for row in range(x.size(0)):
            slot = int(state_indices[row].item())
            outs.append(CPUFallbackOps.depthwise_conv_update(state_arena[slot], x[row].contiguous(), weight, bias))
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def depthwise_conv_update_spec(
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        commit_tokens: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        outs = []
        for row in range(x.size(0)):
            slot = int(state_indices[row].item())
            commit_n = max(0, min(int(commit_tokens[row].item()), int(x.size(1))))
            scratch = state_arena[slot].clone()
            row_out = []
            committed = None
            for step in range(x.size(1)):
                row_out.append(CPUFallbackOps.depthwise_conv_update(scratch, x[row, step].contiguous(), weight, bias))
                if step + 1 == commit_n:
                    committed = scratch.clone()
            if committed is not None:
                state_arena[slot].copy_(committed)
            outs.append(torch.stack(row_out, dim=0).contiguous())
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def depthwise_conv_update_spec_trajectory(
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outs = []
        trajectories = []
        for row in range(x.size(0)):
            slot = int(state_indices[row].item())
            scratch = state_arena[slot].clone()
            row_out = []
            row_traj = []
            for step in range(x.size(1)):
                row_out.append(CPUFallbackOps.depthwise_conv_update(scratch, x[row, step].contiguous(), weight, bias))
                row_traj.append(scratch.clone())
            outs.append(torch.stack(row_out, dim=0).contiguous())
            trajectories.append(torch.stack(row_traj, dim=0).contiguous())
        return torch.stack(outs, dim=0).contiguous(), torch.stack(trajectories, dim=0).contiguous()

    @staticmethod
    def depthwise_conv_update_spec_trajectory_out(
        state_arena: torch.Tensor,
        state_indices: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        out: torch.Tensor,
        trajectory: torch.Tensor,
    ) -> None:
        out_ref, traj_ref = CPUFallbackOps.depthwise_conv_update_spec_trajectory(
            state_arena,
            state_indices,
            x,
            weight,
            bias,
        )
        out.copy_(out_ref)
        trajectory.copy_(traj_ref)

    @staticmethod
    def attention_decode_fp16(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, length: int, scale: float) -> torch.Tensor:
        qf = q.to(torch.float32)
        kf = k_cache[:, :length, :].to(torch.float32)
        vf = v_cache[:, :length, :].to(torch.float32)
        q_heads, dim = qf.shape
        kv_heads = kf.shape[0]
        ratio = q_heads // kv_heads
        out = torch.empty((q_heads, dim), device=q.device, dtype=torch.float32)
        for h in range(q_heads):
            kh = h // ratio
            scores = torch.matmul(kf[kh], qf[h]) * float(scale)
            probs = torch.softmax(scores, dim=0)
            out[h] = torch.matmul(probs, vf[kh])
        return out.to(q.dtype)


    @staticmethod
    def attention_decode_fp16_gated_tkh(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, gate: torch.Tensor, length: int, scale: float) -> torch.Tensor:
        k_live = k_cache[: int(length)].permute(1, 0, 2).contiguous()
        v_live = v_cache[: int(length)].permute(1, 0, 2).contiguous()
        att = CPUFallbackOps.attention_decode_fp16(q, k_live, v_live, int(length), float(scale))
        return CPUFallbackOps.sigmoid_mul(att.reshape(-1), gate.reshape(-1))

    @staticmethod
    def attention_decode_batch_fp16(
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_arena: torch.Tensor,
        v_arena: torch.Tensor,
        state_indices: torch.Tensor,
        write_indices: torch.Tensor,
        live_lengths: torch.Tensor,
        positions: torch.Tensor,
        use_ring: bool,
        scale: float,
    ) -> torch.Tensor:
        outs = []
        max_seq = k_arena.size(2)
        for row in range(q.size(0)):
            slot = int(state_indices[row].item())
            write_idx = int(write_indices[row].item())
            k_arena[slot, :, write_idx, :].copy_(k_new[row])
            v_arena[slot, :, write_idx, :].copy_(v_new[row])
            length = min(int(live_lengths[row].item()), max_seq)
            if use_ring and length == max_seq:
                start = (int(positions[row].item()) + 1) % max_seq
                indices = [(start + i) % max_seq for i in range(length)]
                k_cache = k_arena[slot, :, indices, :].contiguous()
                v_cache = v_arena[slot, :, indices, :].contiguous()
            else:
                k_cache = k_arena[slot]
                v_cache = v_arena[slot]
            outs.append(CPUFallbackOps.attention_decode_fp16(q[row], k_cache, v_cache, length, scale))
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def attention_decode_paged_fp16(
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        scale: float,
    ) -> torch.Tensor:
        outs = []
        for row in range(q.size(0)):
            slot = int(slot_mapping[row].item())
            block = slot // int(block_size)
            offset = slot % int(block_size)
            k_pages[block, :, offset, :].copy_(k_new[row])
            v_pages[block, :, offset, :].copy_(v_new[row])
            length = int(seq_lens[row].item())
            blocks = block_tables[row]
            k_rows = []
            v_rows = []
            for pos in range(length):
                block_idx = pos // int(block_size)
                block_offset = pos % int(block_size)
                block_id = int(blocks[block_idx].item())
                k_rows.append(k_pages[block_id, :, block_offset, :])
                v_rows.append(v_pages[block_id, :, block_offset, :])
            k_cache = torch.stack(k_rows, dim=1).contiguous()
            v_cache = torch.stack(v_rows, dim=1).contiguous()
            outs.append(CPUFallbackOps.attention_decode_fp16(q[row], k_cache, v_cache, length, scale))
        return torch.stack(outs, dim=0).contiguous()

    @staticmethod
    def attention_paged_int4_flash(
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        k_zeros: torch.Tensor,
        v_zeros: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
        scale: float,
        hadamard_order: int,
        bdr_k: bool,
        rotate_v: bool,
        tiled_layout: bool = False,
    ) -> torch.Tensor:
        raise RuntimeError("INT4 paged attention requires the CUDA extension")

    @staticmethod
    def attention_append_paged_int4(
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        k_zeros: torch.Tensor,
        v_zeros: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
        hadamard_order: int,
        bdr_k: bool,
        rotate_v: bool,
        tiled_layout: bool = False,
    ) -> None:
        from .core.kv_cache import hadamard_transform, pack_int4_rows

        for row in range(k_new.size(0)):
            slot = int(slot_mapping[row].item())
            block = slot // int(block_size)
            offset = slot % int(block_size)
            k_store = hadamard_transform(k_new[row], hadamard_order) if bdr_k else k_new[row]
            v_store = hadamard_transform(v_new[row], hadamard_order) if bdr_k and rotate_v else v_new[row]
            k_packed, k_scale, k_zero = pack_int4_rows(k_store)
            v_packed, v_scale, v_zero = pack_int4_rows(v_store)
            if tiled_layout:
                k_pages[block, :, :, offset].copy_(k_packed)
                v_pages[block, :, :, offset].copy_(v_packed)
            else:
                k_pages[block, :, offset, :].copy_(k_packed)
                v_pages[block, :, offset, :].copy_(v_packed)
            k_scales[block, :, offset].copy_(k_scale)
            v_scales[block, :, offset].copy_(v_scale)
            k_zeros[block, :, offset].copy_(k_zero)
            v_zeros[block, :, offset].copy_(v_zero)

    @staticmethod
    def attention_append_paged_int4_spec(
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        k_zeros: torch.Tensor,
        v_zeros: torch.Tensor,
        slot_mapping: torch.Tensor,
        commit_tokens: torch.Tensor,
        block_size: int,
        hadamard_order: int,
        bdr_k: bool,
        rotate_v: bool,
        tiled_layout: bool = False,
    ) -> None:
        from .core.kv_cache import hadamard_transform, pack_int4_rows

        for row in range(k_new.size(0)):
            commit_n = max(0, min(int(commit_tokens[row].item()), int(k_new.size(1))))
            for step in range(commit_n):
                slot = int(slot_mapping[row, step].item())
                block = slot // int(block_size)
                offset = slot % int(block_size)
                k_store = hadamard_transform(k_new[row, step], hadamard_order) if bdr_k else k_new[row, step]
                v_store = hadamard_transform(v_new[row, step], hadamard_order) if bdr_k and rotate_v else v_new[row, step]
                k_packed, k_scale, k_zero = pack_int4_rows(k_store)
                v_packed, v_scale, v_zero = pack_int4_rows(v_store)
                if tiled_layout:
                    k_pages[block, :, :, offset].copy_(k_packed)
                    v_pages[block, :, :, offset].copy_(v_packed)
                else:
                    k_pages[block, :, offset, :].copy_(k_packed)
                    v_pages[block, :, offset, :].copy_(v_packed)
                k_scales[block, :, offset].copy_(k_scale)
                v_scales[block, :, offset].copy_(v_scale)
                k_zeros[block, :, offset].copy_(k_zero)
                v_zeros[block, :, offset].copy_(v_zero)

    @staticmethod
    def attention_spec_decode_paged_int4(
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        k_pages: torch.Tensor,
        v_pages: torch.Tensor,
        k_scales: torch.Tensor,
        v_scales: torch.Tensor,
        k_zeros: torch.Tensor,
        v_zeros: torch.Tensor,
        block_tables: torch.Tensor,
        base_seq_lens: torch.Tensor,
        block_size: int,
        scale: float,
        hadamard_order: int,
        bdr_k: bool,
        rotate_v: bool,
        tiled_layout: bool = False,
    ) -> torch.Tensor:
        raise RuntimeError("INT4 speculative paged attention requires the CUDA extension")

    @staticmethod
    def argmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits).to(torch.long)

    @staticmethod
    def argmax_many(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1).to(torch.long)

    @staticmethod
    def argmax_many_out(logits: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(torch.argmax(logits, dim=-1).to(torch.long))

    @staticmethod
    def count_prefix_matches(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = min(a.numel(), b.numel())
        k = 0
        for i in range(n):
            if int(a[i].item()) != int(b[i].item()):
                break
            k += 1
        return torch.tensor([k], device=a.device, dtype=torch.long)

    @staticmethod
    def resolve_greedy_speculative(
        draft_token_ids: torch.Tensor,
        target_token_ids: torch.Tensor,
        bonus_token_ids: torch.Tensor,
        cu_num_draft_tokens: torch.Tensor,
        scheduled_token_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = draft_token_ids.device
        counts = cu_num_draft_tokens.to(device="cpu", dtype=torch.int32).tolist()
        scheduled = scheduled_token_counts.to(device="cpu", dtype=torch.int32).tolist()
        batch = len(counts)
        prev = 0
        max_draft = 0
        for cur in counts:
            max_draft = max(max_draft, int(cur) - prev)
            prev = int(cur)
        token_matrix = torch.full((batch, max_draft + 1), -1, device=device, dtype=torch.long)
        sampled_counts = torch.empty((batch,), device=device, dtype=torch.int32)
        rejected_counts = torch.empty((batch,), device=device, dtype=torch.int32)
        accepted_counts = torch.empty((batch,), device=device, dtype=torch.int32)
        prev = 0
        for row, cur_raw in enumerate(counts):
            cur = int(cur_raw)
            draft = draft_token_ids[prev:cur].to(dtype=torch.long)
            target = target_token_ids[prev:cur].to(device=device, dtype=torch.long)
            accepted = 0
            for i in range(int(draft.numel())):
                if int(draft[i].item()) != int(target[i].item()):
                    break
                accepted += 1
            if accepted:
                token_matrix[row, :accepted] = draft[:accepted]
            if accepted < int(draft.numel()):
                token_matrix[row, accepted] = target[accepted]
            else:
                token_matrix[row, accepted] = bonus_token_ids[row].to(device=device, dtype=torch.long)
            sampled = accepted + 1
            sampled_counts[row] = sampled
            rejected_counts[row] = max(0, int(scheduled[row]) - sampled)
            accepted_counts[row] = accepted
            prev = cur
        return token_matrix, sampled_counts, rejected_counts, accepted_counts


@lru_cache(maxsize=1)
def cuda_ops():
    force_cpu = os.environ.get("LANGBURST_CPU_FALLBACK", "0") == "1" or os.environ.get("LANGBURST_SKIP_CUDA_EXT", "0") == "1"
    if not force_cpu:
        try:
            return importlib.import_module("langburst_cuda")
        except Exception as exc:  # pragma: no cover
            if os.environ.get("LANGBURST_REQUIRE_CUDA_EXT", "0") == "1":
                raise RuntimeError(
                    "langburst_cuda is not built. Build on the RTX 4080 box with:\n"
                    "  LANGBURST_REQUIRE_CUDA_EXT=1 pip install -v -e ."
                ) from exc
    return CPUFallbackOps()
