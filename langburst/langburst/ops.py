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
