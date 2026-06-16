from __future__ import annotations

import torch


def l2_normalize_last(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x.float() * torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps)


@torch.no_grad()
def gdn_recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    *,
    inplace: bool = True,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference single-token Qwen-style gated delta recurrence.

    This mirrors csrc/gdn_recurrent.cu, including fp16 state write-back before
    producing the output. It is intentionally small and deterministic so the CUDA
    kernel can be parity-tested without depending on HF/FLA internals.

    Shapes:
      q/k:   [kv_heads, 128]
      v:     [v_heads, 128]
      g:     [v_heads] fp32, log-decay preactivation already transformed
      beta:  [v_heads]
      state: [v_heads, 128, 128]
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q/k/v must be rank-2")
    if state.ndim != 3:
        raise ValueError("state must be [v_heads, 128, 128]")
    kv_heads, d = q.shape
    v_heads, dv = v.shape
    if d != 128 or dv != 128 or k.shape != q.shape:
        raise ValueError("this reference is specialized to head_dim=128")
    if v_heads % kv_heads:
        raise ValueError("v_heads must be divisible by kv_heads")
    if state.shape != (v_heads, 128, 128):
        raise ValueError(f"bad state shape: {tuple(state.shape)}")

    out_state = state if inplace else state.clone()
    qn = l2_normalize_last(q, eps=eps)
    kn = l2_normalize_last(k, eps=eps)
    qn = qn * (q.shape[-1] ** -0.5)
    beta_f = beta.float()
    g_f = g.float()
    ratio = v_heads // kv_heads
    outputs = []
    for vh in range(v_heads):
        kh = vh // ratio
        s_old = out_state[vh].float()
        kvec = kn[kh]
        qvec = qn[kh]
        old = kvec @ s_old
        delta = v[vh].float() - old
        s_new = torch.exp(g_f[vh]) * s_old + beta_f[vh] * kvec[:, None] * delta[None, :]
        # The CUDA kernel stores fp16 state, then reads that fp16 state for out.
        out_state[vh].copy_(s_new.to(out_state.dtype))
        outputs.append(qvec @ out_state[vh].float())
    out = torch.stack(outputs, dim=0).to(q.dtype)
    return out, out_state
