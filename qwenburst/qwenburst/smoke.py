from __future__ import annotations

import torch

from .ops import cuda_ops
from .config import Qwen36_27B_TextConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for smoke test")
    ops = cuda_ops()
    cfg = Qwen36_27B_TextConfig()
    x = torch.randn(cfg.hidden_size, device="cuda", dtype=torch.float16)
    w = torch.ones(cfg.hidden_size, device="cuda", dtype=torch.float16)
    y = ops.rmsnorm(x, w, cfg.rms_norm_eps)
    print("rmsnorm", y.shape, y.dtype)

    q = torch.randn(cfg.linear_num_key_heads, 128, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(cfg.linear_num_value_heads, 128, device="cuda", dtype=torch.float16)
    g = torch.full((cfg.linear_num_value_heads,), -0.1, device="cuda", dtype=torch.float32)
    beta = torch.full((cfg.linear_num_value_heads,), 0.5, device="cuda", dtype=torch.float16)
    state = torch.zeros(cfg.linear_num_value_heads, 128, 128, device="cuda", dtype=torch.float16)
    out = ops.gdn_recurrent(q, k, v, g, beta, state)
    print("gdn", out.shape, state.norm().item())


if __name__ == "__main__":
    main()
