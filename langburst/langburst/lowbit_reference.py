from __future__ import annotations

import numpy as np
import torch


def dequant_symmetric_lowbit_cpu(packed: np.ndarray, scales: np.ndarray, cols: int, group_size: int, bits: int) -> torch.Tensor:
    """Dequantize LangBurst symmetric groupwise low-bit layout on CPU."""
    rows = packed.shape[0]
    out = torch.empty(rows, cols, dtype=torch.float32)
    qmask = (1 << bits) - 1
    zero = 1 << (bits - 1)
    for r in range(rows):
        for c in range(cols):
            bit_pos = c * bits
            byte_i = bit_pos // 8
            shift = bit_pos % 8
            word = int(packed[r, byte_i])
            if shift + bits > 8 and byte_i + 1 < packed.shape[1]:
                word |= int(packed[r, byte_i + 1]) << 8
            q = ((word >> shift) & qmask) - zero
            out[r, c] = q * float(scales[r, c // group_size])
    return out
