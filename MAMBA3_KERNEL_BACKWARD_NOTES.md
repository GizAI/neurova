# Mamba-3 Kernel And Backward Notes

Date: 2026-06-13

This file records the current source-level understanding of official Mamba-3 training efficiency.

## Priority

Use official Mamba kernel-level backward recomputation first. Do not wrap Mamba-3 blocks with generic PyTorch activation checkpointing unless the custom autograd path is changed and reverified.

## What The Local Official Source Does

The active Mamba-3 MIMO path is:

```text
mamba_ssm/modules/mamba3.py
  -> mamba_ssm/ops/tilelang/mamba3/mamba3_mimo.py
  -> mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
```

The TileLang backward file explicitly implements a two-pass backward:

- a fused backward-forward pass over chunks,
- recomputation of local forward intermediates,
- storage of per-chunk recurrent states and QK dot products,
- a reverse-chunk backward pass.

This is the correct Mamba-style memory strategy. It is not the same as saving every forward activation.

## Current Bottleneck

On RTX 4080 16GB, larger dense Mamba-3 MIMO-R4 candidates fail in backward with dynamic shared-memory limits:

```text
mimo-r4-16gb-120m   Failed to set allowed dynamic shared memory size to 123552
mimo-r4-16gb-180m   Failed to set allowed dynamic shared memory size to 123552
mimo-r4-paper-180m  Failed to set allowed dynamic shared memory size to 223904
```

The failure is not ordinary VRAM OOM. It is per-kernel dynamic shared memory. That means optimizer tricks, gradient accumulation, and generic activation checkpointing do not solve this specific dense backward failure.

## What Not To Do

Do not use block-level PyTorch activation checkpointing around Mamba-3 TileLang blocks. It failed locally with:

```text
torch.utils.checkpoint.CheckpointError:
Unpack is being triggered for a tensor that was already unpacked once.
```

That failure comes from the custom autograd function's saved tensor behavior. The CLI now rejects `--activation-checkpointing` for Mamba-3.

## Viable Paths

P0:

- Keep official kernel-level recomputation as the default.
- Keep dense trainable candidates within the kernel shared-memory limit.
- Use sparse MoE in the MLP branch to increase total parameters without increasing the Mamba-3 MIMO kernel dimensions.
- Continue measuring forward, decode-step parity, and backward for every candidate.

P1:

- Reduce TileLang MIMO backward shared-memory footprint:
  - smaller `headdim`,
  - smaller `d_state`,
  - smaller `chunk_size`,
  - graph/tiling changes inside `mamba3_mimo_bwd.py`,
  - or a hardware target with larger dynamic shared-memory allowance.
- Add source-level probes that record which kernel config produces the shared-memory request.

P2:

- Revisit generic checkpointing only if the Mamba-3 autograd wrapper is redesigned to support it.
- Explore Triton SISO fallback only as a baseline; it does not satisfy the MIMO-R4 target.
