# FusionX D2 final validation report

## Classification

**Integrated validated source/reference design database — not a foundry manufacturing release.**

## Results

| Gate | Result |
|---|---|
| Source/ABI/compiler/numerical regression | **PASS** |
| Audit-contract checks | **PASS (23/23)** |
| Real HDL tool gate | **BLOCKED_TOOL_MISSING** |
| N3P foundry release | **NO** |

## Source metrics

- RTL files: 27
- Module definitions: 29
- Package definitions: 2
- RTL lines: 1,117

## Executed numerical and ABI evidence

- Correct FP4 E2M1 finite values through +/-6.
- Correct FP8 E4M3FN finite values through +/-448 and NaN handling.
- Packed 4-bit, 8-bit and 16-bit operand tests.
- Signed round-to-nearest-even tests including negative ties.
- Random 64x64x3 GEMM reference and all 128 output `(row, beat)` positions.
- M/N ragged-edge tile traversal.
- RMSNorm, base-e Softmax, RoPE, recurrent-state, Top-K, MTP, gather, Patchify3D and AdaLN reference tests.
- C runtime SQ/CQ ownership regression and C/SV generated ABI consistency.
- Qwen, MiniMax, GLM and DeepSeek high-level graph legalization into primitives.

## HDL evidence boundary

The package contains directed SystemVerilog tests and fail-closed Icarus,
Verilator and Yosys scripts. Those tools are not installed in the current
execution environment, so the real HDL gate is recorded as
`BLOCKED_TOOL_MISSING` and no compilation/simulation/synthesis PASS is claimed.

## Foundry boundary

A real N3P tapeout still requires licensed process/IP/package collateral and all
front-end, DFT, physical and package sign-off outputs listed in
`foundry/N3P_HANDOFF_REQUIREMENTS.yaml`.
