# Verification plan and evidence

## Executed in this package build

- FP4/FP8 finite-value and NaN tests.
- Packed 4-bit/8-bit/16-bit decode tests.
- Signed RNE tie-to-even tests, including negative ties.
- Random 64x64x3 reference GEMM and all 128 `(row, beat)` output positions.
- M/N edge-tile traversal.
- Stream primitive numerical tests.
- C SQ/CQ runtime test.
- C++ numeric test.
- Compiler lowering for six model profiles.
- Generated C/SV ABI consistency.
- Audit-contract source assertions.

## Supplied HDL regressions

- `tb_lowbit`
- `tb_operand_unpacker`
- `tb_mxu_64x64`
- `tb_tensor_engine`
- `tb_stream_primitives`
- `tb_csr`
- `tb_async_fifo`

`make hdl` requires Icarus, Verilator, and Yosys and fails when they are not
installed. No static source check is labeled as HDL compilation.

## Required before product RTL freeze

Commercial lint, CDC/RDC, SVA/formal, UVM constrained-random regressions,
requirements coverage, code coverage and waivers, gate-level/SDF simulation,
emulation, and full-checkpoint numerical/quality correlation.
