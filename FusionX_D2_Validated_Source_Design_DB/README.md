<https://chatgpt.com/c/6a8486f6-da50-83e8-92a4-cc9b9f457df0>

# FusionX D2 Validated Front-End Design Database

FusionX D2 is a clean-room replacement for the previously audited D1 package.
It is an **integrated, synthesizable-intent front-end reference design**, numerical
reference model, compiler legalization layer, software ABI, and fail-closed
verification harness.

It is **not** a TSMC N3P foundry release and does not contain proprietary PDK,
HBM4/PCIe/UCIe/PLL hard-IP views, SRAM compiler macros, package sign-off, or a
GDS/OASIS database.

## D2 fixes that are implemented in source

* 64x64 result readout holds each row until every 512-bit beat is accepted.
* GEMM executes explicit M/N/K tile traversal and edge masks.
* FP4/INT4 and FP8/INT8 are read from packed 4-bit/8-bit memory layouts.
* FP4 E2M1 finite values include +/-4 and +/-6; FP8 E4M3FN includes finite
  values through +/-448 and reserves only the NaN encoding.
* Per-tensor scaling, round-to-nearest-even, saturation, and accumulator-to-Q8.8
  requantization are part of the active tensor path.
* RMSNorm, base-e Softmax, RoPE, recurrent state update, Top-K, MTP prefix
  verification, indexed gather, Patchify3D, and AdaLN are active stream
  primitives; unsupported high-level names do not silently become identity.
* High-level Qwen/MiniMax/GLM/DeepSeek operations are compiler recipes made from
  the validated primitive set, not a dictionary-only `native` claim.
* The compute path can address a 32-bank, 64 MiB-per-die scratchpad target.
* SQ/CQ CSR offsets, C structures, SystemVerilog constants, and runtime logic
  share the generated ABI; submission and completion head/tail directions are
  correct.
* Host/core commands and completions use asynchronous FIFOs; asynchronous status
  bits are synchronized before use.
* `make verify` is fail-closed: missing HDL tools produce a non-zero result.

## Commands

```bash
make source-test   # executable Python/C/C++/ABI/compiler tests
make hdl           # Icarus simulation + Verilator lint + Yosys reduced synthesis
make verify        # source-test and hdl; fails if any tool or test is missing
make package       # rebuild release ZIP/TAR and checksums after all required gates
```

The current execution environment did not provide Icarus, Verilator, or Yosys.
Therefore source-level tests are recorded as PASS, while the HDL release gate is
recorded as BLOCKED rather than misreported as PASS.

See `STATUS.md`, `docs/07_AUDIT_CLOSURE.md`, and
`foundry/N3P_HANDOFF_REQUIREMENTS.yaml` for the exact claim boundary.
