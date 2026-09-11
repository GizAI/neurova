# External audit closure matrix

| Finding | D2 disposition |
|---|---|
| 64x64 row advances once per output beat | Closed: independent row and beat counters; row advances on final beat only. |
| Real tests used COLS=8 and hid the bug | Closed in supplied 64x64/128-beat HDL TB and executable Python regression. HDL TB awaits external tool execution. |
| Qwen operators were disconnected identities | Closed for the D2 primitive set; no identity fallback for accepted opcodes. |
| Toy MLA/Conv3D/VAE blocks were mislabeled native | Closed by removing those high-level opcodes; compiler lowers model nodes to primitives. |
| FP4 exponent 11 was zeroed | Closed: values 4 and 6 are finite. |
| FP8 exponent 15 was all zeroed | Closed: E4M3FN finite encodings through 448; only NaN encoding rejected. |
| No scale/requant path | Closed: input scale, signed RNE, saturation, wide accumulator and output requantization. |
| 4/8-bit weights consumed 16-bit slots | Closed: packed nibble/byte operand loader. |
| GEMM output could not feed stream engine | Closed: canonical Q8.8 tensor output is the stream input ABI. |
| Runtime and CSR queue ABI disagreed | Closed with JSON-generated C/SV ABI and corrected SQ/CQ ownership. |
| 64 MiB scratchpad was not in compute path | Closed structurally: 32-bank scratchpad is behind active tensor/stream memory router. SRAM macro binding remains foundry-gated. |
| Unsafe host/core crossings | Closed structurally with async command/completion FIFOs and synchronized status. Commercial CDC/RDC remains a release gate. |
| Missing HDL tools were swallowed | Closed: `make verify` returns non-zero when HDL tools/tests are absent. |
| 18.45 POPS implied by unproven multiplier replication | Claim withdrawn. D2 reports only instantiated RTL parameters and requires PPA evidence for product scaling. |
| No PDK/IP/physical/DFT/sign-off | Not source-solvable; explicitly blocked in the foundry handoff manifest. |
