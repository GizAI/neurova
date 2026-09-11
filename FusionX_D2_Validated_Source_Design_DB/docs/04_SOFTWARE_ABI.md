# Software ABI

`abi/fusionx_d2_abi.json` is the source of truth. `tools/gen_abi.py` generates
both C and SystemVerilog constants.

- SQ entry: 64-byte descriptor.
- CQ entry: 32-byte completion; two completions share a 64-byte coherent beat.
- Device reads `SQ_TAIL`, writes `SQ_HEAD`.
- Host reads `SQ_HEAD`, writes `SQ_TAIL` and the SQ doorbell.
- Device writes `CQ_TAIL`; host reads it, consumes entries, writes `CQ_HEAD`.

The previous use of `completion_head` as the submission head and
`completion_size` as the completion tail has been removed.
