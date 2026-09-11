# Memory and GEMM contract

A and B are compiler-blocked into logical 64-element vectors. Packed operand
beats are decoded by `fx_operand_unpacker`. The tensor engine traverses all
M/N/K tiles and clears the accumulator only at the start of each M/N output
tile.

The output of each logical row is requantized to 16-bit Q8.8. A 64-column row
therefore consists of exactly two 512-bit beats. `fx_mxu_array` advances the row
only after the second beat handshakes, closing the previous row/beat corruption
bug.

Local scratchpad addresses have bit 63 set. External/HBM addresses have bit 63
clear. The integrated router allows the tensor and stream engine to use the
same 32-bank scratchpad target or the external controller boundary.
