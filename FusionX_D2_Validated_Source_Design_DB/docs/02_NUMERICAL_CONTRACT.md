# Numerical contract

## Memory encodings

- INT4 and FP4 E2M1: two values per byte.
- INT8 and FP8 E4M3FN: one value per byte.
- INT16 and BF16 ingress fallback: one value per 16-bit little-endian word.

## FP4 E2M1

Finite magnitudes are `0, 0.5, 1, 1.5, 2, 3, 4, 6`, with a sign bit. There is
no Inf/NaN encoding in the FP4 path.

## FP8 E4M3FN

Exponent `1111` remains finite for mantissas `000` through `110`; mantissa
`111` is treated as NaN. Maximum finite magnitude is 448.

## Internal formats

- Tensor and stream canonical activation: signed Q8.8 (`int16`).
- Tensor product accumulation: signed wide integer corresponding to Q16.16.
- Softmax output: unsigned Q0.16 in a 16-bit lane.
- State-update gate: signed Q1.15.
- RoPE cosine/sine and AdaLN scale: signed Q8.8 in the current reference block.

## Rounding

Scaling and output quantization use signed-magnitude round-to-nearest-even,
followed by saturation. Internal iterative approximations have their own
operator-specific contract; there is no false universal-rounding claim.
