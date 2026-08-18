# AURORA-LLM v0.1 architecture

## Hot path

```
HTTP token request
  -> fixed request buffer
  -> prompt token IDs
  -> embedding Q8 dequant
  -> layer loop
       RMSNorm
       Q/K/V Q8 matvec
       RoPE
       GQA attention + KV cache
       output Q8 matvec + residual
       RMSNorm
       gate/up Q8 matvec
       SiLU * up
       down Q8 matvec + residual
  -> final RMSNorm
  -> Q8 LM head
  -> greedy argmax
  -> JSON token IDs
```

## Multicore engine

The coordinator and every forked worker are permanently alive. A matrix-vector job is written into one shared 128-byte job descriptor. Each worker owns a deterministic row range:

```
start = rows * worker_id / workers
end   = rows * (worker_id + 1) / workers
```

The hot matrix path uses shared memory, `PAUSE`, and an atomic completion counter; there is no pthread runtime, futex, scheduler wake-up, allocation, or syscall per matvec.

## Weight format

The 64-byte ALI v1 header records the checkpoint's `rms_norm_eps` at byte
offset 44 and attention `head_dim` at byte offset 48. Zero values preserve
the original `1e-5` and `hidden_size / heads` defaults for legacy files;
newly packed files preserve the checkpoint values. Offsets 52 and 56 hold up
to two EOS token IDs, and offset 60 holds the EOS count. A zero count disables
EOS stopping for legacy files.

Each matrix is stored row-major. A row begins with a float32 scale followed by signed int8 values and is padded to 64 bytes:

```
[f32 scale][int8 w0 ... wN-1][padding to 64-byte row stride]
```

The AVX2 kernel loads eight int8 weights, sign-extends to int32, converts to float, applies the row scale and accumulates with FMA against float activations. This is decode-oriented and intentionally simple. The next performance tier should add AVX-512/VNNI W8A8 and AMX batched-prefill kernels.

## Deliberate v0.1 constraints

- x86-64 AVX2 + FMA
- Llama/Mistral dense architecture
- Q8 weights, float activations
- token-ID API; tokenizer is outside the serving runtime
- greedy decoding only
- one request at a time; all cores cooperate on that request
- no continuous batching yet
- approximate scalar exp for softmax/SiLU

These constraints make the current artifact a real, testable inference core, not yet a vLLM replacement.
