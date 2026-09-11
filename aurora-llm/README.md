# AURORA-LLM

A runnable x86-64 assembly Llama/Mistral inference microserver designed for Linux/KVM CPU experiments and aggressively tuned CPU decode.

The **serving runtime is 100% assembly** (GNU `as`, Intel syntax): no libc, pthreads, Python, PyTorch, oneDNN, BLAS, or dynamic libraries. Offline weight conversion and test/client utilities are Python.

## Implemented

- direct Linux x86-64 syscalls only; static ELF runtime
- runtime AVX-512F/DQ detection with AVX2/FMA fallback
- AVX-512 Q8 matvec with adaptive row batching and 64-input unrolling
- lossless **ALI v2 8-row × 16-input interleaved Q8 layout** for contiguous weight streaming
- persistent multicore worker pool created with raw `fork`
- CPUID/affinity-aware physical-core-first worker placement and target auto-tuning
- race-free shared-memory spin dispatcher; no syscall or allocation per matvec
- fixed inference arenas, persistent K/V cache
- AVX-512 vectorized SiLU, softmax exponent/normalization, residual/dequant paths and attention value accumulation
- deterministic RMSNorm reduction, Llama-style RoPE and grouped-query attention
- SwiGLU FFN, Q8 LM head and greedy sampler
- HTTP JSON token-ID API
- local HF Llama/Mistral Safetensors -> ALI Q8 packer
- initramfs appliance builder for KVM experiments

## Build and self-test

```bash
make
python3 tools/selftest.py
```

The self-test generates a tiny transformer, losslessly repacks it from ALI v1 to v2, serves both layouts with 1, 2 and 4 workers, and verifies identical generated token IDs across layouts and worker counts.

## Tiny test model

```bash
python3 tools/make_test_model.py test-v1.ali
python3 tools/repack_ali_v2.py test-v1.ali test-v2.ali
./aurora-llm test-v2.ali 4 8080
```

Request:

```bash
curl -s http://127.0.0.1:8080/v1/token-completions \
  -H 'content-type: application/json' \
  -d '{"tokens":[1,2,3],"max_tokens":4}'
```

## Pack a real local Llama/Mistral checkpoint

The offline packer requires `numpy`, `ml_dtypes`, and `safetensors` (including BF16 checkpoints such as MiniCPM5). `pack_hf.py` emits the portable row-major ALI v1 representation; `repack_ali_v2.py` changes **layout only** and preserves every Q8 scale and int8 weight bit.

```bash
python3 tools/pack_hf.py /models/MyLlama model-v1.ali --max-seq 4096
python3 tools/repack_ali_v2.py model-v1.ali model-v2.ali
./aurora-llm model-v2.ali
```

MiniCPM5-1B example:

```bash
python3 tools/pack_hf.py models/MiniCPM5-1B models/minicpm5-1b-q8.ali --max-seq 2048
python3 tools/repack_ali_v2.py models/minicpm5-1b-q8.ali models/minicpm5-1b-q8-i8.ali
./aurora-llm models/minicpm5-1b-q8-i8.ali
python3 tools/client.py models/MiniCPM5-1B 'What is the capital of South Korea?' --chat --max-tokens 16
```

When `WORKERS` is omitted, the runtime reads the current CPU affinity mask, uses CPUID topology to place workers on distinct physical cores before SMT siblings, and applies the measured AVX-512 decode sweet spot for ALI v2. An explicit worker count is still accepted and is clamped to CPUs available in the affinity mask.

For convenient text I/O while keeping tokenization outside the assembly runtime:

```bash
python3 tools/client.py /models/MyLlama 'Explain NUMA in one paragraph.'
```

## Benchmark

With a server already running:

```bash
python3 tools/bench.py --runs 20 --prompt-len 1 --max-tokens 64
```

`bench.py` reports throughput from the **actual returned token count**, not the requested maximum.

## ALI v2 layout

Embeddings stay row-major because token lookup is random. Dense transformer and LM-head matrices are grouped in blocks of eight output rows. Each block stores the eight original FP32 Q8 scales in one header cache line, followed by consecutive 8×16 int8 tiles. This removes per-row padding and turns eight distant weight streams into one sequential stream while keeping the model values unchanged. ALI v1 remains supported by the same runtime.

## Minimal KVM appliance

```bash
./tools/make_initramfs.sh model-v2.ali aurora-llm.cpio.gz
```

See [`docs/KVM.md`](docs/KVM.md). When executed as `/init`, the runtime defaults to `/model.ali` and listens on port 8080.

## Current boundary

This is a decode-oriented inference core rather than a production vLLM replacement: dense Llama/Mistral-family models, Q8 weights + FP32 activations, one request at a time, greedy decode and a token-ID API. Continuous batching, speculative decoding, lower-bit weight formats, raw-text tokenizer support, multi-NUMA replicas and production HTTP parsing remain outside the current core.

Performance claims should always be tied to the target CPU, affinity/cpuset and model; use the included benchmark and parity tests on the deployment machine.
