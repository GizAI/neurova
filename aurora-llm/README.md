# AURORA-LLM v0.1

A runnable x86-64 assembly Llama/Mistral inference microserver designed for Linux/KVM CPU experiments.

The **serving runtime is 100% assembly** (GNU `as`, Intel syntax): no libc, pthreads, Python, PyTorch, oneDNN, BLAS, or dynamic libraries. Offline weight conversion and test/client utilities are Python.

## Implemented

- direct Linux x86-64 syscalls only
- static 13–20 KB ELF runtime
- AVX2/FMA row-wise Q8 weight kernel
- persistent multicore worker pool created with raw `fork`
- deterministic row sharding across cores
- shared-memory spin dispatcher; no syscall or allocation per matvec
- fixed inference arenas
- RMSNorm
- Llama-style RoPE
- grouped-query attention
- persistent K/V cache
- SwiGLU FFN
- residual path
- Q8 LM head + greedy sampler
- HTTP JSON token-ID API
- local HF Llama/Mistral Safetensors -> ALI Q8 packer
- initramfs appliance builder for KVM experiments

## Build and self-test

```bash
make
python3 tools/selftest.py
```

The self-test generates a tiny transformer, serves it with 1, 2 and 4 workers, and verifies bit-for-bit identical generated token IDs across worker counts.

## Tiny test model

```bash
python3 tools/make_test_model.py test.ali
./aurora-llm test.ali 4 8080
```

Request:

```bash
curl -s http://127.0.0.1:8080/v1/token-completions \
  -H 'content-type: application/json' \
  -d '{"tokens":[1,2,3],"max_tokens":4}'
```

## Pack a real local Llama/Mistral checkpoint

Requires `numpy`, `ml_dtypes`, and `safetensors` only for conversion (including
BF16 checkpoints such as MiniCPM5):

```bash
python3 tools/pack_hf.py /models/MyLlama model.ali --max-seq 4096
./aurora-llm model.ali 32 8080
```

For example, MiniCPM5-1B can be packed with a bounded 2048-token KV cache:

```bash
python3 tools/pack_hf.py models/MiniCPM5-1B models/minicpm5-1b-q8.ali --max-seq 2048
./aurora-llm models/minicpm5-1b-q8.ali 4 8080
python3 tools/client.py models/MiniCPM5-1B 'What is the capital of South Korea?' --chat --max-tokens 16
```

If `WORKERS` is omitted, the runtime counts CPUs in its affinity mask and starts one compute worker per visible CPU:

```bash
./aurora-llm model.ali
```

For convenient text I/O while keeping tokenization outside the assembly runtime:

```bash
python3 tools/client.py /models/MyLlama 'Explain NUMA in one paragraph.'
```

## Minimal KVM appliance

```bash
./tools/make_initramfs.sh model.ali aurora-llm.cpio.gz
```

See [`docs/KVM.md`](docs/KVM.md). When executed as `/init`, the runtime defaults to `/model.ali`, auto-detects the CPU affinity count, and listens on port 8080.

## Current boundary

This is a **working inference core**, not yet a production vLLM replacement. v0.1 is deliberately decode-oriented: Llama/Mistral dense models, Q8 weights + FP32 activations, one request at a time, greedy decode, token-ID API, AVX2/FMA. Continuous batching, Q4, AVX-512/VNNI, AMX prefill, speculative decoding, raw-text tokenizer support, multi-NUMA replicas and production HTTP parsing are the next performance layer.

That distinction matters: the present artifact is suitable for validating the assembly/KVM architecture and benchmarking its core kernels, but it should not be described as faster than vLLM until those features are implemented and measured on the target CPU.
