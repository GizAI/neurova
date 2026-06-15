# QwenBurst

QwenBurst is a low-bit local inference runtime for single-user batch-1 chat on
16GB RTX 4080/4090-class GPUs. The current champion adapter targets
Qwen3.6-27B text inference with flexible groupwise low-bit weights, Qwen Gated
DeltaNet kernels, ring-KV attention, and streaming decode state.

This is not a general vLLM replacement. The runtime is now split into a common
`RuntimeEngine` plus model adapters so future Gemma-style targets do not fork
the server/decode loop. The current canonical adapter contract is:

- Adapter: `qwen36`.
- HF source model: Qwen3.6-27B / Qwen3.5-style text model.
- Champion converted model: Q4 Marlin fused projection checkpoint.
- Quantized tensor kinds: `lowbit_marlin_groupwise`,
  `lowbit_symmetric_groupwise`, and small `fp16_raw` gate projections.
- Supported rowwise weight bits: 2 through 8, selected by checkpoint metadata.
- Runtime entrypoints: `qwenburst-quantize`, `qwenburst-audit`,
  `qwenburst-chat`, `qwenburst-server`, `qwenburst-doctor`,
  `qwenburst-speculative`, `qwenburst-graph-audit`, `qwenburst-profile`.
- CUDA extension symbols: Marlin W4A16 GEMM, rowwise low-bit fallback,
  RMSNorm, GDN recurrence, attention decode, and sampling helpers.

The adapter split is documented in `docs/ADAPTER_ARCHITECTURE.md`.
Runtime feature profiles are documented in `docs/RUNTIME_FEATURES.md`.

## Environment

On `ml-dmc8` the standard environment is:

```bash
cd /home/user/workspace/neurova/qwenburst
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwenburst16g-cu130
```

Build and validate CUDA:

```bash
./scripts/cuda_compile_and_test.sh
```

CPU-only tools can be installed with:

```bash
QWENBURST_SKIP_CUDA_EXT=1 python -m pip install -e .
```

## Convert

Convert a checkpoint with the desired bit width:

```bash
qwenburst-quantize /home/user/models/Qwen3.6-27B /home/user/models/Qwen3.6-27B-qb4 --bits 4 --group-size 128
qwenburst-audit /home/user/models/Qwen3.6-27B-qb4 --hf-model /home/user/models/Qwen3.6-27B
```

For a smaller 16GB target:

```bash
qwenburst-quantize /home/user/models/Qwen3.6-27B /home/user/models/Qwen3.6-27B-qb3 --bits 3 --group-size 128
qwenburst-audit /home/user/models/Qwen3.6-27B-qb3 --hf-model /home/user/models/Qwen3.6-27B
```

## Chat

The default `--weight-device auto` keeps Marlin checkpoints GPU-resident. CPU
offload is only a fallback for checkpoints that cannot fit in VRAM.

```bash
qwenburst-chat \
  --adapter qwen36 \
  --runtime-profile stateful \
  --block-prefill on \
  --prefill-chunk-size 64 \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --prompt "안녕. 너는 누구야?" \
  --recent-window 8192 \
  --max-new-tokens 96 \
  --temperature 0 \
  --stream \
  --stats
```

Force the GPU-resident path:

```bash
qwenburst-chat \
  --adapter qwen36 \
  --runtime-profile original \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --weight-device cuda \
  --prompt "Say hello." \
  --max-new-tokens 32 \
  --temperature 0 \
  --stats
```

## OpenAI-Compatible Server

Run once and keep the model resident:

```bash
QWENBURST_LOWBIT_ROWS_PER_CTA=8 \
qwenburst-server \
  --adapter qwen36 \
  --runtime-profile stateful \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --host 0.0.0.0 \
  --port 8008 \
  --recent-window 256
```

Smoke:

```bash
curl -sS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwenburst-qwen3.6-27b-q4-marlin","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
```

SSE streaming is supported with `"stream": true`.

Inspect enabled runtime features:

```bash
curl http://127.0.0.1:8008/v1/qwenburst/features
```

The OpenAI-compatible chat endpoint also accepts per-request runtime overrides
such as `"runtime_profile": "original"`, `"kv_window_policy": "ring"`,
`"block_prefill": true`, or `"prefill_chunk_size": 64`.

## Runtime Tuning

The CUDA extension compiles low-bit GEMV variants once and selects at runtime:

```bash
QWENBURST_LOWBIT_ROWS_PER_CTA=4 qwenburst-chat ...
QWENBURST_LOWBIT_ROWS_PER_CTA=8 qwenburst-chat ...
QWENBURST_LOWBIT_ROWS_PER_CTA=16 qwenburst-chat ...
```

Benchmark without rebuilding:

```bash
python benchmarks/bench_kernels.py --bits 3 --rows-per-cta 8
```

CUDA Graph is gated by audit, not enabled by default:

```bash
qwenburst-graph-audit --static
```

The current blocker is architectural: decode state still uses Python `pos` /
`kv_len` counters and Python ring-KV logical views. Do not count CUDA Graph as a
speed feature until a real device-counter `GraphDecodeState` path lands.

Long prompt prefill uses chunked `forward_block` by default in every runtime
profile. This preserves target-model math while avoiding the old token-by-token
prefill loop. Disable it only for regression bisects:

```bash
qwenburst-chat --block-prefill off ...
```

Current dmc8 result for q3 5120x5120 GEMV:

```text
rows_per_cta=4  : ~57.8 us
rows_per_cta=8  : ~56.8 us
rows_per_cta=16 : ~119.0 us
```

The rowwise fallback champion is `8`. The current speed path is Q4 Marlin, so
rowwise GEMV tuning is only relevant for fallback tensors and embeddings.

## Current Champion

The current 16GB path is target-only Q4 Marlin with fused projections:

```bash
python -m qwenburst.quantize \
  /home/user/models/Qwen3.6-27B \
  /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --bits 4 \
  --layout marlin \
  --group-size 128 \
  --fuse-projections
```

Run it with all target weights GPU-resident:

```bash
python -m qwenburst.generate \
  --adapter qwen36 \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --device cuda \
  --recent-window 256 \
  --max-new-tokens 512 \
  --stats \
  --prompt "Write a concise technical note about quantized LLM inference."
```

Latest dmc8 result:

```text
128-token English: 29.63 tok/s
512-token English: 34.03 tok/s
```

Detailed per-change speed history is in `docs/PERFORMANCE_LOG.md`.
State/runtime feature coverage is in `docs/V05_FEATURE_TEST_MATRIX.md`.
DFlash-free speculative research is in `docs/SPECULATIVE_RESEARCH.md`.

Break down decode bottlenecks without changing the serving path:

```bash
qwenburst-profile \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --max-new-tokens 16
```

Compare feature profiles when the GPU is free:

```bash
qwenburst-bench-profiles \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4-marlin-fused \
  --profiles original,stateful,research \
  --max-new-tokens 128
```

## dmc8 One-Shot

```bash
MODEL_DIR=/home/user/models/Qwen3.6-27B \
QB_DIR=/home/user/models/Qwen3.6-27B-qb3 \
BITS=3 \
./scripts/dmc8_reconvert_and_chat.sh
```

## Validation

CPU validation:

```bash
QWENBURST_SKIP_CUDA_EXT=1 python -m pytest -q \
  tests/test_quant_lowbit_cpu.py \
  tests/test_gdn_reference_cpu.py \
  tests/test_v04_correctness_cpu.py \
  tests/test_v05_runtime_cpu.py \
  tests/test_state_streaming_cpu.py \
  tests/test_memory_ttt_cpu.py
```

CUDA validation:

```bash
python -m pytest -q \
  tests/test_v05_cuda_kernels.py \
  tests/test_lowbit_gemv_cuda.py \
  tests/test_sampling_cuda.py \
  tests/test_gdn_parity_cuda.py
```

## Current Speed Boundary

The qwenburst kernel path is currently dominated by target-model Marlin
projection work, especially MLP `gate_up` and `down` projections. Reaching 100
emitted tok/s requires native MTP/NEXTN with an exact target-compatible accept
contract, or a deeper fused target layer path that preserves logits.
