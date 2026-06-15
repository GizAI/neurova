# QwenBurst

QwenBurst is a narrow CUDA runtime for Qwen3.6-27B text inference on a single
16GB RTX 4080/4090-class GPU. The engine targets batch-1 chat with flexible
groupwise low-bit weights, Qwen Gated DeltaNet kernels, ring-KV attention, and
streaming decode state.

This is not a general vLLM replacement. The current canonical contract is:

- HF source model: Qwen3.6-27B / Qwen3.5-style text model.
- Converted model format: `qwenburst-lowbit-v4`.
- Quantized tensor kind: `lowbit_symmetric_groupwise`.
- Supported weight bits: 2 through 8, selected by checkpoint metadata.
- Runtime entrypoints: `qwenburst-quantize`, `qwenburst-audit`,
  `qwenburst-chat`, `qwenburst-doctor`.
- CUDA extension symbols: `lowbit_gemv`, `lowbit_row_dequant`, RMSNorm, GDN
  recurrence, attention decode, and sampling helpers.

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

The default `--weight-device auto` keeps q3 checkpoints GPU-resident and keeps
larger checkpoints on the safer CPU staging path.

```bash
qwenburst-chat \
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb4 \
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
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb3 \
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
  --hf-model /home/user/models/Qwen3.6-27B \
  --qb-model /home/user/models/Qwen3.6-27B-qb3 \
  --host 0.0.0.0 \
  --port 8008 \
  --recent-window 256
```

Smoke:

```bash
curl -sS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwenburst-qwen3.6-27b-q3","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16,"temperature":0}'
```

SSE streaming is supported with `"stream": true`.

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

Current dmc8 result for q3 5120x5120 GEMV:

```text
rows_per_cta=4  : ~57.8 us
rows_per_cta=8  : ~56.8 us
rows_per_cta=16 : ~119.0 us
```

The current champion is `8`. This is still not enough for 100 tok/s; the next
major bottleneck is the scalar low-bit MLP projection path, not GDN recurrence.

## DFlash Speculative Path

Do not wire guessed native-MTP code into qwenburst, and do not replace the
qwenburst target runtime with vLLM/SGLang. DFlash must be a draft adapter:

```text
DFlash draft proposes tokens -> qwenburst target verifies -> qwenburst commits
```

Default pairing:

```text
target: /home/user/models/Qwen3.6-27B-qb3 through qwenburst
draft : z-lab/Qwen3.6-27B-DFlash
```

On a 16GB GPU the draft path must also have an explicit memory budget. The
target model is already the q3/q4 low-bit qwenburst checkpoint, so a DFlash
adapter may not silently load a large fp16 side model. The acceptable forms are:

```text
1. DFlash weights are already compact enough to stay resident beside q3 target.
2. DFlash weights are converted to qwenburst low-bit draft tensors.
3. DFlash reuses target hidden/state and only stores small draft heads.
```

If the DFlash safetensors structure requires a full fp16 draft network, it must
be quantized before it is enabled by default.

The server already exposes the runtime option placeholder:

```bash
qwenburst-server --speculative-backend none
```

DFlash artifacts use a separate conversion command:

```bash
python -m qwenburst.dflash inspect /path/to/Qwen3.6-27B-DFlash
python -m qwenburst.dflash convert /path/to/Qwen3.6-27B-DFlash /home/user/models/Qwen3.6-27B-DFlash-qb3 --bits 3
```

Then the server can load the converted draft adapter without launching another
runtime:

```bash
qwenburst-server \
  --speculative-backend dflash \
  --dflash-draft-dir /home/user/models/Qwen3.6-27B-DFlash-qb3
```

The native qwenburst DFlash proposal executor is still the remaining work item;
the option is intentionally wired to the adapter boundary instead of vLLM.

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

The qwenburst kernel path is currently limited by low-bit MLP projection.
Reaching 100 emitted tok/s requires either a DFlash draft adapter verified by
qwenburst or a stronger dequant+MMA projection kernel.
