# Mamba3 Max-Performance Plan

## Current Reality

The current `mamba3_kr` path is a runnable Mamba-3 MIMO prototype, not a capable LLM yet.

Verified on `ml-dmc8`:
- GPU: RTX 4080 16GB
- Mamba-3 source build: works with `torch+cu126`, CUDA toolkit 12.0, `gcc/g++-12`
- MIMO forward: works through TileLang
- MIMO-R4 tiny backward: works with `headdim=32`; larger `headdim=64` variants exceed RTX 4080 dynamic shared-memory limits in the current TileLang backward kernel
- fast recurrent decode: works after patching Mamba-3 step `tile_D` from hard-coded 64 to `self.headdim`
- English bootstrap training: 500 steps completed; it can form short English sentences, but it is not a real pretrained LLM
- warm CUDA-graph decode benchmark: about 344 new tok/s on the current tiny checkpoint after warmup

The remaining quality limit is expected: a randomly initialized tiny model trained for a few hundred steps on a tiny bootstrap corpus is not an English LLM.

## What The Paper Actually Implies

To maximize Mamba-3's advantages, the system must use:

- Mamba-3 MIMO-R4, not SISO, for the target quality path.
- Recurrent step decode, not full-window re-forwarding per token.
- Persistent server process so TileLang/CuTe kernels compile once and stay warm.
- Large enough batch during decode to fill GPU memory bandwidth and tensor cores.
- MIMO rank 4 for the target architecture; use `mimo-r4-tiny` for RTX 4080 kernel validation.
- `chunk_size = 64 / mimo_rank` as the starting point, reduced if shared memory overflows.
- Llama-3.1 tokenizer as the primary tokenizer, not byte-only tokenization.
- Real pretraining scale. The paper's language results use 100B FineWeb-Edu tokens with the Llama-3.1 tokenizer at 2K context.
- Parameter-matched MIMO-R4 MLP dims from Appendix C: 180M=1264, 440M=1792, 880M=2800, 1.5B=3824.
- Paper-scale Mamba models keep `d_state=128` and `headdim=64`; RTX 4080 tiny-kernel validation may temporarily reduce `d_state/headdim`.

## RTX 4080 Target Architecture

Phase 1, stable MIMO-R4:

```text
Mamba3-MIMO-English-180M
  d_model: 768
  layers: 12
  d_state: 128
  headdim: 64
  mimo_rank: 4
  d_intermediate: 1264
  chunk_size: 16
  tokenizer: Llama-3.1
  context train: 2K first
  inference: recurrent step decode
```

Phase 2, stronger MIMO:

```text
Mamba3-MIMO-English-440M
  d_model: 1024
  layers: 16
  d_state: 128
  headdim: 64
  mimo_rank: 4
  d_intermediate: 1792
  chunk_size: 16
```

Phase 3, stronger MIMO:

```text
Mamba3-MIMO-English-880M
  d_model: 1536
  layers: 20
  d_state: 128
  headdim: 64
  mimo_rank: 4
  d_intermediate: 2800
  chunk_size: 16
  activation checkpointing for training
  bf16 weights/activations
```

Phase 4, product-size:

```text
Mamba3-MIMO-English-1.3B-1.5B
  d_model: 2048
  layers: 24
  d_state: 128
  headdim: 64
  mimo_rank: 4
  d_intermediate: 3824
  4bit/8bit inference after training
```

## Performance Rules

Do:
- Use one persistent Python/server process.
- Warm up prefill and step kernels before measuring.
- Measure with `bench-decode --cuda-graph`; do not use cold one-shot CLI timings as throughput.
- Keep recurrent state in GPU memory for active sessions.
- Store recurrent state to CPU/disk only at session boundaries.
- Batch decode requests where possible.
- Compile only RTX 4080 arch: `TORCH_CUDA_ARCH_LIST=8.9`.
- Use `CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12` on `ml-dmc8`.

Do not:
- Launch a new Python process per prompt.
- Use full-window forward for every generated token.
- Judge quality from byte-tokenized tiny training.
- Increase MIMO `d_state`, `headdim`, and `chunk_size` blindly; shared memory fails first.
- Re-add punctuation/digit token suppression; it was only a failed workaround for NaN logits.
- Claim H100 paper throughput on RTX 4080 without measuring the same batch/decode setup.

## Immediate Engineering Tasks

1. Use `--tokenizer llama31` as the default real BPE path.
2. Keep `mimo-r4-paper-180m`, `mimo-r4-440m`, `mimo-r4-880m`, and `mimo-r4-1.5b` presets contract-checked.
3. Use the persistent MIMO-R4 decode server or `bench-decode --cuda-graph` with warmup excluded from metrics.
4. Build the 100B+ token English corpus pipeline outside the 16GB box.
5. Train with the 2K -> 8K -> 32K -> 128K long-state curriculum in `configs/mamba3_english_curriculum.json`.
6. Use verifier-only RLVR; no external LLM teacher or judge.
7. Keep Korean specialization as a later continued-pretraining phase.
