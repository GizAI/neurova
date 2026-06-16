# Mamba-3 Optimization Research Notes

Date: 2026-06-13

This document tracks optimization techniques that are relevant to Neurova's pure Mamba-3 path. The rule is practical: an optimization is useful only if it improves correctness, speed, memory, or quality under measurable gates.

## Current Local Evidence

### Dense Scaling Limit On RTX 4080 16GB

`neuromamba/scripts/mamba3_find_16gb_train_target.sh` showed:

- `mimo-r4-tiny`: backward passes.
- `mimo-r4-16gb-120m`: forward and recurrent decode pass, backward fails with TileLang dynamic shared memory `123552`.
- `mimo-r4-16gb-180m`: forward and recurrent decode pass, backward fails with TileLang dynamic shared memory `123552`.
- `mimo-r4-paper-180m`: forward and recurrent decode pass, backward fails with TileLang dynamic shared memory `223904`.

Conclusion: dense 120M+ Mamba-3 MIMO-R4 is not currently trainable on this RTX 4080 through the official TileLang backward path. Larger dense models can remain inference/probe targets until the backward kernel or hardware changes.

### Sparse MoE Scaling

Implemented `SparseGatedMLP` as top-1 SwiGLU experts while keeping Mamba-3 recurrence dense.

Observed:

- `mimo-r4-moe-260m`: about 284M estimated total parameters.
- Backward passes on RTX 4080 with about 2.7GB peak VRAM in the canary.
- 30 base steps + 30 SFT steps reduce loss, but output collapses into repeated common tokens.
- CUDA graph decode is disabled for sparse MoE because Python/dynamic routing breaks capture.

Conclusion: MoE is the best local path for larger total parameters, but it needs router stabilization, better SFT, and no-graph or vectorized graph-safe routing before promotion.

## Evidence From Papers And Docs

- Mamba-3: use MIMO-R4, exponential-trapezoidal recurrence, complex state, B/C bias, and BC/QK Norm as the architectural target. Source: https://arxiv.org/abs/2603.15569
- Mamba-3 latency claims assume official recurrent/cache kernels, not full-forward one-token-at-a-time fallback. The Princeton/Together writeups report fast SISO and strong MIMO performance at 1.5B with batch-size benchmark settings, so local speed claims must be separated into `full_forward_fallback`, `mamba3_step_fn`, and CUDA-graph variants. Sources: https://arxiv.org/html/2603.15569v1 and https://www.together.ai/blog/mamba-3
- NVIDIA's controlled 8B study reports pure Mamba/Mamba-2 weaknesses on copying, in-context learning, Phonebook lookup, and long-context reasoning, while a small-attention hybrid outperformed the Transformer average in their evaluated setting. This supports the recall-hybrid path, not blind pure-SSM scaling. Source: https://research.nvidia.com/publication/2024-06_empirical-study-mamba-based-language-models
- Hymba supports the same conclusion from a small-LM architecture angle: attention heads provide high-resolution recall while SSM heads summarize context efficiently, and learnable meta tokens are a plausible memory aid. Source: https://arxiv.org/html/2411.13676v1
- MoE-Mamba: combining Mamba with MoE can reach comparable performance in fewer training steps. Source: https://arxiv.org/abs/2401.04081
- BlackMamba: Mamba + MoE can increase capacity while preserving SSM generation advantages; published 340M/1.5B and 630M/2.8B-style sparse models. Source: https://arxiv.org/abs/2402.01771
- LongMamba: long-context degradation can be mitigated by identifying critical tokens and filtering unimportant tokens from global-channel memory. Source: https://arxiv.org/abs/2504.16053
- HAX/CDSA: SSMs alone have limits on multi-query joint recall; context-dependent sparse attention addresses that expressiveness gap. Source: https://arxiv.org/html/2507.00449v3 and https://github.com/DeepGraphLearning/HAX
- RAMba/HSA: hierarchical sparse attention combined with Mamba improves random long-context access while keeping memory nearly constant. Source: https://arxiv.org/abs/2504.16795
- PackMamba: sequence packing for Mamba must avoid state contamination across packed documents; the paper reports large throughput gains from packing variable-length sequences while preventing cross-sequence state leakage. Source: https://arxiv.org/html/2408.03865v1
- FLA is a fast-moving alternative kernel ecosystem with Mamba3, GDN-2, MoBA, and fused modules. It is useful as a comparison/probe path, but official `state-spaces/mamba` remains the default until parity and training gates pass locally. Source: https://github.com/fla-org/flash-linear-attention
- PyTorch activation checkpointing trades compute for memory, but local TileLang Mamba3 autograd currently conflicts with block-level checkpointing. For Mamba-3, official kernel-level backward recomputation is the priority path. Source: https://docs.pytorch.org/docs/stable/checkpoint.html
- CUDA graphs require static long-lived buffers and graph-safe operations; dynamic MoE routing is not graph-safe yet. Source: https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- bitsandbytes 8-bit optimizers are a practical optimizer-state memory reduction path, but small/unstable tensors may need 32-bit handling. Source: https://huggingface.co/docs/bitsandbytes/optimizers
- Quamba and Quamba2 show SSM-specific quantization paths for Mamba/Mamba2. These are post-quality-gate inference optimizations, not a substitute for training quality. Sources: https://openreview.net/forum?id=mnna9LUg7P and https://openreview.net/forum?id=Zm0Kper4yx
- `torch.compile` should be probe-only around custom Mamba kernels until graph breaks are measured. Source: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.fullgraph_true.html

## Optimization Policy

### Training

Use by default:

- bf16 training.
- gradient accumulation.
- official Mamba-3 kernel-level backward recomputation.
- strict quality gates before promotion.
- governed data split into base pretraining and instruction SFT.

Use behind flags:

- `--optimizer adamw8bit` after bitsandbytes availability and stability checks.
- `--deepspeed-config neuromamba/configs/deepspeed_zero2_cpu_offload.json` only after Mamba-3 TileLang/DeepSpeed compatibility is fixed.
- `--deepspeed-config neuromamba/configs/deepspeed_zero3_param_offload.json` only after ZeRO-2 compatibility is fixed and parameter memory is the bottleneck.
- sparse MoE only as a research candidate until router stability and quality improve.
- hybrid GQA attention every 4-6 Mamba blocks as a recall path candidate.

Do not use as shortcuts:

- quantization before quality gate,
- raw-web continuation on chat checkpoints,
- promotion based on loss alone,
- CUDA graph for dynamic MoE routing.
- generic block-level PyTorch activation checkpointing around Mamba-3 TileLang kernels.
- Wan2GP-style layer streaming for the training body unless all other memory tiers fail; it preserves weights but is expected to hurt speed heavily.

### Inference

Use by default:

- recurrent step decode only after full-forward parity passes for that checkpoint and preset,
- CUDA graph only for dense graph-safe models after eager-vs-graph parity passes,
- warmup-excluded speed benchmark,
- token speed shown at response end,
- state save/restore metadata checks.

Current user-facing default:

- `./neurova.sh mamba3` uses `neuromamba/scripts/mamba3_safe_chat.py`, a model-only full-forward deterministic path, because the latest speaking checkpoint gives better answers there than the recurrent/CUDA-graph path.
- This is slower than the theoretical Mamba-3 fast path, but it avoids presenting broken fast-cache outputs as model quality.
- `neuromamba/scripts/mamba3_decode_tune.py` is the canonical quality/speed sweep for deciding when a checkpoint is ready to switch back to recurrent/CUDA-graph decode.

Needed next:

- CUDA graph vs eager parity test,
- state save/restore continuation test,
- stateful long-context speed gate,
- quantized inference export after quality promotion.

## Priority Checklist

P0:

- Run `./neurova.sh mamba3 tune` after every new speaking checkpoint and keep the best config evidence under `neuromamba/runs/.../decode_tune/`.
- Add recurrent state ABI tests: mode/tokenizer metadata, dtype/device migration, batch mismatch rejection.
- Add eager-vs-CUDA-graph decode parity gate for dense models.
- Keep `MAMBA3_KERNEL_BACKWARD_NOTES.md` as the source-level decision record for official backward recomputation.
- Install and validate `bitsandbytes` and `deepspeed` through `neuromamba/scripts/mamba3_install_efficiency_deps.sh`.
- Run `neuromamba/scripts/mamba3_train_stability_ladder.sh` to compare SISO, MIMO r2, MIMO r2+attention, MIMO r4, and MIMO r4+attention on the same no-teacher curriculum.
- Add MoE router z-loss and load-balancing loss.
- Add per-expert token count logs without CUDA graph CPU copies.
- Keep `mimo-r4-moe-260m` as trainable large-parameter research candidate, not default runtime.

P1:

- Implement graph-safe vectorized sparse routing or keep MoE no-graph.
- Add 8-bit optimizer stability policy.
- Validate ZeRO-2 CPU optimizer offload on a short canary before using ZeRO-3.
  - Current canary blocks in Mamba-3 TileLang backward with dynamic shared memory `152768`, so ZeRO is not yet a usable memory tier for this kernel path.
- Add held-out LM loss eval to promotion gate.
- Add long-state recall tests with distractors.
- Add state save/load latency and footprint metrics.
- Add LongMamba-style token filtering and HAX/RAMba-style sparse attention only after exact-recall curriculum metrics exist.

P2:

- Explore Quamba-style SSM-aware quantization after a checkpoint passes quality.
- Add `torch.compile fullgraph=True` probes for prefill and decode separately.
- Revisit dense 180M+ training only after TileLang shared-memory backward issue is solved.
