# QwenBurst vLLM-Class Serving TODO

This is the single backlog for vLLM/TensorRT-class QwenBurst serving. Do not
scatter performance TODOs into separate notes. Promote only changes that keep
output/state parity and show measured benefit.

## P0: Correctness Gates

- Status: in progress
- SSOT: `qwenburst.correctness`, CPU tests, dmc8 short real-model benches
- Required:
  - HF/vLLM/QwenBurst first-token top-k parity where a BF16 reference is feasible
  - batch=1 vs batch=N output parity
  - one-shot prefill vs chunked/block prefill state parity
  - speculative on/off greedy identity
  - long-context exact recall inside the active KV window
  - arena slot release/reset parity

## P1: Continuous Batching Scheduler

- Status: implemented baseline, keep optimizing
- SSOT: `qwenburst.core.scheduler.ContinuousBatchScheduler`
- Done:
  - decode rows scheduled before prefill rows
  - chunked prefill budget
  - reusable `DecodeInputBuffers`
  - stream and non-stream greedy server paths use `BatchGenerationWorker`
- Remaining:
  - cancellation-aware scheduler post-update
  - fairness/backpressure tuning
  - production metrics for batch size, TTFT, tok/s, queue wait

## P2: Slot-Indexed State Arena

- Status: implemented baseline
- SSOT: `qwenburst.state.DecodeStateArena`, `qwenburst.core.state_store.BatchStateStore`
- Done:
  - Qwen batch runner allocates `[slot, ...]` GDN, conv, K, V state buffers
  - request `state_index` maps to arena slot-backed `DecodeState` view
  - release resets/recycles slots
- Remaining:
  - move more state counters (`pos`, `kv_len`) to device tensors
  - expose arena slot ids to CUDA kernels directly
  - COW/fork support for prefix cache and speculative branches

## P3: Paged KV / Block Table

- Status: scaffolded, not hot-path complete
- SSOT: `qwenburst.core.block_table.KVBlockTable`
- Done:
  - logical block allocation/release
  - `block_tables` and `slot_mapping` tensors attached to `DecodeBatchPlan`
- Remaining:
  - attention kernels must consume `block_tables + slot_mapping`
  - remove ring `torch.cat` materialization from hot path
  - add page refcount/copy-on-write for prefix cache
  - add KV page zeroing/admission behavior like vLLM

## P4: Multi-User Batched Model Forward

- Status: partially implemented
- SSOT: `RuntimeEngine.forward_batch(plan, states)`, `QwenBurstModel.forward_batch`
- Done:
  - shared `DecodeBatchPlan`
  - model-native single-token cross-request decode batch route
  - embedding/projection/MLP/head can run on `[B, D]`
  - GDN/attention state commit remains request-correct through arena views
- Remaining:
  - remove row-wise fallback for all common decode cases
  - batch prefill/query_len > 1 through one model forward
  - make speculative verify-N use the same batch model path
  - return hidden/logit tensors in vLLM-compatible flattened form

## P5: Batch-State CUDA Kernels

- Status: next kernel target
- SSOT: CUDA extension under `qwenburst/csrc`
- Required:
  - GDN recurrent kernel reads/writes `[slot, layer, ...]` by `state_indices`
  - GDN conv update reads/writes arena conv buffers by `state_indices`
  - attention decode reads paged/block KV by `block_tables/slot_mapping`
  - no Python loop over requests in hot decode
  - exact parity against current eager arena path

## P6: Marlin/CUTLASS Batch Hot Path

- Status: partial
- SSOT: `LowBitMarlinTensor.gemm`, `linear_any`
- Done:
  - Marlin Q4 path and output/workspace cache
  - `[B, D]` route for projections/head when supported
- Remaining:
  - remove direct batch fallback limitations after parity
  - preallocate all layer workspaces by graph bucket
  - tune MLP gate/up/down scheduling
  - validate tensor core utilization with Nsight Compute

## P7: CUDA Graph Buckets

- Status: scaffolded, not captured
- SSOT: `qwenburst.core.cuda_graph.CudaGraphBucketPlanner`
- Required:
  - device-side `pos`, `kv_len`, slot ids, and output token buffers
  - static input/state/logit buffers per `(B, query_len, spec_tokens)` bucket
  - graph capture for decode1 first
  - graph capture for verify-N after speculative verifier is batch-safe
  - eager vs graph token/state parity

## P8: Native MTP / NEXTN Speculative Decode

- Status: implemented gated path, not default-champion
- SSOT: `qwenburst.qwen_mtp`, `qwenburst.speculative_verifier`
- Current measurement:
  - dmc8 short suite: identity true, average speedup about 1.09x for MTP1
- Remaining:
  - connect proposer to batch verifier, not token-by-token target loop
  - sweep `num_spec=1..4` after verify-N is fused
  - automatic enable only when measured speed-positive and identity-preserving
  - full probabilistic rejection sampling for temperature/top-p

## P9: lm_head + Sampling Fusion

- Status: not complete
- Required:
  - fused greedy argmax/top-k for 248K vocab
  - avoid materializing full logits when only argmax is needed
  - candidate-limited projection for verifier where mathematically valid
  - GPU-only EOS/sampling postprocess

## P10: Prefix / State Cache

- Status: planned
- Required:
  - system prompt snapshot cache
  - GDN + conv + KV-page cache entry
  - page refcount/COW fork
  - longest-prefix match or radix cache
  - OpenWebUI system prompt TTFT benchmark

## P11: KV Quantization / Memory Scaling

- Status: planned
- Required:
  - q8/FP8 KV cache experiments
  - attention parity and quality checks
  - dynamic recent-window admission by free VRAM
  - only enable if quality-preserving

## P12: C++/CUDA Decode Loop

- Status: planned
- Required:
  - move decode step, sampling, EOS check, and post-update out of Python
  - Python only enqueues/dequeues requests
  - async output transfer and detokenization overlap
  - process-level shutdown/cancel cleanup

## P13: Observability and Bench Discipline

- Status: in progress
- Required:
  - Nsight Systems: launch gaps, CPU sync, stream idle
  - Nsight Compute: Marlin occupancy, tensor core utilization, bandwidth, GDN occupancy
  - standard dmc8 benchmark suite:
    - target-only
    - batched decode
    - MTP/NEXTN
    - long prompt prefill
    - multi-user throughput
  - keep only measured speed-positive defaults

## P14: Server Production Semantics

- Status: in progress
- Required:
  - request cancellation releases arena slots and KV blocks
  - OOM admission control before allocation
  - max active slots and KV pages surfaced in health
  - model unload shuts down workers and frees arena
  - multi-model LRU remains safe under active requests

## Default Policy

- Defaults may only change when all three are true:
  - output identity or approved sampling equivalence
  - state parity under continuation
  - measured speed or capacity improvement on dmc8

## Current Next Action

Implement P5: batch-state CUDA kernels for GDN/attention consuming arena
buffers and `state_indices`, then wire them into P4/P7/P8.
