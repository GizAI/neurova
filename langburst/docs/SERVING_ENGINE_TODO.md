# LangBurst external serving engine-Class Serving TODO

This is the single backlog for production-class LangBurst serving. Do not
scatter performance TODOs into separate notes. Promote only changes that keep
output/state parity and show measured benefit.

## P0: Correctness Gates

- Status: in progress
- SSOT: `langburst.correctness`, CPU tests, dmc8 short real-model benches
- Required:
  - reference/LangBurst first-token top-k parity where a BF16 reference is feasible
  - batch=1 vs batch=N output parity
  - one-shot prefill vs chunked/block prefill state parity
  - speculative on/off greedy identity
  - long-context exact recall inside the active KV window
  - arena slot release/reset parity

## P1: Continuous Batching Scheduler

- Status: implemented baseline, measured dmc8 serving path
- SSOT: `langburst.engines.native.ContinuousBatchScheduler`
- Done:
  - decode rows scheduled before prefill rows
  - chunked prefill budget
  - reusable `DecodeInputBuffers`
  - stream and non-stream greedy server paths use `BatchGenerationWorker`
  - cancellation-aware worker cleanup releases runner state before handle done
  - request metrics: queue wait, TTFT, E2E latency, decode wall time, ITL,
    aggregate output tok/s, aggregate decode tok/s
  - fixed ready-batch admission split: already queued requests are admitted up
    to `max_num_requests` before the first model step
- Remaining:
  - fairness/backpressure tuning
  - production metrics export from HTTP health/metrics endpoints

## P2: Slot-Indexed State Arena

- Status: implemented baseline
- SSOT: `langburst.adapters.qwen36_impl.state.DecodeStateArena`, `langburst.core.state_store.BatchStateStore`
- Done:
  - Qwen batch runner allocates `[slot, ...]` GDN, conv, K, V state buffers
  - request `state_index` maps to arena slot-backed `DecodeState` view
  - release resets/recycles slots
- Remaining:
  - move more state counters (`pos`, `kv_len`) to device tensors
  - expose arena slot ids to CUDA kernels directly
  - COW/fork support for prefix cache and speculative branches

## P3: Paged KV / Block Table

- Status: decode hot path implemented, default-on for multi-user serving
- SSOT: `langburst.engines.native.KVBlockTable`
- Done:
  - logical block allocation/release
  - `block_tables` and `slot_mapping` tensors attached to `DecodeBatchPlan`
  - decode attention consumes `block_tables + slot_mapping`
  - chunked/block prefill publishes canonical KV into paged KV before decode
  - arena/KV blocks release to zero active usage after worker completion
  - dmc8 measured tradeoff, prompt=256/max_new=128:
    - batch=1: paged 28.07 decode tok/s vs non-paged 29.57
    - batch=4: paged 102.32 decode tok/s vs non-paged 78.41
- Remaining:
  - fix paged prefill parity before enabling query_len > 1 paged prefill
  - remove ring `torch.cat` materialization from long-context non-paged fallback
  - add page refcount/copy-on-write for prefix cache
  - add KV page zeroing/admission behavior like external serving engine

## P4: Multi-User Batched Model Forward

- Status: partially implemented
- SSOT: `RuntimeEngine.forward_batch(plan, states)`, `Qwen36Model.forward_batch`
- Done:
  - shared `DecodeBatchPlan`
  - model-native single-token cross-request decode batch route
  - embedding/projection/MLP/head can run on `[B, D]`
  - GDN/attention state commit remains request-correct through arena views
  - dmc8 batch=4 aggregate decode throughput reaches about 102 tok/s with
    paged KV and batch-state kernels
  - canonical/no-paged timestep batch prefill is output-identical to the
    existing path and improves dmc8 batch=4 prompt=256 TTFT from 23.06s to
    12.59s; aggregate output tok/s improves from 5.19 to 9.02
- Remaining:
  - remove row-wise fallback for all common decode cases
  - paged KV batch prefill/query_len > 1 with continuation parity. A direct
    paged-attention timestep prefill improved TTFT to 9.44s but changed
    deterministic continuation, so it is guarded off for paged plans.
  - make speculative verify-N use the same batch model path
  - return hidden/logit tensors in batch-runtime-compatible flattened form
  - investigate batch=6 regression: runs but drops to about 40 decode tok/s;
    batch=8 OOMs on 16GB with current arena + paged KV allocation

## P5: Batch-State CUDA Kernels

- Status: default-on for deterministic decode batches
- SSOT: CUDA extension under `langburst/csrc`
- Done:
  - GDN recurrent kernel reads/writes `[slot, layer, ...]` by `state_indices`
  - GDN conv update reads/writes arena conv buffers by `state_indices`
  - attention decode reads paged/block KV by `block_tables/slot_mapping`
  - no Python loop over requests in hot decode
  - fixed real Qwen shape GDN batch output drift at `kv_heads=16`, `v_heads=48`
  - dmc8 deterministic batch=2 runner: batch-state ON token-identical to eager path
  - dmc8 speed: 17.46 tok/s -> 20.35 tok/s on the short batch=2 parity prompt
- Remaining:
  - broaden parity prompts and longer generation lengths
  - keep monitoring paged KV + batch-state together under long-context tests

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
- SSOT: `langburst.core.cuda_graph.CudaGraphBucketPlanner`
- Required:
  - device-side `pos`, `kv_len`, slot ids, and output token buffers
  - static input/state/logit buffers per `(B, query_len, spec_tokens)` bucket
  - graph capture for decode1 first
  - graph capture for verify-N after speculative verifier is batch-safe
  - eager vs graph token/state parity

## P8: Native MTP / NEXTN Speculative Decode

- Status: auto-gated adaptive path with model-side batch verifier
- SSOT: `RuntimeEngine.generate_decode_result`, `RuntimeEngine.verify_nextn_tokens`, `langburst.adapters.qwen36_mtp`, `langburst.research.qwen_nextn_bench_verifier`, `Qwen36Model.forward_verify_batch`
- Current measurement:
  - dmc8 short suite: identity true, average speedup about 1.09x for MTP1
  - dmc8 decode-only smoke, context=256, draft=4/6/8/10:
    - target-only: about 40.5 tok/s
    - draft=4: 17.4 tok/s, 0.43x, identity true
    - draft=6: 13.6 tok/s, 0.34x, identity true
    - draft=8: 11.6 tok/s, 0.29x, identity true
    - draft=10: 10.3 tok/s, 0.25x, identity true
    - decision: do not promote larger draft counts
- Remaining:
  - fused CUDA batch verifier single hot path. Current flow has the right
    contract but can still fall back through `forward_verify_batch`,
    `forward_verify_block`, `forward_block`, or sequential target verification.
    It is not yet one production-class fused verifier path for all common cases.
  - keep `bench-auto-nextn` as the only NEXTN auto-adopt gate
  - run long-context sweeps after prefill is optimized; context=65K did not complete under a 120s smoke timeout
  - wire true batch=2 NEXTN generation through the multi-request batch verifier before benchmarking it
  - full probabilistic rejection sampling for temperature/top-p

## P9: lm_head + Sampling Fusion

- Status: partial
- Done:
  - worker non-spec greedy rows use batched GPU argmax-many instead of row-wise
    CPU `sample_next()`
- Current finding:
  - this removed an obvious CPU-sync pattern but did not materially improve the
    dmc8 worker benchmark; remaining gap vs direct target-only decode is in the
    serving/state/paged forward path and full-logits materialization
- Required:
  - fused greedy argmax/top-k for 248K vocab
  - avoid materializing full logits when only argmax is needed
  - candidate-limited projection for verifier where mathematically valid
  - GPU-only EOS/sampling postprocess

## P10: Prefix / State Cache

- Status: next TTFT target
- Required:
  - system prompt snapshot cache
  - GDN + conv + KV-page cache entry
  - page refcount/COW fork
  - longest-prefix match or radix cache
  - OpenWebUI system prompt TTFT benchmark
  - first implementation may be exact full-prefix cache for repeated prompts;
    promote only if it preserves output identity and reduces paged TTFT

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
    - external engine external baseline on the same prompt/context/output-token
      matrix when the target model format is available
  - keep only measured speed-positive defaults

## P14: Server Production Semantics

- Status: in progress
- Required:
  - OOM admission control before allocation
  - max active slots and KV pages surfaced in health
  - model unload shuts down workers and frees arena
  - multi-model LRU remains safe under active requests
- Done:
  - request cancellation releases arena slots and KV blocks through `BatchGenerationWorker`

## P15: Adapter Portability / Multi-Model Plugin Surface

- Status: in progress
- Goal: keep Qwen3.6 as the champion adapter without letting Qwen-specific
  research/benchmark code leak into the generic runtime surface.
- Already done:
  - `AdapterDescriptor`, `ModelAdapter`, `AdapterRegistry`
  - entry-point adapter discovery via `langburst.adapters`
  - registry-based `add_adapter_arg()` for generic CLI adapter selection
  - `RuntimeEngine` delegates config/tokenizer/model/state/chat template to the
    selected adapter
  - `ModelResourceSpec.adapter_id` supports multi-model manager specs
  - model path CLI arguments are now centralized in `add_model_path_args()` and
    can be supplied through `LANGBURST_HF_MODEL` / `LANGBURST_QB_MODEL`
- Done:
  - added a reusable minimal adapter conformance helper
  - folded infinite streaming / episodic memory / TTT sidecar into the shared
    `RuntimeFeatures` -> `RuntimeCapabilities` -> `RuntimePlan` resolver
  - consolidated env/autotune/default execution policy in
    `RuntimePolicyResolver`
  - kept Qwen NEXTN/MTP sweep tools under adapter/research scope
  - replaced Qwen-specific wording in generic CLI descriptions where it is not
    adapter-specific
  - make quantize/checkpoint contracts explicit per adapter instead of assuming
    Qwen low-bit naming

## P16: External Baseline / EXL3 Comparison

- Status: planned
- Goal: do not optimize in isolation. Keep LangBurst measurements comparable
  to external-engine engines on the same hardware, prompt shape, context length,
  output length, and cache mode.
- Required:
  - single-user decode-only baseline
  - serving end-to-end baseline including TTFT
  - long-context prefill baseline at 1K/4K/8K and, when feasible, 65K+
  - multi-user batch profile: short context batch 1/2/4 and long context batch
    1/2
  - cache-mode or KV precision comparison where the baseline exposes it
  - text-only loader path verification so vision/preprocessor code is not
    accidentally initialized
- Current decision:
  - LangBurst internal champion remains batch=4 paged KV for multi-user decode
    on 16GB.
  - External baselines are comparison gates, not a reason to reintroduce
    duplicate serving code inside LangBurst.

## Default Policy

- Defaults may only change when all three are true:
  - output identity or approved sampling equivalence
  - state parity under continuation
  - measured speed or capacity improvement on dmc8

## Current Next Action

Highest ROI order:

1. P4 batch prefill/query_len > 1 through one model forward. This attacks TTFT,
   currently the largest end-to-end serving bottleneck.
2. P10 prefix/state cache for repeated system/chat prefixes. This can skip
   prefill work outright and is likely the fastest TTFT win for OpenWebUI-style
   repeated prompts.
3. P9 avoid full logits materialization for greedy decode where possible.
4. P15 adapter conformance and policy resolver cleanup, so Gemma/Llama-style
   adapters can be added without touching every CLI/bench path.
5. P8 true multi-request NEXTN only after P4 verifier batches are fast enough
   to make speculation speed-positive.
