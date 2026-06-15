# QwenBurst v0.5 Feature Test Matrix

This matrix tracks advanced runtime features separately from Q4 Marlin decoding
speed. The goal is to prevent hidden state regressions while optimizing the
runtime.

## Current Test Command

```bash
QWENBURST_SKIP_CUDA_EXT=1 python -m pytest -q \
  qwenburst/tests/test_state_streaming_cpu.py \
  qwenburst/tests/test_memory_ttt_cpu.py \
  qwenburst/tests/test_quant_lowbit_cpu.py \
  qwenburst/tests/test_gdn_reference_cpu.py \
  qwenburst/tests/test_v04_correctness_cpu.py \
  qwenburst/tests/test_v05_runtime_cpu.py \
  qwenburst/tests/test_adapter_runtime_cpu.py
```

Current result:

```text
34 passed, 1 skipped
```

## Matrix

| Feature | Guarantee | Test coverage |
| --- | --- | --- |
| InfiniteStreamingRuntime | Chunked ingest must produce the same state as one-shot ingest. | `test_chunked_ingest_matches_one_shot_state` |
| DecodeState carry | Prior ingested tokens must change later logits/hidden state. | `test_decode_state_carry_changes_next_logits` |
| GDN conv buffer state | Causal conv short memory must survive chunk boundaries, fork, snapshot. | `test_chunked_ingest_matches_one_shot_state`, `test_decode_state_fork_decay_and_snapshot` |
| Ring KV recent window | KV must stay bounded to `recent_window`; 63/64/65 boundaries must rotate correctly. | `test_ring_kv_boundaries_63_64_65_and_snapshot_roundtrip` |
| State snapshot save/load | Warm-boot load must reproduce the same next result and state. | `test_state_only_warm_boot_matches_continue` |
| State fork / branch | Branches must not mutate parent or each other. | `test_state_fork_branch_and_parent_isolation` |
| State decay / reset | Decay must scale GDN/conv state; reset must clear state and counters. | `test_decay_reset_and_state_delta_apply` |
| Stateful multi-turn chat | A multi-turn state holder must reuse the same `DecodeState`; history must change later logits/state versus a fresh state. | `test_stateful_multi_turn_chat_uses_same_decode_state` |
| State-only warm boot | Snapshot load must produce equivalent continuation state without replaying the original ingest. | `test_state_only_warm_boot_matches_continue` |
| EpisodicMemory / State-RAG | Exact facts must be recoverable via top-k lexical fallback. | `test_episodic_memory_exact_fact_topk` |
| State delta record | State delta apply must match direct ingest for deterministic small states. | `test_decay_reset_and_state_delta_apply` |
| TTT sidecar memory | Sidecar updates must change sidecar read output. | `test_ttt_sidecar_update_read_and_state_dict` |
| TTT does not contaminate base state | TTT on/off must leave base DecodeState identical. | `test_ttt_sidecar_does_not_mutate_decode_state` |
| Long streaming memory budget | KV must be bounded by recent window; GDN/conv state fixed-size; logical token count may grow without growing state allocation. | `test_long_streaming_memory_budget_is_bounded` |
| Low-bit flexible checkpoint | Q3/Q4 must share loader/model path and CUDA/CPU dequant parity. | `test_quant_lowbit_cpu.py`, `test_lowbit_gemv_cuda.py` |
| MTP/speculative state safety | Rejected candidate tokens must not mutate parent state. | `test_simulated_speculative_reject_rolls_back_state` |
| CUDA Graph state reuse | Graph replay must match eager token/state sequence. | `test_decode1_graph_audit_blocks_current_python_state_contract`, `test_sample_next_tensor_keeps_greedy_token_as_tensor`; full eager-vs-graph replay remains skipped until a real graph path exists. |
| Process-level persistence | Process B load+continue must match process A continue. | `test_process_level_persistence_load_continue_matches` |

## Fixes Added With This Matrix

- `DecodeState.allocate()` now zero-initializes attention KV buffers. This makes
  state hashes, deltas, and snapshots deterministic.
- `DecodeState.snapshot_dict(include_attention_kv=True)` now stores physical
  ring KV buffers instead of an invalid prefix. This preserves warm-boot state
  after ring wrap.
- `DecodeState.load_snapshot()` uses `weights_only=True`.
- `EpisodicMemory.HashEmbedding` tokenizes exact facts with a regex so facts
  like `UUID=abc-042` and `phone=555-0142` are retrievable by natural queries.
- `DecodeStateDelta` was added for exact small-state delta recording and apply.
- `test_stateful_multi_turn_chat_uses_same_decode_state` verifies stateful
  multi-turn behavior on the deterministic state model.
- `test_long_streaming_memory_budget_is_bounded` verifies 1K/10K/100K ingest
  with fixed state allocation and a 1M-token logical position without state
  growth. Full 1M-token throughput telemetry remains a performance benchmark,
  not a unit-test default.
- `test_process_level_persistence_load_continue_matches` verifies snapshot
  save/load across two Python processes.
- CUDA Graph state reuse is explicitly skipped until qwenburst has a real graph
  decode path; no fake graph test is counted as coverage. `qwenburst-graph-audit`
  now records the exact blockers: Python `pos/kv_len`, Python ring-KV logical
  views, and Python-visible state mutation during `forward_one()`.

## OpenWebUI Connection

OpenWebUI container `neurova-open-webui` is configured with:

```text
OPENAI_API_BASE_URL=http://host.docker.internal:5000/v1
OPENAI_API_KEY=...
```

QwenBurst runs on host port `8008`, and a lightweight `socat` bridge exposes it
on host port `5000` for OpenWebUI:

```bash
./scripts/start_openwebui_qwenburst.sh
```

Expected model:

```text
qwenburst-qwen3.6-27b-q4-marlin
```
