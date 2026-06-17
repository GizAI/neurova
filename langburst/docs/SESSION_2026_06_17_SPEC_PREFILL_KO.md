# 2026-06-17 LangBurst Speculative / Prefill 세션 기록

이 문서는 이번 세션에서 진행한 LangBurst 추론 런타임 작업을 한곳에 정리한 기록이다. 기준 환경은 `ml-dmc8`, `langburst` conda env, RTX 4080 16GB, 모델은 `/home/user/models/Qwen3.6-27B-langburst-q3`이다.

## 목표

```text
1. Qwen3.6 Native MTP/NEXTN을 production batch verifier와 결합
2. legacy verifier / rollback / replay 파편화를 제거
3. 멀티 요청 serving에서 80 tok/s 이상 aggregate decode 확인
4. 16GB 환경에서 MTP가 OOM을 만들지 않도록 memory gate 적용
5. 다음 병목인 batch=1 prefill 속도를 근본적으로 최적화
```

## 완료한 구조 변경

### 1. Speculative verifier 단일 hot path

`Native MTP proposer -> batch verifier -> GPU reducer -> committed output` 흐름을 production 경로로 정리했다.

핵심 변경:

```text
langburst/langburst/speculative_batch.py
  resolve_speculative_gpu(...) 중심의 GPU reducer contract

langburst/langburst/engines/native/model_runner.py
  speculative row는 production forward_verify_batch hot path만 허용
  rollback-style commit_states 경로는 production에서 거부

langburst/langburst/adapters/qwen36_impl/model.py
  forward_verify_batch(...)
  _forward_verify_batch_uniform_hot(...)
  _forward_speculative_uniform_layers(...)
  _commit_speculative_trajectory(...)
```

기존 `speculative_verifier.py`는 production runtime에서 제거하고 `research/speculative_verifier.py` 쪽으로 격리했다.

### 2. GDN / conv / KV speculative CUDA op

Qwen3.6 hybrid state를 token replay로 처리하지 않기 위해 trajectory/commit 전용 CUDA op를 추가했다.

```text
gdn_recurrent_ab_spec_trajectory_out
depthwise_conv_update_spec_trajectory_out
attention_append_paged_int4_spec
```

등록/래퍼:

```text
langburst/csrc/langburst_ext.cpp
langburst/langburst/ops.py
```

검증:

```text
tests/test_v05_cuda_kernels.py
```

### 3. Scheduler / state arena 보정

MTP batch verifier는 arena-backed paged state가 필수다. 단일 slot이라도 speculative decoding이 켜져 있으면 arena를 생성하도록 고쳤다.

```text
BatchStateStore:
  speculative_decoding=True이면 single-slot도 arena 사용

ContinuousBatchScheduler:
  speculative draft row와 plain decode row를 같은 scheduled batch에 섞지 않음
```

추가 테스트:

```text
test_batch_state_store_uses_single_slot_arena_when_speculative_enabled
test_continuous_batch_scheduler_does_not_mix_speculative_and_plain_decode_rows
```

### 4. MTP memory gate

Q3 27B 16GB 환경에서 MTP proposer까지 올리면 여유 VRAM이 매우 작다. MTP가 이득을 내기 전에 OOM을 만들면 운영 가치가 없으므로 load-time / draft-time memory gate를 추가했다.

```text
LANGBURST_MTP_MIN_FREE_VRAM_MIB
SpeculativeDecodePolicy.min_free_vram_mib
RuntimeEngine._has_speculative_load_headroom()
BatchedModelRunner._has_speculative_vram_headroom()
```

의미:

```text
MTP proposer는 사용 가능하면 사용하되,
정해진 free VRAM watermark 아래에서는 draft 생성 자체를 skip한다.
```

### 5. Bench CLI 정책 정리

`bench_serving.py`가 더 이상 stateful profile과 MTP 여부를 하드코딩하지 않도록 정리했다.

```text
--runtime-profile
--enable-mtp
--disable-mtp
```

이제 benchmark도 production policy와 같은 feature resolver를 탄다.

## 빌드 / 테스트 기록

원격 CUDA extension build:

```bash
ssh ml-dmc8 'cd /home/user/workspace/neurova/langburst && \
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate langburst && \
  source scripts/langburst_cuda_env.sh && \
  export LANGBURST_CUDA_ARCH_LIST=8.9 LANGBURST_REQUIRE_CUDA_EXT=1 && \
  python -m pip install -v --no-build-isolation -e .'
```

원격 CUDA/CPU 핵심 테스트:

```text
tests/test_v05_cuda_kernels.py
tests/test_sampling_cuda.py
tests/test_gdn_parity_cuda.py
tests/test_cuda_graph_cpu.py
tests/test_model_runner_cpu.py
tests/test_speculative_batch_cpu.py
tests/test_adapter_runtime_cpu.py
tests/test_state_arena_cpu.py
tests/test_block_table_cpu.py

결과: 90 passed, 1 skipped
```

후속 CPU regression subset:

```text
tests/test_model_runner_cpu.py
tests/test_speculative_cpu.py
tests/test_scheduler_cpu.py
tests/test_state_arena_cpu.py

결과: 52 passed
```

## 성능 측정

### Q3 / int4_bdr / MTP OFF / batch=4

조건:

```text
prompt_tokens=256
max_new_tokens=64
requests=4
recent_window=2048
kv_blocks=256
```

결과:

```text
aggregate_decode_tok_s: 96.90
aggregate_output_tok_s: 19.50
mean_ttft_s: 10.49
prefill_tok_s: 약 24 tok/s
```

### Q3 / int4_bdr / MTP ON / batch=4

조건:

```text
LANGBURST_MTP_MIN_FREE_VRAM_MIB=512
prompt_tokens=256
max_new_tokens=64
requests=4
recent_window=2048
kv_blocks=256
```

결과:

```text
aggregate_decode_tok_s: 90.75
aggregate_output_tok_s: 18.83
OOM 없음
accepted_prediction_tokens: request당 1~2 수준
```

해석:

```text
batch=4 aggregate decode 80 tok/s 이상은 달성.
현재 MTP는 안정적으로 gated 되지만, 이 설정에서는 throughput champion은 아님.
```

### Q3 / int4_bdr / MTP ON / batch=1

조건:

```text
prompt_tokens=128
max_new_tokens=32
recent_window=1024
```

결과:

```text
aggregate_decode_tok_s: 34.01
accepted_prediction_tokens: 11
rejected_prediction_tokens: 5
prefill_tok_s: 328.12
```

### Sanity generation

한국어 3문장 설명 요청에서 Unicode 깨짐, 반복 붕괴, 조기 OOM 없이 정상 응답을 확인했다.

```text
completion_tokens=80
generate_s=2.724
tok_s=29.36
cuda_free_gib=1.60
```

## 현재 남은 핵심 병목

### batch=1 serving prefill

같은 Q3/int4_bdr 환경에서 batch=1 prefill이 케이스에 따라 크게 흔들린다.

관찰:

```text
prompt_tokens=128 MTP ON:  약 328 tok/s
prompt_tokens=128 MTP OFF: 약 20 tok/s
prompt_tokens=256 batch=4: request당 약 24 tok/s
```

이 수치는 과거 Q4/no-paged block SDPA 또는 GPU embed block path에서 봤던 `900~1100 tok/s`보다 낮다.

가장 유력한 원인:

```text
server forward_batch_logits() prefill 경로가 조건에 따라
_forward_prefill_paged_block_single() 대신
_forward_prefill_timestep_batch() 또는 row/token fallback을 탄다.
```

특히 `forward_batch()`와 `forward_batch_logits()`의 batch=1 paged block prefill 조건이 서로 다르다.

```text
forward_batch():
  num_requests == 1이면 paged block prefill 가능

forward_batch_logits():
  num_requests == 1이어도 canonical attention KV가 있으면 paged block prefill 차단
```

다음 작업은 이 조건 불일치를 실제 profile/benchmark로 확정하고, batch=1 serving prefill이 항상 안전한 block path를 타도록 고치는 것이다.

## batch=1 prefill 추가 최적화

### 확인한 병목

원격 `ml-dmc8`에서 batch=1, Q3/int4_bdr, prompt 256, max_new 16을 측정했다.

초기 기준:

```text
prefill_s: 12.64s
prefill_tok_s: 20.26 tok/s
decode_tok_s: 21.27 tok/s
```

원인:

```text
1. Q3/int4 KV에서는 no-paged여도 fp16 SDPA block prefill을 탈 수 없음
2. serving forward_batch_logits()가 256 token prefill을 _forward_prefill_timestep_batch()로 보내고 있었음
3. int4 short SDPA path에서도 persistent int4 KV append가 token별 Python loop였음
```

### 적용한 변경

```text
1. batch=1 canonical prefill은 timestep batch가 아니라 forward_block()을 우선 사용
2. int4 forward_block()에 short prefill SDPA staging 추가
   - persistent KV는 int4_bdr로 유지
   - attention 계산용 fp16 staging은 LANGBURST_SHORT_PREFILL_SDPA_TOKENS 이하에서만 사용
   - free VRAM watermark 부족 시 staging 자동 비활성화
3. DecodeState.append_attention_kv_block_at(...) 추가
   - int4/BDR KV block을 한 번에 pack/write
   - token별 append_attention_kv_at loop 제거
4. start script 기본값 정리
   - LANGBURST_BATCH_STATE_ARENA=auto
   - LANGBURST_SHORT_PREFILL_SDPA_TOKENS=2048
   - LANGBURST_ATTENTION_RECENT_TOKENS=128
```

### 최종 측정

동일 조건:

```text
prompt_tokens=256
max_new_tokens=16
requests=1
recent_window=2048
kv_cache_dtype=int4_bdr
MTP OFF
```

결과:

```text
prefill_s: 5.51s
prefill_tok_s: 46.49 tok/s
decode_tok_s: 21.38~21.86 tok/s
TTFT: 5.51s
```

개선:

```text
prefill_tok_s: 20.26 -> 46.49 tok/s
TTFT: 12.65s -> 5.51s
decode 속도 유지
```

주의:

```text
LANGBURST_BATCH_STATE_ARENA=1 강제는 prefill을 약 46 tok/s로 올리지만
decode가 약 8 tok/s까지 떨어져 기본 champion이 아니다.
따라서 운영 기본은 auto가 맞다.
```

남은 큰 병목:

```text
Q3/int4_bdr prefill은 아직 true FlashAttention-style int4 block prefill이 아니다.
현재는 short fp16 staging SDPA + int4 persistent KV 유지 구조다.
700~900 tok/s를 다시 노리려면 attention 계산 자체를 Python/PyTorch SDPA 경계가 아니라
int4_bdr paged FlashAttention-style CUDA block prefill op로 내려야 한다.
```

## 다음 최적화 순서

```text
1. batch=1 prefill baseline 재측정
2. forward_batch_logits prefill route tracing
3. safe paged block prefill 조건을 forward_batch와 일치
4. parity / continuation / long prompt sanity test
5. batch=1 prefill tok/s 재측정
6. 이후 multi-request prefill은 prefix/state cache와 true multi-row block prefill로 확장
```

## 운영상 결론

현재 LangBurst는 Q3/int4_bdr 기준으로 multi-request decode throughput은 80 tok/s 이상까지 확인했다. 그러나 batch=1 TTFT/prefill은 아직 최종 상태가 아니며, 다음 성능 개선의 최우선 대상은 MTP가 아니라 prefill route 정리다.

## 2026-06-17 추가 정리: prefill 병목 원인과 최종 기본값

### 근본 원인

Q3/int4_bdr prefill이 46 tok/s까지 떨어진 직접 원인은 두 가지였다.

```text
1. prefill_chunk_size=256인데 LANGBURST_MARLIN_DIRECT_MAX_BATCH=64라
   Marlin projection이 block GEMM이 아니라 row loop fallback으로 실행됨

2. LANGBURST_SHORT_PREFILL_SDPA_TOKENS=512라
   1K 이상 prompt에서 3번째 chunk부터 large-past INT4 direct path로 넘어가며 급격히 느려짐
```

### 최종 정책

```text
Decode:
  INT4_BDR paged KV 유지
  CUDA Graph hot path 유지

Fresh prefill:
  BF16/FP16 SDPA scratch만 사용
  attention 후 persistent KV는 즉시 INT4_BDR append
  state에 _prefill_fp16_kv 같은 shadow cache를 남기지 않음

Chunked/extend prefill:
  live context <= LANGBURST_SHORT_PREFILL_SDPA_TOKENS이면 dequant+SDPA
  그 이상은 tiled INT4_BDR fused attention kernel 대상

Scheduler:
  decode row는 batch 유지
  prefill row는 기본 1개씩 block fast path로 스케줄
  max_prefill_rows_per_batch는 EngineResourcePolicy/env/CLI에서만 설정

Precision:
  persistent KV 기본은 INT4_BDR
  prefill attention scratch만 FP16/BF16
  GDN state 정책은 attention KV 정책과 분리
```

### 코드 정리

```text
삭제/정리:
  short_prefill_sdpa_staging()
  state._prefill_fp16_kv cleanup 경로
  model 코드의 scattered short_prefill env 직접 참조

추가/정규화:
  PrefillAttentionPolicy
  EngineResourcePolicy.max_prefill_rows_per_batch
  prefill_chunk_size <= marlin_direct_max_batch 자동 cap
```

### 최신 측정

ml-dmc8 / RTX 4080 / Qwen3.6-27B-langburst-q3 / int4_bdr / MTP OFF:

```text
256 prompt, 16 generation:
  prefill_tok_s: 518.11 tok/s
  TTFT:          0.50s
  decode_tok_s: 20.96 tok/s

1024 prompt, 16 generation:
  prefill_tok_s: 933.91 tok/s
  TTFT:          1.11s
  decode_tok_s: 21.14 tok/s

2048 prompt, 8 generation:
  prefill_tok_s: 1040.81 tok/s
  TTFT:          1.98s
  decode_tok_s: 22.34 tok/s
```

### 남은 병목

```text
1. 2명 이상이 동시에 prefill 중이면 scheduler/arena 경로가 아직 batch=1만큼 빠르지 않다.
   현재 기본은 prefill row를 1개씩 처리해 빠른 block path를 보존한다.

2. LANGBURST_SHORT_PREFILL_SDPA_TOKENS를 넘는 long-past prefill은 아직 최종 fused kernel이 아니다.
   진짜 최종형은 INT4_BDR paged KV를 직접 읽는 tiled FlashAttention-style block prefill CUDA op다.

3. prefill path parity는 argmax/continuation gate를 통과했지만 내부 state max diff는 0이 아니다.
   완전 exact state parity가 필요한 테스트에서는 별도 gate로 유지해야 한다.
```
