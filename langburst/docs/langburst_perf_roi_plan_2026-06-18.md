# LangBurst Qwen3.6 Q4 성능/VRAM 최종 연구 기록

작성일: 2026-06-19

대상:

- 호스트: `ml-dmc8`
- GPU: RTX 4080 16GB
- 모델: `/home/user/models/Qwen3.6-27B-qb4-marlin-fused`
- 서버 모델명: `langburst-qwen3.6-27b-q4`
- 기본 KV: `int4_bdr`
- 기본 serving 목표: 4K 소형 슬롯 2개 + 64K 대형 슬롯 1개, 최대 active 3

이 문서는 이번 세션의 최종 판단만 남긴다. 중간에 실패한 값이나 예전 champion 값은
같은 조건에서 재현된 경우만 기록한다.

## 최종 결론

현재 트리에서 검증된 최적 운영 방향은 다음이다.

```text
single request:
  MTP K=4 유지
  decode 약 66.2 tok/s

2+ active requests:
  MTP draft를 K=1로 자동 cap
  batch=3 aggregate decode 약 80.2 tok/s

long context serving:
  context tiers 4096,65536 / slots 2,1
  Marlin runtime cache cap 32 MiB
  prefill 전 model runtime cache pressure trim
```

중요한 점은 `K=4`를 모든 상황에 고정하면 16GB에서 동시 요청 중 verifier/GDN
trajectory workspace가 OOM을 만든다는 것이다. 단일 요청 속도와 3동시 안정성을 동시에
얻으려면 active pressure 기반 draft cap이 필요하다.

## 운영 기본값

`scripts/start_langburst.sh`의 핵심 기본값:

```bash
MODEL_NAME=langburst-qwen3.6-27b-q4
KV_CACHE_DTYPE=int4_bdr

CONTEXT_TIERS=4096,65536
CONTEXT_TIER_SLOTS=2,1
CONTEXT_WINDOW=65536
MAX_ACTIVE_REQUESTS=3

MARLIN_OUT_CACHE_POLICY=all
LANGBURST_MARLIN_CACHE_MAX_MIB=32
LANGBURST_MARLIN_CACHE_MIN_FREE_MIB=256
LANGBURST_TRIM_MODEL_CACHE_BEFORE_PREFILL=1
LANGBURST_TRIM_MODEL_CACHE_PREFILL_FREE_BELOW_MIB=1024

LANGBURST_MTP_MAX_DRAFT=4
LANGBURST_MTP_DRAFT_CANDIDATES=4
LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE=2:1
LANGBURST_MTP_BATCH_PROPOSER=1
LANGBURST_MTP_LEGACY_LIST_CACHE=0
LANGBURST_MTP_LOCAL_TKH_ATTENTION=1
LANGBURST_MTP_FC_SPLIT=0

LANGBURST_MARLIN_INTERNAL_ARGMAX=1
LANGBURST_GDN_BA_LOWBIT_PAIR=1
LANGBURST_MLP_TENSORCORE_DOWN_SILU_A=1
LANGBURST_SPEC_TRAJECTORY_COPY_CUDA=1
LANGBURST_CUDA_GRAPH=0
LANGBURST_VERIFY_FULL_LOGITS=0
```

`LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE=2:1`은 단일 요청에서는 K=4를 유지하고,
서버 pressure가 2개 이상이면 기존 draft와 새 draft를 K=1로 제한한다. 이 값은
속도 옵션이 아니라 OOM 방지와 aggregate throughput을 동시에 맞추는 serving 정책이다.

## 실측 결과

### 단일 요청 decode

조건:

```text
recent_window: 2048
prompt_tokens: 1
max_new_tokens: 512
requests: 1
MTP: K=4
cache: benchmark-only cache cap 0
```

결과:

```text
aggregate_decode_tok_s: 66.21
aggregate_output_tok_s: 63.75
accepted/rejected: 312 / 484
scheduled_batches: 200
```

비교:

```text
K=5:
  64.89 tok/s
  rejected 증가로 K=4보다 느림
```

현재 트리에서는 K=4가 단일 요청 champion이다. K=5는 켜지 않는다.

### batch=3 decode

조건:

```text
recent_window: 2048
prompt_tokens: 1
max_new_tokens: 128
requests: 3
MTP base K=4
active pressure cap: 2:1
```

결과:

```text
aggregate_decode_tok_s: 80.23
aggregate_output_tok_s: 75.52
avg_scheduled_tokens_per_batch: 4.62
scheduled_batches: 120
```

per request:

```text
req1: decode 39.78 tok/s, accepted/rejected 47 / 33
req2: decode 40.45 tok/s, accepted/rejected 48 / 31
req3: decode 36.67 tok/s, accepted/rejected 42 / 106
```

K=2/3/4를 3동시로 그대로 유지하면 16GB에서 12~20 MiB allocation OOM이 났다.
전역 K=1 또는 pressure cap K=1은 OOM 없이 통과했다.

### prefill

대표 long prefill smoke:

```text
prompt_tokens: 약 5.2K~5.7K
prefill: 약 906~908 tok/s
status: ok
```

짧은 요청은 prompt 길이가 너무 작아 prefill tok/s가 낮게 보일 수 있다. 긴 입력에서
실제 prefill path는 900 tok/s급을 회복했다.

## VRAM 분석

64K/3-slot arena:

```text
context tiers: 4096,65536
slots: 2,1
arena total: 약 1412 MiB
paged_kv: 약 1188 MiB
gdn_recurrent: 약 216 MiB
```

80K급 대형 슬롯은 16GB에서 너무 빡빡했다.

```text
4096,81920:
  arena total: 약 1536 MiB
  first request 후 free VRAM: 약 15 MiB
  20 MiB allocation도 OOM

4096,65536:
  arena total: 약 1412 MiB
  stress 후 free VRAM: 약 529 MiB
  short/long/3 concurrent smoke 통과
```

따라서 16GB 운영 기본은 64K 대형 슬롯이 현실적인 상한이다. 더 큰 context는
24GB 이상 GPU나 더 강한 KV/state workspace 절감이 필요하다.

## 이번 세션에서 남긴 코드 변경

### Marlin runtime cache SSOT

추가된 구조:

```text
marlin_runtime_cache_bytes()
marlin_runtime_cache_summary()
clear_marlin_runtime_caches()
marlin_cache_admitted()
```

목적:

- `cache=all`의 single decode 속도 이득 유지
- 긴 prefill/동시 요청에서는 cache 성장을 제한
- OOM 발생 시 Marlin runtime cache를 먼저 비우고 allocation retry
- health/debug에서 runtime cache bytes 확인 가능

### Serving pressure 기반 MTP draft cap

추가된 구조:

```text
LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE=2:1
BatchedModelRunner.set_serving_pressure_request_count()
ContinuousBatchScheduler.cap_active_draft_tokens()
BatchGenerationWorker._update_runner_pressure_count()
```

목적:

- pending/deferred request까지 포함해 serving pressure를 감지
- 이미 붙은 K4 draft도 pressure 진입 시 K1로 truncate
- pressure signature가 바뀔 때 model runtime cache와 allocator cache를 정리

### Native MTP proposer 정리

핵심 변경:

- single-row batch proposer는 scalar proposer로 fallback하여 draft trajectory를 보존
- `LANGBURST_MTP_LEGACY_LIST_CACHE=0`을 기본으로 사용
- `LANGBURST_MTP_FC_SPLIT=0` 기본값. 현재 트리에서는 split path가 확실한 이득이 아님

### CUDA/ops hot path

추가 또는 유지:

```text
copy_selected_trajectory_out
gdn_recurrent_ab_batch_norm_gate
gdn_recurrent_ab_spec_trajectory_norm_gate
lowbit_marlin_gemm_argmax_out
lowbit_marlin_gemm_silu_packed_out
```

기본 ON/OFF 판단:

```text
ON:
  MARLIN_INTERNAL_ARGMAX
  GDN_BA_LOWBIT_PAIR
  MLP_TENSORCORE_DOWN_SILU_A
  SPEC_TRAJECTORY_COPY_CUDA

OFF:
  CUDA_GRAPH
  VERIFY_FULL_LOGITS
  GDN_RECURRENT_NORM_GATE_FUSED
  GDN_SPEC_NORM_GATE_FUSED
  MTP_FC_SPLIT
```

CUDA Graph는 아직 production 기본값으로 켜지 않는다. 현재 성능 병목은 graph launch
overhead보다 verifier/GDN/MLP workspace와 projection 비용 쪽이 크다.

## 검증

로컬:

```text
python -m pytest -q \
  langburst/tests/test_model_runner_cpu.py \
  langburst/tests/test_speculative_batch_cpu.py \
  langburst/tests/test_speculative_cpu.py

결과: 46 passed
```

원격 `ml-dmc8`:

```text
CUDA_VISIBLE_DEVICES= python -m pytest -q \
  tests/test_model_runner_cpu.py \
  tests/test_quant_lowbit_cpu.py \
  tests/test_speculative_batch_cpu.py \
  tests/test_speculative_cpu.py

결과: 51 passed
```

API smoke:

```text
POST http://127.0.0.1:8008/v1/chat/completions
prompt: 안녕. 한 문장으로 답해줘.
result: 안녕하세요.
status: OK
```

OpenWebUI backend:

```text
http://192.168.0.47:8008/v1
model: langburst-qwen3.6-27b-q4
max_concurrency: 3
```

## 남은 고ROI 작업

싱글 80 tok/s는 아직 미달이다. 현재 single 66 tok/s에서 80 tok/s까지는 약 21%의
추가 개선이 필요하다. 단순 옵션 스윕으로는 어렵고 아래 순서가 현실적이다.

1. Marlin lm_head 내부 argmax 완전 fusion
   현재는 GEMM 결과와 argmax 경로가 완전히 하나의 Marlin tile reduction으로 합쳐진
   상태가 아니다. vocab tile별 max/index를 Marlin write path 안에서 누적해야 한다.

2. GDN recurrent/norm/gate fused kernel parity 후 기본 ON 재검토
   실제 Q4 shape에 맞춘 커널은 있으나 아직 기본 ON 실측 champion은 아니다.

3. MLP gate_up/down streaming fusion
   `gate_up -> SiLU*up -> down` 사이의 global memory 왕복을 더 줄여야 한다.

4. speculative verifier workspace 절감
   batch=3에서 K>=2가 OOM 나는 직접 원인이다. trajectory 전체 보존 대신 selected
   state만 유지하는 더 작은 commit-aware CUDA contract가 필요하다.

5. long-past INT4_BDR fused block prefill
   fresh prefill은 900 tok/s급이지만, 긴 past extend prefill은 tiled INT4_BDR
   attention kernel이 더 필요하다.

## 운영 규칙

기본값으로 올리는 기준:

```text
1. 같은 benchmark bucket에서 tok/s 상승
2. accepted/rejected/scheduled trajectory가 설명 가능
3. long prefill과 3동시 OOM smoke 통과
4. 코드 경계가 production/research/debug로 분리
```

금지:

```text
synthetic parity만 보고 기본 ON
OOM을 숨기기 위해 context를 무작정 낮추기
단일 요청 champion을 깨는 옵션을 운영 기본값으로 유지
같은 정책을 proposer/scheduler/model에 중복 구현
```
