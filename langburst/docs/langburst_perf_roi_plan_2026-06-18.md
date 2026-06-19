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
  decode 약 66.7 tok/s

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

운영 SSOT는 `configs/ml-dmc8-q4.yaml`이다. `scripts/start_langburst.sh`는 이
YAML을 읽어 기존 low-level 모듈이 요구하는 호환 환경변수와 CLI 인자를 한 번만
생성한다. 핵심 값은 다음과 같다.

```yaml
model:
  name: langburst-qwen3.6-27b-q4
  qb_model: /home/user/models/Qwen3.6-27B-qb4-marlin-fused
serving:
  context_window: 65536
  context_tiers: [4096, 65536]
  context_tier_slots: [2, 1]
  max_active_requests: 3
  default_max_tokens: 256
  min_completion_budget: 0
kv:
  cache_dtype: int4_bdr
marlin:
  out_cache_policy: all
  cache_max_mib: 32
  cache_min_free_mib: 256
  internal_argmax: true
mtp:
  max_draft: 4
  draft_candidates: 4
  max_draft_by_active: "2:1"
  batch_proposer: true
  legacy_list_cache: false
  local_tkh_attention: false
runtime:
  trim_model_cache_before_prefill: true
  trim_model_cache_prefill_free_below_mib: 1024
  repetition_stop_min_ngram_size: 32
  repetition_stop_ngram_size: 96
  repetition_stop_repeats: 2
kernels:
  gdn_ba_lowbit_pair: true
  gdn_spec_norm_gate_fused: true
  mlp_tensorcore_down_silu_a: true
  spec_trajectory_copy_cuda: true
  cuda_graph: false
  verify_full_logits: false
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
aggregate_decode_tok_s: 66.75
aggregate_output_tok_s: 63.75
accepted/rejected: 314 / 478
scheduled_batches: 199
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

### 긴 반복 출력 RCA

OpenWebUI 긴 대화에서 보였던 `요약\n요약...` 또는 긴 입력 구문 복사형 반복은
MTP만의 문제가 아니었다. 직접 `langburst.generate`로 같은 긴 반복 입력을 target-only와
MTP 양쪽에서 돌렸을 때는 정상 요약이 나왔고, API/chat 경로에서는 긴 반복 입력을 그대로
답변으로 복사하다가 `finish_reason=length`로 끝나는 케이스가 재현됐다.

구조적 원인은 기존 반복 종료 guard가 최대 8토큰 n-gram만 검사했다는 점이다. 짧은 단어
반복은 잡지만, 한국어 문장이나 입력 문단처럼 10토큰 이상인 구간이 반복되면 같은 suffix가
여러 번 이어져도 반복으로 판정하지 못했다. 이 문제는 logits를 억지로 누르는 suppression이
아니라, 이미 생성된 suffix가 같은 구간을 여러 번 반복하는 정확한 퇴화 패턴을 종료 조건으로
처리하는 문제다.

이번 정리 후 운영 기본값은 다음이다.

```text
repetition_stop_min_ngram_size: 32
repetition_stop_ngram_size: 96
repetition_stop_repeats: 2
```

추가 CPU gate:

```text
test_batch_generation_handle_stops_repeated_phrase_loop
test_batch_generation_handle_does_not_stop_non_repeated_long_tail
```

이 변경은 정상적인 긴 답변을 짧게 자르는 정책이 아니라, 같은 생성 suffix가 2회 반복되는
degenerate tail만 `finish_reason=repetition`으로 종료한다. `finish_reason=length`는
서버/클라이언트의 `max_tokens` 예산을 다 쓴 것이므로 조기 EOS와 구분해서 봐야 한다.

남은 조기 EOS 이슈도 분리해서 확인했다.

```text
repro:
  같은 한국어 문장을 약 140회 반복한 뒤 "한 문장으로 요약" 요청

observed:
  finish_reason=stop
  finish_detail=eos_token:248046 (<|im_end|>)
  completion_tokens=13~16
  출력이 첫 반복 문장을 짧게 인용하고 끝남

not root cause:
  CUDA OOM 아님
  stream truncation 아님
  MTP verifier 단독 문제로 확정되지 않음
```

`min_tokens=48`을 요청에 명시하면 이 재현 프롬프트는 정상 요약으로 회복됐지만,
`min_tokens=16/32`에서는 chat role 문자열이 출력에 섞이는 leakage가 나타났다. 따라서
전역 최소 생성 길이를 기본값으로 넣는 것은 정답 구조가 아니다. 다음 수정 대상은 Qwen
non-thinking chat template이 만드는 빈 `<think>\n\n</think>\n\n` 블록과 긴 반복 입력의
greedy EOS 확률이 결합되는 경로를 더 좁혀, 정상 짧은 요청과 속도를 희생하지 않는
template/decoding contract를 찾는 것이다.

2026-06-19 추가 재현:

```text
turns:
  1. 안녕?
  2. 자기소개해봐
  3. 더 자세히

before:
  OpenWebUI/API가 max_tokens=256을 보내면 2턴이 finish_reason=length로 잘림
  잘린 assistant history가 3턴 prompt에 들어가며 3턴이
  "저는 Qwen이라고 하는 대규모 언어" 같은 중간 단어에서 eos_token으로 종료

second root:
  Qwen3.6 tokenizer의 non-thinking generation prompt는
  <think>\n\n</think>\n\n 빈 thinking 블록으로 시작한다.
  그런데 이전 assistant history는 preserve_thinking=false일 때 thinking 블록 없이
  저장되어, generation prompt와 assistant history 형식이 달라진다.
  이 불일치가 "더 자세히" 같은 follow-up에서 이전 assistant 답변 반복과 조기 EOS를 유발한다.
```

수정:

```text
1. serving.default_max_tokens=256 / min_completion_budget=0
   클라이언트가 명시한 짧은 max_tokens를 서버가 임의로 늘리지 않는다. 대신 long-document
   context selection, stop filtering, repetition suffix detector로 history 오염을 줄인다.

2. Qwen36Adapter.encode_messages()
   thinking이 꺼진 기본 경로에서는 chat_template_kwargs.preserve_thinking도 false로 둔다.
   thinking을 명시적으로 켠 요청에서만 preserve_thinking이 함께 켜지며, 별도 override도 허용한다.
```

검증:

```text
non-stream API, max_tokens=256:
  안녕?          completion=30,  finish=stop
  자기소개해봐   completion=normal, finish=stop
  더 자세히      completion=normal, finish=stop

stream API, max_tokens=256:
  안녕?          chunks=26,  completion=30
  자기소개해봐   chunks=normal, completion=normal
  더 자세히      chunks=normal, completion=normal

더 이상 "대규모 언어" 같은 중간 단어에서 멈추지 않음.
```

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
  GDN_BA_LOWBIT_PAIR
  GDN_SPEC_NORM_GATE_FUSED
  MLP_TENSORCORE_DOWN_SILU_A
  SPEC_TRAJECTORY_COPY_CUDA
  MARLIN_INTERNAL_ARGMAX

OFF:
  CUDA_GRAPH
  VERIFY_FULL_LOGITS
  GDN_RECURRENT_NORM_GATE_FUSED
  MTP_FC_SPLIT
```

CUDA Graph는 아직 production 기본값으로 켜지 않는다. 현재 성능 병목은 graph launch
overhead보다 verifier/GDN/MLP workspace와 projection 비용 쪽이 크다.

## 검증

로컬:

```text
python -m pytest -q \
  langburst/tests/test_config_cpu.py \
  langburst/tests/test_tuning_cpu.py \
  langburst/tests/test_model_runner_cpu.py \
  langburst/tests/test_speculative_batch_cpu.py \
  langburst/tests/test_speculative_cpu.py

결과: 52 passed
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
prompt: 간단히 서울의 장점을 두 문장으로 말해줘.
result: 자연스러운 한국어 2문장
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

1. Marlin lm_head 내부 argmax 추가 검증
   vocab tile별 max/index를 Marlin write path 안에서 누적하는 경로는 구현됐다.
   batch 1/2/4/8 `gemm_argmax == gemm().argmax()` sanity도 통과했다.
   2026-06-19 YAML 전환 후 K4/512 bucket에서는 `MARLIN_INTERNAL_ARGMAX=1`이
   65.51 tok/s, `0`이 65.24 tok/s로 소폭 speed-positive였다. 이 기준으로
   `configs/ml-dmc8-q4.yaml` 기본값은 ON이다. 단, 이전 세션 중 OFF가 더 빨랐던
   run도 있었으므로 계속 champion bucket에서 gate한다.

2. GDN spec recurrent/norm/gate fused kernel 유지
   2026-06-19에 fused kernel의 q normalization trajectory를 기존
   `gdn_recurrent_ab_spec_trajectory + rmsnorm_silu_gate` 경로와 맞췄다.
   K4/256 bucket에서 accepted/rejected/scheduled가 `147/285/109`로 동일해졌고
   속도는 60.04 -> 60.84 tok/s로 개선됐다. K4/512 bucket도
   `314/478/199` trajectory 동일 조건에서 65.87 -> 66.75 tok/s로 개선되어
   `configs/ml-dmc8-q4.yaml` 기본값은 ON이다.

3. MLP gate_up/down streaming fusion
   `gate_up -> SiLU*up -> down` 사이의 global memory 왕복을 더 줄여야 한다.

4. speculative verifier workspace 절감
   batch=3에서 K>=2가 OOM 나는 직접 원인이다. trajectory 전체 보존 대신 selected
   state만 유지하는 더 작은 commit-aware CUDA contract가 필요하다.

5. long-past INT4_BDR fused block prefill
   fresh prefill은 900 tok/s급이지만, 긴 past extend prefill은 tiled INT4_BDR
   attention kernel이 더 필요하다.

6. prefill benchmark 조건 분리
   `prompt_tokens > recent_window` 벤치는 ring/window overflow 비용까지 측정한다.
   예를 들어 `prompt_tokens=4096`, `recent_window=2048`은 2K 이후 window update
   path가 개입해 약 64 tok/s까지 떨어진다. long-context prefill 성능 비교는
   `recent_window >= prompt_tokens` 또는 실제 운영 tier window에서 측정한다.

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
