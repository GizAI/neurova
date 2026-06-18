# LangBurst Qwen3.6 Q4/MTP 성능 개선 ROI 계획

작성일: 2026-06-18  
대상: `langburst` native runtime, Qwen3.6-27B Q4 Marlin fused, MTP/NEXTN speculative decode  
전제: 현재 최고 운영 조합은 **기능을 많이 켜는 조합이 아니라 champion trajectory를 유지하는 최소 조합**이다.

---

## 1. 현재 기준선

최종 A/B 기준 운영 default:

```bash
LANGBURST_CUDA_GRAPH="${LANGBURST_CUDA_GRAPH:-0}"
LANGBURST_MTP_BATCH_PROPOSER="${LANGBURST_MTP_BATCH_PROPOSER:-0}"
LANGBURST_MLP_TENSORCORE_DOWN_SILU_A="${LANGBURST_MLP_TENSORCORE_DOWN_SILU_A:-1}"
LANGBURST_MARLIN_INTERNAL_ARGMAX="${LANGBURST_MARLIN_INTERNAL_ARGMAX:-0}"
LANGBURST_GDN_BA_LOWBIT_PAIR="${LANGBURST_GDN_BA_LOWBIT_PAIR:-0}"
LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED="${LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED:-0}"
```

최근 champion bucket:

```text
Q4 fused
recent_window=2048
prompt_tokens=1
max_new_tokens=512
MTP K=5
Marlin cache=decode_only
MTP legacy/list path
TKH attention OFF
```

측정 요약:

```text
baseline:
  CUDA_GRAPH=0
  MTP_BATCH_PROPOSER=0
  MLP_TENSORCORE_DOWN_SILU_A=0
  MARLIN_INTERNAL_ARGMAX=0
  aggregate_decode_tok_s: 62.76

MLP_TENSORCORE_DOWN_SILU_A=1:
  aggregate_decode_tok_s: 64.53, repeat 65.11
  accepted/rejected: 324 / 616
  scheduled_batches: 189
  결론: speed-positive, champion trajectory 유지, 기본 ON

CUDA_GRAPH=1:
  aggregate_decode_tok_s: 61.69
  결론: static-buffer graph 구현은 보존, 운영 기본 OFF

MTP_BATCH_PROPOSER=1:
  aggregate_decode_tok_s: 58.99
  accepted/rejected: 312 / 683
  scheduled_batches: 200
  결론: trajectory 회귀, 운영 기본 OFF

GDN_BA_LOWBIT_PAIR=1:
  aggregate_decode_tok_s: 55.15
  accepted/rejected: 315 / 670
  결론: 운영 기본 OFF

MARLIN_INTERNAL_ARGMAX=1:
  aggregate_decode_tok_s: 62.26~63.52
  MLP=1과 같이 켜도 64.14로 MLP 단독보다 낮음
  결론: 운영 기본 OFF

MLP_TENSORCORE_DOWN_SILU_A=1 + CUDA_GRAPH=1:
  aggregate_decode_tok_s: 64.21
  결론: MLP 단독보다 낮아 graph 기본 OFF
```

---

## 2. ROI 우선순위

| ROI | 과제 | 현재 상태 | 기대 이득 | 기본 정책 |
|---:|---|---|---|---|
| 1 | MTP batch proposer parity repair | ON 시 trajectory 회귀 | legacy trajectory 유지 + proposer overhead 감소 | OFF 유지, parity 후 재평가 |
| 2 | Marlin internal argmax v2 | op 존재, speed-positive 아님 | lm_head argmax scratch/materialization 감소 | OFF 유지 |
| 3 | speculative resolve/cache_update fused kernel | post-verify commit 경로 파편화 | launch/dispatch 감소 | 신규 구현 후보 |
| 4 | CUDA Graph micrograph화 | full verifier graph는 느림 | resolve/cache_update 일부 graph화 | full graph OFF |
| 5 | GDN recurrent+norm+gate shape 대응 | 실제 Q4 shape mismatch | GDN post recurrent launch/materialization 감소 | OFF 유지 |
| 6 | MLP fused path hot 분기 정리 | 이미 ON, speed-positive | 소폭 overhead 감소 | ON 유지 |
| 7 | 72K prefill/chunk autotune | 72K는 메모리 여유 얇음 | long prompt 안정성/속도 개선 | 신중한 실험 |
| 8 | paged attention/KV append tuning | long context에서 비중 증가 | 72K decode 안정성/속도 개선 | 중기 과제 |
| 9 | GDN BA low-bit pair 재검증 | 현재 느리고 trajectory 악화 | 낮음, quant/scale 재검증 필요 | OFF 유지 |
| 10 | scalar full-streaming MLP | reference/debug | production 이득 낮음 | 하지 않음 |

---

## 3. ROI 1: MTP batch proposer parity repair

### 문제

`LANGBURST_MTP_BATCH_PROPOSER=1`은 speed만 낮은 것이 아니라 draft trajectory 자체를 바꾼다.

```text
legacy/batch OFF:
  accepted/rejected/scheduled = 324 / 616 / 189

batch proposer ON:
  accepted/rejected/scheduled = 312 / 683 / 200
```

이 상태에서는 batch proposer를 최적화해도 의미가 없다. 먼저 legacy와 같은 draft sequence를 만들어야 한다.

### 목표

```text
B=1에서 legacy proposer와 batch proposer draft token sequence bit-identical
그 다음 B=2/4에서 batch proposer speed 이득 확인
```

### 구현 계획

1. legacy proposer와 batch proposer의 step별 draft token을 로그/텐서로 비교한다.
2. `first_token`, `raw_hidden`, `pos` signal이 완전히 같은지 검증한다.
3. MTP local KV/list cache 업데이트 순서가 legacy와 같은지 확인한다.
4. B=1에서 token sequence parity를 통과시킨다.
5. B=2/4 continuous batching에서 speed를 측정한다.

### 성공 기준

```text
accepted/rejected/scheduled = 324 / 616 / 189 유지
aggregate_decode_tok_s > MLP=1 단독 65.11 tok/s
B=2 이상에서 proposer overhead 감소 확인
```

---

## 4. ROI 2: Marlin internal argmax v2

### 문제

현재 `LANGBURST_MARLIN_INTERNAL_ARGMAX=1`은 op가 존재하지만 speed-positive가 아니다.

```text
MARLIN_INTERNAL_ARGMAX=1:
  aggregate_decode_tok_s: 62.26~63.52

MLP=1 + MARLIN_INTERNAL_ARGMAX=1:
  aggregate_decode_tok_s: 64.14

MLP=1 단독:
  aggregate_decode_tok_s: 64.53~65.11
```

### 가능성

Verifier lm_head는 매 decode step에서 반복된다. argmax만 필요할 때 full logits/scratch materialization을 줄이면 이론적으로 이득이 있다.

### 의심 병목

```text
- global spin-wait / atomic sync 비용
- argmax_state 초기화/완료 동기화 비용
- tie-breaking deterministic 처리 비용
- batch*T row 수가 작을 때 kernel 내부 argmax overhead가 더 큼
```

### 구현 계획

1. `lowbit_marlin_gemm_argmax_out` 내부 sync 비용 측정.
2. row 수별 benchmark: `M=1`, `M=5`, `M=6`, `M=10`.
3. full logits argmax와 token parity 확인.
4. global spin-wait를 줄이거나 per-row/block reduction 구조로 재작성.
5. MLP=1 기준에서 재측정.

### 성공 기준

```text
MLP=1 + ARGMAX=1 > MLP=1 단독 65.11 tok/s
accepted/rejected/scheduled = 324 / 616 / 189 유지
```

---

## 5. ROI 3: speculative resolve/cache_update fused kernel

### 문제

Verifier 후처리 경로가 여러 단계로 흩어져 있다.

```text
resolve_speculative_gpu
commit_tokens 계산
conv trajectory select
GDN trajectory select
paged KV append
arena.pos / arena.kv_len advance
state.last_raw_hidden update
```

이 흐름은 매 speculative decode step마다 실행된다.

### 가능성

Full CUDA Graph보다 작은 범위를 직접 fused kernel로 묶는 것이 ROI가 높다. full verifier graph는 느렸지만, post-verify commit은 fixed-shape/작은 tensor 중심이라 통합 효과가 있을 수 있다.

### 구현 계획

1. 현재 `cache_update` profile time 측정.
2. `commit_tokens` 기반으로 conv/gdn trajectory select + arena state copy를 단일 CUDA op로 통합.
3. paged KV append spec path와 arena advance를 가능하면 같은 launch군으로 묶는다.
4. Python list 기반 trajectory append를 static workspace 구조로 정리한다.

### 성공 기준

```text
cache_update profile time 감소
aggregate_decode_tok_s 개선
trajectory parity 유지
```

---

## 6. ROI 4: CUDA Graph micrograph화

### 현재 결론

Full verifier graph는 구현되어 있지만 현재 champion bucket에서 느리다.

```text
CUDA_GRAPH=1: 61.69 tok/s
MLP=1 + CUDA_GRAPH=1: 64.21 tok/s
MLP=1 단독: 65.11 tok/s
```

따라서 full verifier graph는 운영 기본 OFF가 맞다.

### 새 방향

전체 model forward가 아니라 다음처럼 작은 영역만 graph화한다.

```text
- resolve_speculative_gpu
- cache_update
- arena.advance_slots
- argmax_many_out 일부
```

### 성공 기준

```text
MLP=1 + micrograph > MLP=1 단독
full verifier graph보다 memory overhead 작음
```

---

## 7. ROI 5: GDN recurrent+norm+gate 실제 shape 대응

### 문제

`gdn_recurrent_ab_batch_norm_gate` op는 존재하지만 실제 Q4 model shape에서 `norm_w hidden mismatch`가 났다.

### 가능성

GDN recurrent output 이후 RMSNorm + SiLU gate + out projection 전처리를 줄일 수 있다. 다만 현재 ROI는 MTP/argmax/commit보다 낮다.

### 구현 계획

1. 실제 Qwen3.6 Q4 fused shape dump.
2. `z`, `gdn_norm_w`, `core` shape 계약 재정의.
3. Qwen3.6 linear_value_head layout에 맞춘 fused kernel variant 추가.
4. random parity가 아니라 실제 layer weight/tensor에서 parity test.
5. K=5 champion bucket에서 A/B.

### 성공 기준

```text
shape mismatch 없음
실제 model parity OK
aggregate_decode_tok_s 개선
```

---

## 8. ROI 6: MLP fused path hot 분기 정리

### 현재 상태

`LANGBURST_MLP_TENSORCORE_DOWN_SILU_A=1`은 이미 speed-positive다.

### 할 일

1. MLP path에서 env check를 layer hot path마다 반복하지 않도록 model init 시 flag로 고정한다.
2. `Qwen36MLP` 내부의 reference/debug path와 production path를 더 명확히 분리한다.
3. `_streaming_mlp_scalar_reference`는 debug-only라는 이름/주석을 유지한다.
4. production path는 `gate_up Marlin + down.gemm_silu_packed`로 단순화한다.

### 기대 이득

크지는 않지만 이미 winner path라 소폭 개선과 코드 안정성 측면에서 ROI가 있다.

---

## 9. ROI 7: 72K prefill/chunk autotune

### 관찰

72K server stress는 통과했다.

```text
4.4K prompt: prefill_tok_s 873.83
9.2K prompt: prefill_tok_s 880.68
동시 2요청도 통과
GPU process memory: 약 15820 MiB
```

### 위험

16GB에서 headroom이 매우 얇다. 72K slot을 2개로 늘리는 것은 금지한다.

### 실험 후보

```text
prefill_chunk_size 64 / 128 / 256
max_prefill_rows_per_batch 1 유지
prefix cache max tokens 조정
trim cache threshold 조정
```

### 성공 기준

```text
72K server OOM 없음
9K~16K prompt prefill 유지 또는 개선
동시 2요청 안정성 유지
```

---

## 10. ROI 8: paged attention / KV append tuning

72K 운영에서는 long-context attention/KV cost가 점점 커진다.

후보:

```text
- INT4_BDR tiled layout kernel occupancy 개선
- attention_append_paged_int4_spec 최적화
- recent_tokens/window policy 재검증
- KV block table update와 append 경로 결합
```

이 과제는 decode가 72K 근처에서 길어질 때 중요하다. 단기 champion bucket보다 long-context 운영 안정성 쪽 ROI가 높다.

---

## 11. 낮은 ROI / 하지 말 것

### GDN BA low-bit pair 기본 ON 금지

현재 결과:

```text
GDN_BA_LOWBIT_PAIR=1:
  aggregate_decode_tok_s: 55.15
  accepted/rejected: 315 / 670
```

속도와 trajectory가 모두 나빠진다.

### scalar full-streaming MLP 성능화 금지

현재 구조:

```text
scalar dequant + atomic accumulation + CTA spin-wait
```

debug/reference로는 의미 있지만 production 성능용 구조가 아니다.

### no-materialization single-kernel tensor-core MLP 집착 금지

SiLU nonlinear boundary 때문에 gate/up reduction 완료 후 down GEMM이 필요하다. full no-materialization single-kernel은 재계산 또는 global sync/scratch가 필요해 비용 모델이 나쁘다.

---

## 12. 작업 순서

권장 순서:

```text
1. MTP batch proposer legacy parity repair
2. Marlin internal argmax v2
3. speculative resolve/cache_update fused kernel
4. CUDA Graph micrograph화
5. GDN recurrent+norm+gate actual shape 대응
6. MLP fused hot path 정리
7. 72K prefill/chunk autotune
8. paged attention/KV append tuning
```

각 단계는 반드시 다음 방식으로 검증한다.

```text
- 한 번에 옵션 하나만 변경
- Q4 fused / K=5 / prompt=1 / max_new=512 champion bucket 측정
- accepted/rejected/scheduled 기록
- tok/s 최고값이 아니라 repeat 평균 기록
- 72K server smoke는 별도 실행
```

---

## 13. 최종 판단

가장 ROI가 높은 과제는 **MTP batch proposer parity repair**다.

이유:

```text
현재 batch proposer는 켜면 느려지지만, 원인은 overhead보다 trajectory 회귀다.
legacy와 bit-identical한 draft sequence를 만들면 batch proposer의 원래 목적이 살아난다.
특히 동시 요청/continuous batching에서 이득 가능성이 가장 크다.
```

두 번째는 **Marlin internal argmax v2**다. lm_head argmax는 매 step 반복되는 hot path라 제대로 만들면 작지만 확실한 이득 가능성이 있다.

세 번째는 **speculative commit/cache_update fused kernel**이다. full CUDA Graph보다 작은 후처리 path를 직접 통합하는 쪽이 더 현실적이다.
