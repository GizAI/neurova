# LangBurst Qwen3.6 Q4/MTP 성능 연구 운영 문서

작성일: 2026-06-18  
대상: `langburst` native runtime, Qwen3.6-27B Q4 Marlin fused, MTP/NEXTN speculative decode  
주요 원칙: **op 존재 / synthetic parity / CUDA build 성공 / production 성능 검증을 구분한다.**

---

## 1. 운영 결론 요약

현재 운영 기본값은 기존 champion 복구와 안정성을 우선한다.

```bash
LANGBURST_CUDA_GRAPH="${LANGBURST_CUDA_GRAPH:-0}"
LANGBURST_VERIFY_FULL_LOGITS="${LANGBURST_VERIFY_FULL_LOGITS:-0}"
LANGBURST_MARLIN_INTERNAL_ARGMAX="${LANGBURST_MARLIN_INTERNAL_ARGMAX:-0}"

LANGBURST_MTP_MAX_DRAFT="${LANGBURST_MTP_MAX_DRAFT:-5}"
LANGBURST_MTP_DRAFT_CANDIDATES="${LANGBURST_MTP_DRAFT_CANDIDATES:-5}"
LANGBURST_MTP_BATCH_PROPOSER="${LANGBURST_MTP_BATCH_PROPOSER:-0}"
LANGBURST_MTP_LOCAL_TKH_ATTENTION="${LANGBURST_MTP_LOCAL_TKH_ATTENTION:-0}"
LANGBURST_MTP_ADAPTIVE="${LANGBURST_MTP_ADAPTIVE:-0}"

LANGBURST_MLP_TENSORCORE_DOWN_SILU_A="${LANGBURST_MLP_TENSORCORE_DOWN_SILU_A:-1}"
LANGBURST_MLP_SCALAR_STREAMING_DEBUG="${LANGBURST_MLP_SCALAR_STREAMING_DEBUG:-0}"

LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED="${LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED:-0}"
LANGBURST_GDN_BA_LOWBIT_PAIR="${LANGBURST_GDN_BA_LOWBIT_PAIR:-0}"
```

운영 기본 ON 후보는 제한적이다.

- MTP K=5 고정
- MTP batch proposer ON
- verifier full logits OFF
- MTP local TKH attention OFF
- CUDA Graph static-buffer replay 구현 보존, 운영 기본 OFF

기본 OFF 후보는 개별 실험으로만 켠다.

- Marlin internal argmax
- MLP scalar full-streaming reference
- GDN recurrent+norm+gate fusion
- GDN BA low-bit pair

---

## 2. 빌드 운영 규칙

CUDA extension 빌드는 반드시 lock wrapper만 사용한다.

```bash
cd /home/user/workspace/neurova/langburst
./scripts/build_langburst_cuda.sh
```

금지:

```bash
rm -rf build langburst_cuda*.so
python setup.py build_ext --inplace
```

금지 이유:

- 여러 에이전트가 동시에 raw build를 돌리면 `build/`, `.ninja_lock`, object file, 최종 `.so`가 서로 깨진다.
- Python 3.11/3.12 ABI mismatch, missing object, missing op 문제가 발생했다.
- `build_langburst_cuda.sh`는 `/tmp/langburst_cuda_build.lock`과 isolated build dir를 사용한다.

---

## 3. 검증 레벨 정의

이 문서에서 쓰는 표현은 다음 기준을 따른다.

| 표현 | 의미 |
|---|---|
| source/code path added | 코드 경로가 추가됨. build/import나 correctness는 별도 확인 필요 |
| CUDA build/import OK | `langburst_cuda.so` 빌드 및 Python import/symbol 존재 확인됨 |
| synthetic parity OK | 작은 랜덤/합성 텐서에서 기준 경로와 수치 비교 통과 |
| model smoke OK | 실제 모델 로드/짧은 생성 완료 |
| production perf validated | 목표 모델/설정/토큰 bucket에서 tok/s, acceptance, scheduled batch가 기준 이상 |

절대 하지 말아야 할 것:

- op가 존재한다는 이유로 production default ON이라고 주장하지 않는다.
- synthetic parity 통과를 실제 Q4/MTP champion bucket 성능 검증으로 간주하지 않는다.
- GPU bench가 OOM/서버 점유로 실패한 경우 성공으로 말하지 않는다.

---

## 4. MTP/NEXTN speculative decode

### 현재 운영값

```bash
LANGBURST_MTP_MAX_DRAFT=5
LANGBURST_MTP_DRAFT_CANDIDATES=5
LANGBURST_MTP_BATCH_PROPOSER=0
LANGBURST_MTP_LOCAL_TKH_ATTENTION=0
LANGBURST_MTP_ADAPTIVE=0
```

### 연구 내용

1. 기존 champion은 K=5였다.
2. 중간 측정에서 K=4가 현재 회귀 상태에서 더 좋아 보인 적이 있으나, 목표는 기존 52 tok/s+ champion 회복이므로 최종 기본값은 K=5로 복구했다.
3. `LANGBURST_MTP_LOCAL_TKH_ATTENTION=1`은 K=5에서 느렸다.

관측값:

```text
K=5, batch proposer ON, TKH ON  : 47.60 tok/s, accepted 312, rejected 683
K=5, batch proposer ON, TKH OFF : 48.34 tok/s, accepted 312, rejected 683
```

따라서 TKH local attention은 기본 OFF.

### batch proposer

`LANGBURST_MTP_BATCH_PROPOSER=0`이 운영 기본이다. 테스트에서 batch proposer 의존 CPU 계약은 통과했다.

```bash
CUDA_VISIBLE_DEVICES= LANGBURST_MTP_BATCH_PROPOSER=0 python -m pytest -q \
  langburst/tests/test_model_runner_cpu.py \
  langburst/tests/test_speculative_batch_cpu.py \
  langburst/tests/test_speculative_cpu.py \
  langburst/tests/test_adapter_runtime_cpu.py
```

결과:

```text
59 passed
```

---

## 5. Verifier / lm_head argmax

### 현재 운영값

```bash
LANGBURST_VERIFY_FULL_LOGITS=0
LANGBURST_MARLIN_INTERNAL_ARGMAX=0
```

### 연구 내용

Verifier는 greedy speculative resolve에 full vocab logits가 아니라 target/bonus argmax id만 필요하다. 따라서 기본 경로는 full logits materialization을 피한다.

- `verify_lm_head_full`: debug/research path
- `verify_lm_head_argmax`: production path

`lowbit_marlin_gemm_argmax_out` / `LowBitMarlinTensor.gemm_argmax()`는 구현/빌드/import 되었지만, MTP acceptance/speed가 champion 기준으로 충분히 검증되기 전까지 기본 OFF로 둔다.

---

## 6. MLP 연구

### 6.1 Scalar full-streaming MLP reference

구현 op:

```text
lowbit_marlin_mlp_streaming_out
```

구조:

```text
x
→ gate_i scalar dequant/MMA-like accumulation
→ up_i scalar dequant/MMA-like accumulation
→ act_i = silu(gate_i) * up_i
→ down weight scalar dequant
→ atomicAdd output accumulation
```

특징:

- gate/up activation tensor를 materialize하지 않는다.
- 한 CUDA kernel 안에서 MLP 전체를 streaming 형태로 처리한다.
- 하지만 tensor-core Marlin이 아니라 scalar dequant + atomic accumulation이다.
- CTA 간 spin-wait 구조가 있어 production 기본으로 위험하다.

운영값:

```bash
LANGBURST_MLP_SCALAR_STREAMING_DEBUG=0
```

상태:

- synthetic Marlin parity OK
- production 성능용 아님
- debug/reference 전용

### 6.2 Tensor-core Marlin down-side SiLU fusion

구현 op:

```text
lowbit_marlin_gemm_silu_packed_out
LowBitMarlinTensor.gemm_silu_packed()
Qwen36MLP._fused_marlin_down()
```

구조:

```text
gate_up Marlin GEMM → mixed [gate, up]
down Marlin GEMM에서 A operand 로드 시 silu(gate) * up 즉석 계산
```

제거되는 것:

- `silu_mul_packed` 별도 kernel launch
- activation `[B, I]` materialization

유지되는 것:

- mixed `[B, 2I]` materialization

운영값:

```bash
LANGBURST_MLP_TENSORCORE_DOWN_SILU_A=1
```

상태:

- synthetic parity OK
- 실제 Q4/MTP champion bucket에서는 이득이 없거나 느렸다.
- default ON 금지.

---

## 7. GDN 연구

### 7.1 GDN recurrent + norm_gate fused op

구현 op:

```text
gdn_recurrent_ab_batch_norm_gate
```

의도:

```text
GDN recurrent update
→ RMSNorm
→ SiLU gate
```

을 한 CUDA op로 합치기.

상태:

- CUDA op build/import 확인됨
- 랜덤 parity 일부 OK
- 실제 Q4 모델에서 `norm_w hidden mismatch` 발생
- shape guard 추가
- 운영 기본 OFF

운영값:

```bash
LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED=0
```

### 7.2 GDN BA low-bit pair

중요한 모델 구조 관찰:

```text
q4-marlin-fused checkpoint:
in_proj_qkvz = marlin_v1
in_proj_ba   = fp16
split in_proj_b/a = low-bit, not Marlin
```

따라서 현재 checkpoint에서 “in_qkvz + in_ba true dual-Marlin”은 성립하지 않는다. `in_proj_ba`가 Marlin이 아니라 fp16이기 때문이다.

추가한 옵션:

```bash
LANGBURST_GDN_BA_LOWBIT_PAIR=0
```

의미:

- `in_proj_ba` fp16 대신 split `in_proj_b` / `in_proj_a` low-bit pair를 사용할 수 있게 한 실험 옵션.
- production 기본은 OFF.
- 실제 성능/품질 검증 전까지 ON 금지.

---

## 8. CUDA Graph static-buffer replay

### 문제였던 이전 구조

이전 graph wrapper는 다음 문제가 있었다.

```text
- closure가 현재 plan/states/tensor를 붙잡음
- graph key에 dynamic tensor data_ptr 포함
- replay가 static input/output buffer 기반이 아님
- 잘못하면 이전 token/state를 재사용할 위험
```

### 새 구조

추가 구조:

```text
_VerifyGraphStaticBuffers
```

포함하는 static tensors:

```text
token_matrix
positions
query_start_loc
seq_lens
logits_indices
cu_num_logits
state_indices
slot_mapping
block_tables
row_lengths
draft_token_ids
zero_commit
metadata
static_plan
results
```

실행 흐름:

```text
current DecodeBatchPlan
→ _build_uniform_spec_plan()
→ _copy_verify_graph_inputs(): current 값들을 static buffers에 copy_
→ captured CUDA graph replay
→ buffers.results를 current states로 materialize
```

핵심 함수:

```text
_VerifyGraphStaticBuffers
_arena_batch_for_plan()
_new_verify_graph_buffers()
_copy_verify_graph_inputs()
_forward_verify_batch_uniform_hot_graph()
_forward_verify_batch_uniform_hot_core()
```

graph key 구성:

```text
arena id
batch size
query length
speculative token count
context bucket
verify_full_logits flag
marlin_internal_argmax flag
draft_counts
```

제거한 위험:

- dynamic `data_ptr()` 기반 graph key
- 현재 plan tensor를 closure로 직접 잡는 구조
- graph replay 시 이전 token/state tensor를 재사용할 가능성

운영값:

```bash
LANGBURST_CUDA_GRAPH=0
```

상태:

- 구조 구현 완료
- local CPU regression OK
- remote CPU regression OK
- GPU smoke/perf는 서버/GPU 점유로 완주 검증 못 함

주의:

- graph ON 서버 적용은 서버 재시작이 필요하다.
- GPU 독점 상태에서 `CUDA_GRAPH=0` vs `CUDA_GRAPH=1` 비교가 아직 필요하다.

---

## 9. KV / attention 운영값

권장 운영값:

```bash
LANGBURST_KV_CACHE_DTYPE=int4_bdr
LANGBURST_PAGED_KV=1
LANGBURST_PAGED_KV_SHADOW=1
LANGBURST_PAGED_ATTENTION_KERNELS=1
LANGBURST_PAGED_ATTENTION_BACKEND=auto
LANGBURST_INT4_KV_LAYOUT=tiled
LANGBURST_ATTENTION_RECENT_TOKENS=128
```

주의:

- `LANGBURST_ATTENTION_RECENT_TOKENS`는 서버 운영에서 더 큰 값이 쓰일 수 있다.
- 현재 start 로그에는 `recent-window 28672` 같은 운영값이 별도 CLI로 들어간다.

---

## 10. 기능/옵션 상태표

| 옵션/기능 | 기본 | 상태 | 운영 판단 |
|---|---:|---|---|
| `LANGBURST_CUDA_GRAPH` | `0` | static-buffer replay 구현됨, champion bucket 느림 | OFF |
| `LANGBURST_MTP_MAX_DRAFT` | `5` | 기존 champion 기준 | ON |
| `LANGBURST_MTP_DRAFT_CANDIDATES` | `5` | K=5 고정 | ON |
| `LANGBURST_MTP_BATCH_PROPOSER` | `0` | trajectory/speed 회귀 | OFF |
| `LANGBURST_MTP_LOCAL_TKH_ATTENTION` | `0` | K=5에서 느림 | OFF |
| `LANGBURST_VERIFY_FULL_LOGITS` | `0` | full logits 제거 | OFF 유지 |
| `LANGBURST_MARLIN_INTERNAL_ARGMAX` | `0` | op 존재, acceptance 확인 필요 | OFF |
| `LANGBURST_MLP_TENSORCORE_DOWN_SILU_A` | `1` | champion bucket 64.53~65.11 tok/s | ON |
| `LANGBURST_MLP_SCALAR_STREAMING_DEBUG` | `0` | reference/debug | OFF |
| `LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED` | `0` | shape mismatch | OFF |
| `LANGBURST_GDN_BA_LOWBIT_PAIR` | `0` | 실험 후보 | OFF |

---

## 11. 실험 순서

GPU가 완전히 빈 상태에서만 실행한다. 서버가 떠 있으면 bench가 OOM 나므로 먼저 확인한다.

```bash
ssh ml-dmc8 'ps -eo pid,etime,cmd | grep -E "langburst.server|bench_serving" | grep -v grep || true'
ssh ml-dmc8 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true'
```

권장 실험 순서:

```text
1. baseline
   CUDA_GRAPH=0, K=5, TKH=0, BATCH_PROPOSER=1

2. graph
   CUDA_GRAPH=1만 변경, 기본보다 느리면 유지하지 않음

3. MLP fused
   MLP_TENSORCORE_DOWN_SILU_A=1 기본값 유지 검증

4. GDN BA pair
   GDN_BA_LOWBIT_PAIR=1만 변경

5. Marlin argmax
   MARLIN_INTERNAL_ARGMAX=1만 변경
```

성공 기준:

```text
- aggregate_decode_tok_s 증가
- accepted/rejected/scheduled batch가 champion보다 악화되지 않음
- API smoke 정상
- OOM 없음
- repeated run에서 재현성 있음
```

---

## 12. 현재 남은 검증 TODO

1. GPU 독점 상태에서 CUDA Graph smoke
2. CUDA Graph ON/OFF throughput 비교
3. K=5 champion bucket 재현
4. GDN_BA_LOWBIT_PAIR=1 개별 벤치
5. MLP_TENSORCORE_DOWN_SILU_A=1 개별 벤치
6. MARLIN_INTERNAL_ARGMAX=1 acceptance/speed 재검증
7. 서버 재시작 후 `/proc/$PID/environ`로 start env 실제 반영 확인

---

## 13. 운영 체크리스트

서버 시작 전:

```bash
cd /home/user/workspace/neurova/langburst
./scripts/build_langburst_cuda.sh
python -m compileall -q langburst
python -m pytest -q tests/test_model_runner_cpu.py tests/test_speculative_batch_cpu.py tests/test_speculative_cpu.py tests/test_adapter_runtime_cpu.py
```

서버 시작:

```bash
./scripts/start_langburst.sh
```

환경 확인:

```bash
pid=$(cat /tmp/langburst_server.pid)
tr '\0' '\n' < /proc/$pid/environ | grep -E 'LANGBURST_CUDA_GRAPH|LANGBURST_MTP_MAX_DRAFT|LANGBURST_MTP_BATCH_PROPOSER|LANGBURST_MTP_LOCAL_TKH_ATTENTION|LANGBURST_VERIFY_FULL_LOGITS|LANGBURST_MARLIN_INTERNAL_ARGMAX'
```

API smoke:

```bash
curl -sS http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"langburst-qwen3.6-27b-q4","messages":[{"role":"user","content":"간단히 서울의 장점을 두 문장으로 말해줘."}],"max_tokens":64,"stream":false}'
```

---

## 14. 최종 원칙

```text
synthetic parity / op import 성공 ≠ production default ON
```

production default ON 조건은 다음 모두를 만족해야 한다.

```text
1. 실제 Qwen3.6-27B Q4 fused 모델에서 smoke OK
2. target bucket에서 tok/s 개선
3. acceptance/rejection/scheduled batch 악화 없음
4. OOM 없음
5. repeated run 재현
```

이 조건을 만족하지 않은 기능은 env-gated OFF로 유지한다.


---

## 15. 2026-06-18 최종 A/B 반영

최종 재측정에서 이전 중간 결론이 일부 뒤집혔다. 운영 기본은 “기능을 많이 켜는 조합”이 아니라, champion trajectory를 유지하는 최소 조합이다.

조건:

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

측정 결과:

```text
baseline:
  CUDA_GRAPH=0
  MTP_BATCH_PROPOSER=0
  MLP_TENSORCORE_DOWN_SILU_A=0
  MARLIN_INTERNAL_ARGMAX=0
  aggregate_decode_tok_s: 62.76

CUDA_GRAPH=1:
  aggregate_decode_tok_s: 61.69
  결론: static-buffer graph 구조는 보존하되 운영 기본 OFF

MTP_BATCH_PROPOSER=1:
  aggregate_decode_tok_s: 58.99
  accepted/rejected: 312 / 683
  scheduled_batches: 200
  결론: trajectory 회귀가 있어 기본 OFF

MLP_TENSORCORE_DOWN_SILU_A=1:
  aggregate_decode_tok_s: 64.53, repeat 65.11
  accepted/rejected: 324 / 616
  scheduled_batches: 189
  결론: speed-positive이며 champion trajectory 유지, 기본 ON

GDN_BA_LOWBIT_PAIR=1:
  aggregate_decode_tok_s: 55.15
  accepted/rejected: 315 / 670
  결론: 기본 OFF

MARLIN_INTERNAL_ARGMAX=1:
  aggregate_decode_tok_s: 62.26~63.52
  MLP=1과 같이 켜도 64.14로 MLP 단독보다 낮음
  결론: 기본 OFF

MLP_TENSORCORE_DOWN_SILU_A=1 + CUDA_GRAPH=1:
  aggregate_decode_tok_s: 64.21
  결론: MLP 단독보다 낮아 graph 기본 OFF
```

최종 운영 default:

```bash
LANGBURST_CUDA_GRAPH="${LANGBURST_CUDA_GRAPH:-0}"
LANGBURST_MTP_BATCH_PROPOSER="${LANGBURST_MTP_BATCH_PROPOSER:-0}"
LANGBURST_MLP_TENSORCORE_DOWN_SILU_A="${LANGBURST_MLP_TENSORCORE_DOWN_SILU_A:-1}"
LANGBURST_MARLIN_INTERNAL_ARGMAX="${LANGBURST_MARLIN_INTERNAL_ARGMAX:-0}"
LANGBURST_GDN_BA_LOWBIT_PAIR="${LANGBURST_GDN_BA_LOWBIT_PAIR:-0}"
LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED="${LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED:-0}"
```

72K server stress:

```text
CONTEXT_TIERS=4096,73728
CONTEXT_WINDOW=73728
KV=int4_bdr
MTP K=5
```

통과 항목:

```text
짧은 순차 요청 5회 OK
4.4K prompt: prefill_tok_s 873.83, decode_tok_s 23.79
동시 2요청: 2.6K prompt 각 OK, prefill_tok_s 867.39 / 426.85
9.2K prompt: prefill_tok_s 880.68, decode_tok_s 26.01
GPU process memory: 약 15820 MiB
```

운영 판단:

```text
72K는 16GB에서 빡빡하지만 실제 API 경로에서 통과했다.
80K는 OOM이므로 72K가 현재 최대 운영 후보이다.
72K slot을 2개 이상으로 늘리는 것은 아직 금지한다.
```
