# 2026-06-18 LangBurst Q4/Q3, MTP, VRAM 세션 기록

이 문서는 `ml-dmc8` RTX 4080 16GB에서 진행한 LangBurst Qwen3.6-27B 성능/VRAM 조사와 운영 결론을 정리한다. 기준 런타임은 conda `langburst` 환경의 Python 3.11.15이며, CUDA extension ABI도 이 환경에 맞아야 한다.

## 운영 결론

```text
decode 속도 champion:
  /home/user/models/Qwen3.6-27B-qb4-marlin-fused

현재 start script 기본:
  Qwen3.6-27B-qb4-marlin-fused
  model name: langburst-qwen3.6-27b-q4

Q3의 역할:
  VRAM headroom이 아주 빡빡한 경우의 fallback 후보.
  하지만 현재 측정에서는 Q4 fused보다 decode가 느리고 VRAM 절감도 작다.
```

## 2026-06-18 최종 정정: 현재 운영 champion

이번 세션 후반에 같은 ml-dmc8 / conda `langburst` / Q4 fused / `int4_bdr` / MTP K=5 조건으로 다시 분리 측정했다. 최종 운영 결론은 다음이다.

```text
start script 기본:
  QB_DIR=/home/user/models/Qwen3.6-27B-qb4-marlin-fused
  MODEL_NAME=langburst-qwen3.6-27b-q4
  KV_CACHE_DTYPE=int4_bdr
  CONTEXT_TIERS=4096,73728
  CONTEXT_TIER_SLOTS=1,1
  CONTEXT_WINDOW=73728
  LANGBURST_MARLIN_OUT_CACHE_POLICY=decode_only
  LANGBURST_MTP_MAX_DRAFT=5
  LANGBURST_MTP_DRAFT_CANDIDATES=5
  LANGBURST_MTP_LEGACY_LIST_CACHE=1
  LANGBURST_MTP_BATCH_PROPOSER=0
  LANGBURST_MTP_LOCAL_TKH_ATTENTION=0
  LANGBURST_MARLIN_INTERNAL_ARGMAX=0
  LANGBURST_CUDA_GRAPH=0
  LANGBURST_MLP_TENSORCORE_DOWN_SILU_A=1
  LANGBURST_MLP_SCALAR_STREAMING_DEBUG=0
  LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED=0
```

`LANGBURST_MARLIN_OUT_CACHE_POLICY=all`은 더 이상 운영 기본이 아니다. 같은 조건에서 `decode_only`가 decode와 short/medium prefill 모두 더 좋았다.

### Decode 재측정

조건:

```text
Q4 fused
KV=int4_bdr
MTP K=5 fixed
recent_window=2048
prompt_tokens=1
max_new_tokens=512
requests=1
kv_blocks=512
prefill_chunk_size=64
max_num_batched_tokens=256
```

결과:

```text
cache=all:
  aggregate_decode_tok_s: 50.66~50.72
  accepted/rejected: 324 / 616
  scheduled_batches: 189

cache=off:
  aggregate_decode_tok_s: 59.09
  accepted/rejected: 312 / 683
  scheduled_batches: 200

cache=decode_only:
  aggregate_decode_tok_s: 62.58~62.64
  accepted/rejected: 324 / 616
  scheduled_batches: 189
```

해석:

```text
decode champion은 cache=decode_only.
MTP candidate trajectory도 기존 champion과 같은 324/616/189를 유지한다.
과거 52 tok/s보다 낮아진 것이 아니라, 잘못된 cache=all 기본값 때문에 성능을 잃고 있었다.
```

256 token decode 참고:

```text
cache=all + MTP on:         54.35 tok/s
cache=decode_only + MTP on: 56.78 tok/s
cache=off + MTP on:         52.97 tok/s
cache=all + MTP off:        19.24 tok/s
```

MTP off는 19 tok/s 수준으로 크게 느리므로 현재 Q4 운영 default는 MTP K=5 on이 맞다.

### Prefill 재측정

조건:

```text
Q4 fused
KV=int4_bdr
requests=1
max_new_tokens=1
prefill_chunk_size=256
max_num_batched_tokens=256
MTP off 중심 측정
```

결과:

```text
prompt_tokens=1024:
  cache=all:         46.88 tok/s
  cache=off:        863.81 tok/s
  cache=decode_only: 866.81 tok/s

prompt_tokens=2048:
  cache=off:        929.35 tok/s
  cache=decode_only: 926.31 tok/s

prompt_tokens=4984:
  cache=off, context 8192:       46.26 tok/s
  cache=decode_only, context 8192: 46.29 tok/s
  cache=off, context 2048:       57.45 tok/s
```

해석:

```text
1K~2K fresh prefill은 860~930 tok/s로 회복됐다.
cache=all은 prefill을 46 tok/s 수준으로 망가뜨리므로 운영 금지.
4.9K prompt부터는 현재 long-prompt path가 느린 경로로 빠진다.
이건 아직 남은 prefill regression이며, true long-prompt block prefill / INT4_BDR tiled attention hot path 쪽 과제로 남긴다.
```

### Context window 재측정

실제 OpenAI API 서버 경로로 `CONTEXT_TIERS=4096,X`, `CONTEXT_TIER_SLOTS=1,1`, Q4 fused, `decode_only`, MTP K=5 상태에서 첫 요청 성공 여부를 측정했다.

```text
X=16384: OK, GPU process 약 14864 MiB
X=20480: OK, GPU process 약 14946 MiB
X=24576: OK, GPU process 약 15008 MiB
X=28672: OK, GPU process 약 15070 MiB
X=32768: OK, GPU process 약 15134 MiB
X=40960: OK, GPU process 약 15278 MiB
X=49152: OK, GPU process 약 15402 MiB
X=57344: OK, GPU process 약 15526 MiB
X=65536: OK, GPU process 약 15670 MiB
X=73728: OK, GPU process 약 15800 MiB
X=81920: OOM, 20 MiB allocation 실패
```

2026-06-18 추가 stress 결과로 운영 기본은 `4096,73728`까지 올린다.

이유:

```text
73728은 단일 첫 요청뿐 아니라 반복 짧은 요청, 동시 2요청, 9K prompt까지 통과했다.
81920은 실제 20 MiB 추가 할당에서 OOM이 난다.
73728은 81920 실패 전의 최대 운영 후보이며, 기존 49152보다 크다.
```

OOM 원인 분해:

```text
Q4 fused weight + MTP proposer + CUDA extension/private pools + state arena + prefix/cache/scratch가 이미 15GiB 중후반까지 점유한다.
INT4_BDR KV는 KV 자체를 크게 줄였지만, weight 13GiB대와 MTP 약 288MiB, runtime scratch/cache는 그대로 남는다.
따라서 context만 무한히 올릴 수는 없고, 80K 근처에서는 20MiB 단위 할당도 실패한다.
```

남은 VRAM 최적화 후보:

```text
1. MTP proposer weight를 필요 시 lazy-load/unload 또는 lower-bit화
2. prefix cache eviction을 free-block/free-MiB 기준으로 더 공격적으로 적용
3. long-prompt prefill scratch를 cache=decode_only와 충돌하지 않게 고정 pool화
4. lm_head/proposer temporary logits/output buffer 재사용 강화
5. 4.9K 이상 prompt가 느린 경로로 빠지는 원인을 제거해 긴 입력에서 불필요한 staging을 줄이기
```

### 72K stress 결과

서버 조건:

```text
CONTEXT_TIERS=4096,73728
CONTEXT_TIER_SLOTS=1,1
CONTEXT_WINDOW=73728
Q4 fused
KV=int4_bdr
Marlin cache=decode_only
MTP K=5
```

결과:

```text
짧은 순차 요청 5회:
  모두 OK
  2회차 이후 prefill 약 419~426 tok/s
  decode 약 30 tok/s

4.4K prompt:
  prompt_tokens: 4423
  prefill_tok_s: 873.83
  decode_tok_s: 23.79

동시 2요청:
  request A prompt_tokens: 2660, prefill_tok_s: 867.39, decode_tok_s: 22.30
  request B prompt_tokens: 2660, prefill_tok_s: 426.85, decode_tok_s: 23.93
  둘 다 OK

9.2K prompt:
  prompt_tokens: 9260
  prefill_tok_s: 880.68
  decode_tok_s: 26.01
  finish_reason: length

GPU process memory:
  약 15820 MiB
```

해석:

```text
72K tier는 16GB에서 매우 빡빡하지만 실제 API 경로에서 버틴다.
80K는 OOM이므로 72K가 현재 "올릴 수 있는 만큼"의 기본값이다.
긴 prompt prefill이 항상 46 tok/s로 떨어지는 것은 아니며, 서버 경로에서는 9K prompt도 880 tok/s를 기록했다.
이전 4.9K slow result는 bench path / 설정 조합의 별도 regression으로 봐야 한다.
```

### 옵션 A/B 재검증

조건:

```text
Q4 fused
recent_window=2048
prompt_tokens=1
max_new_tokens=512
MTP K=5
Marlin cache=decode_only
```

결과:

```text
baseline:
  CUDA_GRAPH=0
  MTP_BATCH_PROPOSER=0
  MLP_TENSORCORE_DOWN_SILU_A=0
  MARLIN_INTERNAL_ARGMAX=0
  aggregate_decode_tok_s: 62.76

CUDA_GRAPH=1:
  aggregate_decode_tok_s: 61.69
  결론: 구현은 있으나 현재 champion bucket에서는 기본 ON 금지

MTP_BATCH_PROPOSER=1:
  aggregate_decode_tok_s: 58.99
  accepted/rejected: 312 / 683
  scheduled_batches: 200
  결론: trajectory가 나빠져 기본 ON 금지

MLP_TENSORCORE_DOWN_SILU_A=1:
  aggregate_decode_tok_s: 64.53, repeat 65.11
  accepted/rejected: 324 / 616
  scheduled_batches: 189
  결론: 현재 조건에서는 speed-positive, 기본 ON

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

최종 운영 옵션:

```text
LANGBURST_CUDA_GRAPH=0
LANGBURST_MTP_BATCH_PROPOSER=0
LANGBURST_MLP_TENSORCORE_DOWN_SILU_A=1
LANGBURST_MARLIN_INTERNAL_ARGMAX=0
LANGBURST_GDN_BA_LOWBIT_PAIR=0
LANGBURST_GDN_RECURRENT_NORM_GATE_FUSED=0
```

Q3가 느린 핵심 원인은 “3bit라서 항상 빠르다”가 아니라, 현재 Q3 checkpoint가 순수 Q3가 아닌 Q3/Q4 hybrid라는 점이다. hot projection 261개는 Q4 Marlin으로 동일하고, Q3는 일부 rowwise tensor만 줄인다. 반면 MTP acceptance는 Q4 fused보다 낮아져 reject와 scheduled batch가 늘었다.

## 재현 환경

반드시 다음 환경을 사용한다.

```bash
cd /home/user/workspace/neurova/langburst
source ~/miniconda3/etc/profile.d/conda.sh
conda activate langburst
source ./scripts/langburst_cuda_env.sh
python -V
```

기대값:

```text
/home/user/miniconda3/envs/langburst/bin/python
Python 3.11.15
```

금지:

```text
python3.12
```

`python3.12`로 `langburst_cuda.so`를 빌드하면 Python 3.11 런타임에서 ABI mismatch가 난다.

정상 확인:

```bash
LANGBURST_REQUIRE_CUDA_EXT=1 python - <<'PY'
from langburst.ops import cuda_ops
ops = cuda_ops()
print(hasattr(ops, "lowbit_marlin_gemm_argmax_out"))
PY
```

빌드가 필요하면 이 명령만 사용한다.

```bash
rm -rf build langburst_cuda*.so
LANGBURST_REQUIRE_CUDA_EXT=1 MAX_JOBS=1 python setup.py build_ext --inplace
```

## Q4 fused vs Q3 내부 차이

디스크 index 기준:

```text
Qwen3.6-27B-qb4-marlin-fused:
  lowbit_marlin_groupwise: 261개
  lowbit_symmetric_groupwise: 97개
  fp16_raw: 742개
  referenced_total: 14.098 GiB
    marlin_q4: 12.471 GiB
    fp16:       1.005 GiB
    q4:         0.622 GiB

Qwen3.6-27B-langburst-q3:
  lowbit_marlin_groupwise: 261개
  lowbit_symmetric_groupwise: 49개
  fp16_raw: 694개
  referenced_total: 13.903 GiB
    marlin_q4: 12.471 GiB
    fp16:       0.961 GiB
    q3:         0.471 GiB
```

중요한 점:

```text
양쪽 모두 marlin_q4가 12.471 GiB로 동일하다.
Q3가 줄이는 것은 일부 non-hot rowwise tensor와 embedding 쪽이다.
그래서 전체 VRAM 절감은 작다.
```

## 실제 로드 VRAM

모델 weight만 로드:

```text
Q4 fused:
  free after load: 2199 MiB
  allocated:       13367 MiB
  reserved:        13460 MiB

Q3:
  free after load: 2399 MiB
  allocated:       13167 MiB
  reserved:        13260 MiB

차이:
  약 200 MiB
```

MTP proposer 추가:

```text
Q4 fused:
  model only delta: 13496 MiB
  with MTP delta:  13784 MiB
  MTP extra:         288 MiB

Q3:
  model only delta: 13296 MiB
  with MTP delta:  13584 MiB
  MTP extra:         288 MiB
```

MTP 포함 tensor breakdown:

```text
Q4 fused:
  lowbit_marlin_groupwise:   12770.74 MiB
  lowbit_symmetric_groupwise:  636.80 MiB
  fp16_raw:                   150.10 MiB

  text_layers:   12024.14 MiB
  text_embedding:  625.20 MiB
  lm_head:         625.20 MiB
  native_mtp:      283.10 MiB

Q3:
  lowbit_marlin_groupwise:   12770.74 MiB
  lowbit_symmetric_groupwise:  482.42 MiB
  fp16_raw:                   105.10 MiB

  text_layers:   11976.33 MiB
  text_embedding:  473.63 MiB
  lm_head:         625.20 MiB
  native_mtp:      283.10 MiB
```

2-slot / 32K / `int4_bdr` state arena:

```text
total: 1205.62 MiB
  paged_kv:       1056.00 MiB
  gdn_recurrent:   144.00 MiB
  gdn_conv:          5.62 MiB
```

KV dtype 비교, 같은 2-slot / 32K 조건:

```text
fp16:     4245.62 MiB
fp8_e4m3: 2197.62 MiB
int4:     1205.62 MiB
int4_bdr: 1205.62 MiB
```

MTP + 2-slot / 32K / `int4_bdr` arena 포함 총량:

```text
Q4 fused:
  total delta: 15004 MiB
  free:          691 MiB
  allocated:   14858 MiB
  reserved:    14968 MiB

Q3:
  total delta: 14806 MiB
  free:          889 MiB
  allocated:   14658 MiB
  reserved:    14770 MiB
```

## 성능 재현

Q4 fused 과거 champion bucket:

```text
HF=/home/user/models/Qwen3.6-27B
QB=/home/user/models/Qwen3.6-27B-qb4-marlin-fused

recent_window=2048
prompt_tokens=1
max_new_tokens=512
requests=1
kv_blocks=512
max_num_batched_tokens=256
prefill_chunk_size=64
MTP K=5
KV=int4_bdr
Marlin output cache=all
```

실측 당시 값:

```text
aggregate_decode_tok_s: 52.39136605927192
decode_s: 9.772602596785873
accepted_prediction_tokens: 324
rejected_prediction_tokens: 616
scheduled_batches: 189
```

이 값은 이번 세션 후반 재측정으로 superseded 되었다. 현재 같은 bucket의 최종 champion은 위쪽 “최종 정정” 섹션의 `Marlin output cache=decode_only` 결과다.

```text
aggregate_decode_tok_s: 62.58~62.64
accepted_prediction_tokens: 324
rejected_prediction_tokens: 616
scheduled_batches: 189
```

같은 조건에서 `QB=/home/user/models/Qwen3.6-27B-langburst-q3`만 바꾼 결과:

```text
aggregate_decode_tok_s: 44.069577960488445
decode_s: 11.617991904960945
accepted_prediction_tokens: 285
rejected_prediction_tokens: 855
scheduled_batches: 229
```

해석:

```text
Q3는 weight VRAM을 약 200 MiB 줄이지만,
MTP acceptance가 나빠져 reject/scheduled batch가 증가한다.
현재 조건에서는 Q4 fused가 속도와 운영 안정성의 champion이다.
```

## 이번 세션에서 확인한 중요 사항

### 1. 51~52 tok/s 재현 실패의 원인

재현 실패의 대부분은 런타임 조건 불일치였다.

```text
1. Python 3.12로 실행 또는 extension 빌드
2. stale langburst_cuda.so ABI mismatch
3. 기존 langburst.server 또는 다른 bench process가 VRAM 점유
4. Q4 fused champion 조건이 아니라 Q3/32K/server 조건으로 비교
```

Q4 fused champion 숫자는 “Q4 fused + small recent_window benchmark bucket”의 결과다. Q3 32K server 조건에서 같은 tok/s가 나오는 숫자가 아니다.

### 2. Q3는 당장 속도 champion이 아니다

Q3가 더 빠르려면 Q3 hot path가 실제로 빠른 kernel을 타야 한다. 현재는 hot projection이 여전히 Q4 Marlin이고, Q3가 줄이는 부분은 전체 VRAM/latency에서 작은 비중이다.

### 3. INT4_BDR KV는 효과가 크다

32K 2-slot 기준:

```text
fp16 KV arena:     4245.62 MiB
int4_bdr KV arena: 1205.62 MiB
절감:              약 3040 MiB
```

Q3 weight 절감보다 KV dtype 절감 효과가 훨씬 크다.

### 4. MTP K=5가 현재 champion

이번 세션 기준 운영 default는 다음이다.

```text
LANGBURST_MTP_MAX_DRAFT=5
LANGBURST_MTP_DRAFT_CANDIDATES=5
LANGBURST_MTP_ADAPTIVE=0
LANGBURST_MTP_MIN_FREE_VRAM_MIB=128
```

K=6/7은 reject 증가와 overhead 때문에 K=5보다 느렸다.

### 5. 남은 큰 성능 과제

현재 55 tok/s 이상을 확실히 넘기려면 단순 wrapper fusion보다 더 깊은 CUDA 작업이 필요하다.

```text
1. Marlin lm_head 내부에서 vocab tile max/index reduction까지 수행
2. MTP proposer/verifier projection fusion
3. target verifier의 GDN/MLP projection deep fusion
4. uniform batch verifier의 state trajectory/commit path 추가 최적화
5. long-past int4_bdr fused prefill attention kernel
```

## 현재 start script 기본값

`scripts/start_langburst.sh`는 Q4 fused를 기본으로 사용한다.

```text
QB_DIR=/home/user/models/Qwen3.6-27B-qb4-marlin-fused
MODEL_NAME=langburst-qwen3.6-27b-q4
KV_CACHE_DTYPE=int4_bdr
CONTEXT_TIERS=4096,73728
CONTEXT_TIER_SLOTS=1,1
CONTEXT_WINDOW=73728
MAX_ACTIVE_REQUESTS=2
MAX_PREFILL_ROWS_PER_BATCH=1
MARLIN_OUT_CACHE_POLICY=decode_only
PREFIX_CACHE=on
MTP enabled, K=5 fixed
```

Q4 fused에서 MTP / Marlin cache all 조합은 16GB RTX 4080에서 prefill과 OpenWebUI 요청 안정성을 크게 해친다. OpenWebUI는 첫 메시지 뒤 제목/요약 등 추가 요청을 만들 수 있으므로 운영 기본은 `decode_only` cache와 prefill row 1개로 둔다. 실제 재측정 결과 `decode_only`는 decode 512 bucket에서도 `all`보다 빨랐다.

2명 동시 stress 결과:

```text
Marlin cache all:
  20K / 24K 모두 OOM

Marlin cache decode_only:
  20K: 30.96 aggregate decode tok/s
  24K: 34.11 aggregate decode tok/s
  28K: 33.14 aggregate decode tok/s
  32K: 24.08 aggregate decode tok/s, MTP accepted 0

Marlin cache off:
  20K: 32.65 aggregate decode tok/s
  24K: 34.27 aggregate decode tok/s
  28K: 30.91 aggregate decode tok/s
  32K: 23.86 aggregate decode tok/s, MTP accepted 0
```

해석:

```text
이 중간 stress 결과는 `decode_only` 최종 정정 이전의 값이다.
최종 API load test에서는 4K+64K tier가 통과했고, 4K+72K는 너무 빡빡하며, 4K+80K는 OOM으로 실패했다.
따라서 현재 운영 기본은 4K+64K tier다.
```

1명 단독 / 28K / `decode_only` / prompt 512 / max_new 256:

```text
aggregate_decode_tok_s: 40.5911
aggregate_output_tok_s: 35.4561
mean_ttft_s: 0.9132
prefill_tok_s: 561.4492
accepted_prediction_tokens: 145
rejected_prediction_tokens: 405
scheduled_batches: 118
```

해석:

```text
이 값은 `decode_only` 최종 정정 이전의 28K 운영 프로파일 값이다.
최종 decode champion은 recent_window=2048 / prompt_tokens=1 / max_new_tokens=512 / cache=decode_only에서 62.58~62.64 tok/s다.
OpenWebUI 운영값은 실제 prompt 길이, max token, context tier, title/summary background request에 따라 달라지므로 benchmark champion과 분리해 해석한다.
```

## 가변 context slot 연구

요구:

```text
2명 동시 기준에서 한 요청은 4K 정도의 작은 context,
다른 요청은 가능한 큰 context를 쓰게 하고 싶다.
작은 입력은 작은 쪽으로, 큰 입력은 큰 쪽으로 보내고 싶다.
```

현재 코드 구조:

```text
DecodeStateArena:
  num_slots와 max_seq_len은 engine 단위로 고정.
  GDN recurrent/conv state는 slot 수에만 비례하고 context 길이에 거의 무관.
  paged KV mirror가 꺼져 있으면 max_seq_len 자체가 dense KV를 만들지는 않는다.

KVBlockTable:
  요청별로 필요한 KV block만 동적으로 allocate/release.
  prefix cache도 block refcount 기반으로 붙어 있다.
```

즉 KV memory 관점에서는 이미 “가변 사용량”에 가깝다. 하지만 engine의 `recent_window/max_seq_len`이 하나라서, 지금은 모든 request가 같은 최대 context cap을 가진다. 그러므로 4K/큰 context 혼합을 운영 기본으로 만들려면 slot 자체를 나누기보다 **block-budget admission**이 정답이다.

권장 최종 구조:

```text
LANGBURST_CONTEXT_TIERS=4096,73728
LANGBURST_CONTEXT_TIER_SLOTS=1,1

engine recent_window:
  max(tiers) = 73728

KV blocks:
  sum(ceil(tier / block_size) * slots)
  4K + 72K라면 76K tokens worth of blocks만 preallocate.
  같은 큰 window 2개를 균등 제공하는 것보다 작은 요청용 slot과 긴 요청용 slot을 나누는 편이 낫다.

admission:
  prompt+generation budget이 4K 이하이면 small tier 우선.
  4K 초과이면 large tier 필요.
  large tier가 사용 중이면 queue에서 대기.
  두 개의 large request를 동시에 받아 KV block exhaustion으로 죽이지 않는다.
```

예상 KV arena, 2-slot equivalent:

```text
4K + 72K tier:
  total logical KV budget = 76K tokens
  16GB Q4 fused + decode_only 조건에서 실제 API stress 통과.

4K + 72K tier:
  첫 요청은 성공하지만 GPU process 약 15800 MiB로 headroom이 너무 작음.

4K + 80K tier:
  20 MiB allocation에서 OOM.
```

따라서 가변 tier를 제대로 구현하면:

```text
작은 요청 하나 + 큰 요청 하나를 동시에 처리할 수 있다.
현재 28K 균등 슬롯보다 큰 single-request context를 제공할 수 있다.
총 KV VRAM은 오히려 줄거나 비슷하게 유지된다.
```

주의:

```text
단순히 recent_window만 49K로 올리고 kv_blocks를 줄이면 안 된다.
block table exhausted 상황을 admission/queue가 처리해야 한다.
그렇지 않으면 두 번째 큰 요청에서 OOM 대신 MemoryError/503이 난다.
```

구현 우선순위:

```text
1. ResourcePolicy에 context_tiers / tier_slots 추가
2. kv_blocks default를 max_active*context가 아니라 tier 합으로 계산
3. AdmissionController가 prompt_tokens + max_generation_tokens로 tier 선택
4. large tier busy이면 queue 대기, small tier는 계속 처리
5. /v1/models capabilities에 context_tiers 노출
6. stress: 4K+64K 동시, 8K+56K 동시, 2개 long request queue 확인
```

이 구조가 “손실 없이 VRAM을 아끼면서 2명 동시 기준 최대 context를 늘리는” 다음 정답 구조다.

## 2026-06-18 추가 구현: tier 기반 context admission

구현 내용:

```text
LANGBURST_CONTEXT_TIERS=4096,73728
LANGBURST_CONTEXT_TIER_SLOTS=1,1

EngineResourcePolicy:
  context_tiers/context_tier_slots를 단일 설정 소스로 파싱
  kv_blocks를 균등 max_active*context가 아니라 tier별 block 합으로 계산
  max_active_requests와 max_state_pool_size를 tier slot 합으로 정규화

AdmissionController:
  prompt token 수로 tier를 선택
  들어갈 수 있는 가장 작은 빈 tier slot에 admit
  small tier가 차고 large tier가 비어 있으면 작은 요청도 large tier로 overflow
  큰 요청 1개가 active여도 작은 tier slot이 비어 있으면 작은 요청은 admit
  모든 fitting tier가 차면 queue에서 대기
  largest tier를 넘는 prompt는 admission 단계에서 거절

start_langburst.sh default:
  Q4 fused weight
  INT4_BDR paged KV
  MTP K=5 fixed
  Marlin cache decode_only
  4K small slot 1개 + 72K large slot 1개
```

CPU gate:

```text
python -m pytest -q \
  langburst/tests/test_scheduler_cpu.py \
  langburst/tests/test_resource_policy_cpu.py \
  langburst/tests/test_engine_manager_cpu.py

40 passed
```

이 변경의 목적:

```text
1. 62.6 tok/s champion decode bucket은 그대로 보존
   조건: recent_window=2048, prompt_tokens=1, requests=1, K=5, Marlin cache decode_only

2. OpenWebUI 실사용 default는 OOM 방지와 2명 동시 사용을 우선
   조건: Q4 fused, INT4_BDR, K=5, Marlin cache decode_only, tier 4K+72K

3. 균등 28K+28K보다 큰 single-request context를 제공
   4K+72K total KV budget = 76K tokens

4. 두 번째 긴 요청이 들어오면 OOM으로 망가지지 않고 queue 대기
```

중요한 해석:

```text
decode 속도가 62.6에서 40대까지 보이는 것은 회귀라기보다 benchmark 조건 차이다.
62.6은 extremely short-context single-request champion bucket이다.
OpenWebUI default는 long-context headroom, 2-user admission, OOM 방지를 위해 다른 resource profile을 쓴다.
decode 속도를 잃지 않으려면 serving default와 benchmark champion을 섞지 말고,
benchmark 재현은 2048 window / prompt 1 / cache decode_only로 따로 해야 한다.
```

## 2026-06-18 OpenWebUI 긴 대화 OOM / 반복 출력 RCA

증상:

```text
OpenWebUI 긴 대화에서 "요약\n요약..." 반복 출력 발생
이후 다음 프롬프트에서 CUDA OOM:
  Tried to allocate 20 MiB
  free VRAM 약 1 MiB
```

근본 원인:

```text
1. OpenWebUI가 큰 completion budget을 보내면 반복 degeneracy가 max length까지 이어질 수 있음
2. 4K+72K tier Q4 profile은 load 가능하지만 idle headroom이 너무 작음
3. Prefix cache token budget 버그로 cached_tokens가 max_prefix_tokens를 초과할 수 있었음
   - 예: max_prefix_tokens=16384인데 cached_tokens=35904
   - pinned KV/page refcount가 남아 다음 요청에서 free VRAM을 잠금
4. Prefix cache를 비운 뒤에도 72K/Q4/2-slot profile은 긴 prefill transient 후 headroom이 부족
5. 기존 memory admission은 reserve를 active request 수만큼 잡지 않아,
   두 요청이 각각 통과한 뒤 합산 transient에서 OOM이 발생할 수 있었음
```

정답 구조:

```text
EngineManager.acquire_request(prompt_tokens, engine)
  ↓
AdmissionController가 prompt+completion budget으로 tier slot 예약
  ↓
RuntimeMemoryPressure로 active runtime headroom 검증
  ↓
활성 요청 때문에 headroom이 부족하면 lease를 반납하고 queue/wait
  ↓
단독 실행에서도 reserve를 만족하지 못하면 실행 전 명확히 거절
```

이번 수정:

```text
1. arena_allocated 상태에서도 free VRAM < reserve이면 RuntimeMemoryPressure 발생
2. server.py에서 validate_runtime_memory()를 admission lease 내부로 이동
3. manager.acquire_request(..., engine=engine)가 tier slot + VRAM watermark를 함께 만족할 때만 실행
4. LANGBURST_MAX_GENERATION_TOKENS로 OpenWebUI max_tokens를 운영 cap에 맞춰 clamp
5. BatchGenerationHandle에 반복 n-gram 종료 guard 추가
   - 1-token 반복과 multi-token 반복을 모두 감지
   - finish_reason="repetition"
6. start_langburst.sh가 resource policy CLI 인자를 명시 전달
   - max_active_requests / queued_requests / state_pool_size
   - max_generation_tokens / prefill_chunk / batched_tokens
   - context_tiers / context_tier_slots
   - reserve_free_vram_mib=768
7. RadixPrefixCache가 실제 cached token 합계를 SSOT로 유지
   - 단일 prefix가 max_cached_tokens를 넘으면 저장하지 않고 pinned blocks를 즉시 release
   - 저장 후에도 free VRAM/block watermark를 위반하면 prefix cache를 clear
8. active request 수에 따라 execution reserve를 배수로 적용
   - active_requests=2이면 reserve도 2배 요구
   - 두 번째 요청은 합산 transient가 위험하면 큐에서 대기/거절
```

운영 해석:

```text
72K tier는 "로드 가능/중간 요청 통과"와 "항상 안전"이 다르다.
16GB Q4에서 4K+72K/2-slot은 긴 prefill 뒤 동시 요청 headroom이 부족했다.
운영 기본값은 4K+64K/2-slot + reserve 768MiB로 낮춘다.
72K는 CONTEXT_TIERS=4096,73728 명시 override로만 실험한다.
초장문 raw prompt는 recurrent state와 ring KV를 통해 chunked ingest한다.
admission/VRAM/block allocation은 raw prompt 길이가 아니라 bounded exact KV window 기준으로 계산한다.
즉 logical position과 recurrent state는 전체 스트림을 따라가고, attention KV는 최근 window만 physical block에 순환 저장한다.
정확한 long-range verbatim recall은 최근 KV window 안에서 보장되고, window 밖 정보는 모델의 recurrent compression에 의존한다.
```

최신 ml-dmc8 fast stress:

```text
profile:
  Q4 fused
  context_tiers=4096,65536
  reserve_free_vram_mib=768
  kv_cache_dtype=int4_bdr
  prefix_cache=on

small repeated:
  success, no repetition
  decode 약 28~34 tok/s

long repeated prompt:
  success, no "<think>" loop, no "요약" loop
  prefill 약 866~870 tok/s
  prefix_cache entries=0 under pressure

two concurrent story requests after long prefill:
  success, no CUDA OOM
  one request may run slower due queue/batch scheduling, but server remains healthy

overflow 100KB-ish prompt:
  이전 상태: tokens=75626, max_prompt_tokens=65536에서 HTTP 413
  수정 후 계약: LANGBURST_ALLOW_CONTEXT_OVERFLOW=1이면 거절하지 않고 chunked prefill로 ingest
  KV blocks: min(raw_prompt_tokens, exact_kv_window_tokens)만 allocation
  slot mapping: logical position은 유지하고 physical KV slot은 ring capacity로 modulo 매핑
  prefix cache: overflow prompt는 잘못된 prefix/KV 의미를 피하기 위해 exact KV capacity 안에서만 저장
  주의: latency는 raw token 수에 선형 비례하므로 75K tokens는 866 tok/s 기준 약 87초 prefill이 필요

ml-dmc8 실측:
  input bytes=326452
  prompt_tokens=83223
  max_prompt_tokens/exact_kv_window=65536
  status=200
  prefill_s=96.3244
  prefill_tok_s=863.9866
  completion_tokens=16
  CUDA OOM/illegal access 없음
  health ok=true
  cuda_free_mib=781
  kv used_blocks=0 after release

추가 RCA:
  overflow ingest 자체는 통과했지만, overflow prefill 직후 기존 MTP verifier가
  speculative uniform GDN path에서 CUDA illegal memory access를 냈다.
  원인은 MTP verifier가 exact KV window 밖 logical position/ring trajectory를
  아직 production-safe contract로 처리하지 못하는 점이다.
  해결은 전역 MTP OFF가 아니라, computed_tokens > kv_token_capacity인 overflow row만
  target-only decode로 유지하고 exact-window row에서는 기존 MTP를 그대로 사용한다.
  또한 create_batch_worker(features=...)가 인자를 무시하던 버그를 수정했다.
  요청 단위 feature override가 runner 생성에 반영되지 않아 speculative_decoding=false가
  실제로 적용되지 않는 파편화 원인이었다.
```

추가 CPU gate:

```text
python -m pytest -q \
  tests/test_scheduler_cpu.py \
  tests/test_block_table_cpu.py \
  tests/test_engine_manager_cpu.py::test_engine_manager_generation_admission_limits \
  tests/test_engine_manager_cpu.py::test_engine_manager_allows_overflow_prompt_with_bounded_admission_tokens \
  tests/test_resource_policy_cpu.py

28 passed

추가 gate:

```text
python -m pytest -q \
  tests/test_model_runner_cpu.py::test_batched_model_runner_overflow_row_blocks_next_draft \
  tests/test_engine_manager_cpu.py::test_engine_manager_does_not_duplicate_runner_for_request_level_feature_flags

2 passed
```

새 CPU gate:

```text
env LANGBURST_MTP_BATCH_PROPOSER=1 python -m pytest -q \
  langburst/tests/test_model_runner_cpu.py \
  langburst/tests/test_speculative_batch_cpu.py \
  langburst/tests/test_speculative_cpu.py \
  langburst/tests/test_adapter_runtime_cpu.py \
  langburst/tests/test_batch_worker_cpu.py \
  langburst/tests/test_engine_manager_cpu.py \
  langburst/tests/test_server_config_cpu.py \
  langburst/tests/test_scheduler_cpu.py \
  langburst/tests/test_resource_policy_cpu.py

fast gate:
  tests/test_prefix_cache_cpu.py
  tests/test_engine_manager_cpu.py
  tests/test_model_runner_cpu.py
  tests/test_batch_worker_cpu.py
  tests/test_server_config_cpu.py

63 passed
```
