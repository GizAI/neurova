# Qwen GDN/Recurrent State와 LangBurst 상태 계층

이 문서는 LangBurst native Qwen 경로에서 말하는 `GDN recurrent
state`, `GDN conv state`, `ring KV`, `session state`, `episodic memory`,
`TTT sidecar`가 각각 무엇인지 구분하기 위한 설명이다.

핵심 결론부터 정리하면 다음과 같다.

- Gated DeltaNet(GDN) 자체는 Qwen 계열 hybrid architecture의 모델 구조다.
- LangBurst가 새로 발명한 것은 GDN이 아니라, GDN 상태를 자체 low-bit
  serving runtime에서 직접 보관하고 갱신하는 `DecodeState`, CUDA kernel,
  snapshot/session/ring-KV/sidecar 정책 계층이다.
- 현재 native hot path에 실제 연결된 장기/상태 계층은 GDN recurrent
  state, GDN conv state, bounded/ring attention KV, explicit session state,
  prefix cache, speculative rollback이다.
- `InfiniteStreamingRuntime`, `EpisodicMemory`, `TTTSidecarMemory`는 구현과
  테스트가 있는 research 계층이지만, 기본 OpenAI-compatible chat serving
  path에는 아직 의미 있는 semantic memory로 자동 결합되어 있지 않다.

## 외부 아키텍처 기준

Qwen3-Next와 Qwen3.5 계열은 full attention만 쌓은 일반 Transformer가
아니라, Gated DeltaNet 계열 linear attention layer와 full attention layer를
섞은 hybrid architecture로 설명된다.

참고한 공개 자료:

- vLLM의 Qwen3-Next 지원 문서는 Qwen3-Next의 핵심을 Gated DeltaNet
  linear attention과 full attention의 hybrid attention으로 설명한다.
  또한 vLLM은 hybrid model의 linear/full attention state를 함께 관리하기
  위해 hybrid KV cache manager와 block-size 조정을 사용한다고 설명한다.
  <https://vllm.ai/blog/2025-09-11-qwen3-next>
- NVIDIA Megatron Bridge Qwen 3.5 문서는 Qwen 3.5가 GDN layer와 standard
  attention layer, SwiGLU, RMSNorm을 결합한 hybrid architecture라고 설명한다.
  <https://docs.nvidia.com/nemo/megatron-bridge/0.4.1/models/vlm/qwen35-vl.html>
- DeltaNet 계열 설명에서는 linear attention을 matrix-valued recurrent
  state가 key/value 정보를 누적하는 구조로 해석한다. 즉 전체 과거 token을
  KV cache처럼 모두 저장하는 대신, 고정 크기 행렬 상태를 업데이트한다.
  <https://sustcsonglin.github.io/blog/2024/deltanet-1/>
- GDN decode가 고정 크기 recurrent state를 쓰지만, batch-1 decode에서는 그
  state를 매 token마다 HBM에서 읽고 써야 해서 memory-bound가 될 수 있다는
  분석도 있다. 따라서 GDN은 KV memory를 줄이는 구조이지, decode 연산이
  무조건 공짜가 되는 구조는 아니다.
  <https://arxiv.org/abs/2603.05931>

LangBurst 코드의 `qwen36` 명명은 현재 프로젝트 내부 이름이다. 외부 생태계
문서에서는 Qwen3-Next/Qwen3.5로 표현되는 경우가 많고, LangBurst 구현도
`Qwen3.5/Qwen3.6 split GDN` checkpoint layout을 다루는 주석이 있다. 따라서
여기서는 "Qwen hybrid GDN 계열"이라는 구조적 의미와 LangBurst 내부의
`qwen36` adapter 이름을 분리해서 읽어야 한다.

## 세 가지 기본 상태

LangBurst native Qwen의 `DecodeState`는 세 가지 상태를 함께 들고 간다.

1. GDN recurrent state
2. GDN depthwise-conv state
3. full-attention KV cache

코드상 단일 소스는 다음이다.

- `langburst/adapters/qwen36_impl/state.py`
- `langburst/adapters/qwen36_impl/config.py`
- `langburst/adapters/qwen36_impl/model.py`

### 1. GDN recurrent state

이것은 Qwen GDN layer의 모델 수학에 필요한 고정 크기 recurrent matrix다.
역할은 긴 과거 문맥을 token 수에 비례해 저장하지 않고, layer별 행렬 상태로
압축해 누적하는 것이다.

LangBurst 기본 Qwen shape 기준:

- GDN layer 수: 48
- layer당 shape: `[linear_num_value_heads, linear_key_head_dim,
  linear_value_head_dim]`
- 기본값: `[48, 128, 128]`
- 전체 fp16 용량: 약 72 MiB

이 상태는 "몇 token짜리 버퍼"가 아니다. token 수가 1K든 100K든 recurrent
state 자체의 tensor shape는 변하지 않는다. 대신 매 token마다 같은 행렬을
읽고 새 정보로 갱신한다.

실제 사용 지점:

- 단일 token decode: `Qwen36GDNLayer.__call__()`에서
  `ops.gdn_recurrent_ab(..., state.gdn_states[layer])`
- block/prefill: `Qwen36GDNLayer.forward_block()`에서
  `ops.gdn_recurrent_ab_scan(..., state.gdn_states[layer])`
- batch decode: arena-backed 경로에서
  `ops.gdn_recurrent_ab_batch(..., arena.gdn_states[layer], state_indices)`

즉 LangBurst native에서는 GDN recurrent state가 실제 hot path에서 in-place로
갱신된다.

### 2. GDN conv state

GDN layer 내부에는 recurrent matrix update 전에 q/k/v 입력을 causal
depthwise convolution으로 한 번 통과시키는 경로가 있다. 이 convolution은
최근 몇 token의 local history가 있어야 다음 token 출력을 정확히 만들 수
있다. 그 local history buffer가 `gdn_conv_states`다.

LangBurst 기본 Qwen shape 기준:

- `conv_dim = key_dim * key_heads * 2 + value_dim * value_heads`
- 기본값: `128 * 16 * 2 + 128 * 48 = 10240`
- conv kernel dim: 4
- 저장 history 길이: `kernel_dim - 1 = 3`
- 전체 fp16 용량: 약 2.8 MiB

역할 차이는 다음과 같다.

- GDN recurrent state: 긴 과거를 압축한 matrix memory
- GDN conv state: GDN 입력 projection 주변의 아주 짧은 local causal
  convolution history

둘 다 Qwen GDN layer를 정확히 streaming 실행하기 위해 필요하지만, 의미가
다르다. conv state는 장기 기억이 아니라 짧은 FIR/causal conv buffer에
가깝다.

### 3. Attention KV cache

Qwen hybrid architecture에는 GDN layer뿐 아니라 full attention layer도
있다. full attention layer는 일반 Transformer처럼 K/V cache가 필요하다.

LangBurst 기본 Qwen shape 기준:

- attention layer 수: 16
- KV heads: 4
- head dim: 256
- fp16 KV 대략 용량:
  - 2K window: 약 128 MiB
  - 8K window: 약 512 MiB
  - 16K window: 약 1 GiB

이 값이 `--recent-window`, `LANGBURST_CONTEXT_WINDOW`,
`LANGBURST_RECENT_WINDOW`로 조정되는 token window다.

현재 LangBurst 기본 KV dtype은 `int4_bdr`이며,
`LANGBURST_KV_CACHE_DTYPE` 또는 `--kv-cache-dtype`으로 바꿀 수 있다.

## Qwen 원래 기능과 LangBurst 추가 기능

### Qwen/GDN 원래 구조

다음은 모델 아키텍처 자체에 해당한다.

- GDN layer와 full attention layer를 섞는 hybrid layout
- GDN recurrent matrix update
- GDN gate/decay 계열 수식
- GDN conv/state가 필요한 streaming recurrence
- full attention layer의 KV cache

즉 "GDN recurrent state가 있다"는 사실 자체는 LangBurst 발명이 아니다.
Qwen hybrid GDN 계열을 정확히 inference하려면 어떤 runtime이든 그 의미를
표현해야 한다.

### LangBurst가 추가한 것

LangBurst가 추가한 것은 Qwen 구조를 자체 serving runtime에서 실행하기 위한
구체적인 상태 관리와 kernel/serving 정책이다.

- `DecodeState`
  - GDN recurrent state, GDN conv state, attention KV를 한 객체에 묶는다.
  - fork, copy, reset, decay, snapshot, rollback을 지원한다.
- CUDA/CPU kernel
  - `gdn_recurrent_ab`
  - `gdn_recurrent_ab_scan`
  - `gdn_recurrent_ab_batch`
  - `depthwise_conv_update`
  - `depthwise_conv_update_scan`
  - `depthwise_conv_update_batch`
- ring KV 정책
  - full attention KV를 무한히 키우지 않고 bounded window로 유지한다.
  - window가 찼을 때 logical position modulo window로 physical slot에 쓴다.
- low-bit KV cache
  - `fp16`, `fp8_e4m3`, `int4`, `int4_bdr` 같은 storage policy를 둔다.
- arena-backed batch state
  - continuous batching에서 request별 state를 slot으로 나누고 batch GDN/conv
    kernel이 `state_indices`로 해당 slot을 갱신한다.
- explicit session state
  - `session_id` 또는 `stateful_session`이 있을 때 request 사이에 같은
    `DecodeState`를 보존한다.
- prefix cache
  - 동일 prefix에 대한 state snapshot/KV block을 저장해 재사용한다.
- speculative rollback
  - speculative verification이 state를 먼저 써본 뒤 reject 시 GDN/conv/KV
    state를 되돌린다.
- research sidecars
  - infinite streaming wrapper
  - episodic memory index
  - TTT sidecar memory

## 현재 활성화 상태

여기서 "활성화"는 세 단계로 봐야 한다.

1. 코드/테스트가 존재한다.
2. RuntimeFeature/Capability로 요청할 수 있다.
3. 실제 server/generate hot path에서 의미 있게 사용된다.

### 기본 `stateful` profile

`RuntimeFeatures.from_profile("stateful")` 기준 기본값:

- `kv_window_policy = ring`
- `kv_cache_dtype = DEFAULT_KV_CACHE_DTYPE`, 현재 기본 `int4_bdr`
- `stateful_chat = true`
- `state_pool = true`
- `gpu_sampling = true`
- `speculative_decoding = true`
- `block_prefill = true`
- `prefix_cache = true`
- `snapshots = false`
- `infinite_streaming = false`
- `episodic_memory = false`
- `ttt_sidecar = false`

주의할 점: `stateful_chat = true`는 runtime capability/profile 의미다. 실제
OpenAI-compatible chat request가 request 사이에서 state를 유지하려면
`session_id`, `stateful_session`, 또는 `previous_response_id`가 필요하다.
그 값이 없으면 server default chat은 stateless로 처리되고 요청 종료 시
state가 release/reset된다.

### `research` profile

`RuntimeFeatures.from_profile("research")`는 다음을 추가로 켠다.

- `snapshots = true`
- `infinite_streaming = true`
- `episodic_memory = true`
- `ttt_sidecar = true`

하지만 현재 코드 기준으로 `episodic_memory`와 `ttt_sidecar`는 feature plan과
research module 수준의 구현이다. 기본 chat generation loop에 retrieved
episode를 자동 삽입하거나 TTT residual을 logits/model hidden에 주입하는
완성된 product path는 아니다.

### 실제 hot path에 붙은 것

현재 native chat/server/generate에서 실제로 작동하는 상태 계층:

- GDN recurrent state
- GDN conv state
- bounded/ring attention KV
- int4/fp8/fp16 KV storage policy
- request-local state pool
- explicit session state
- prefix cache
- block prefill
- speculative state snapshot/rollback
- arena-backed state slotting

현재 research/demo 성격인 것:

- `InfiniteStreamingRuntime`
  - unbounded token ingestion API와 snapshot API는 있다.
  - 기본 server chat path와는 별도 wrapper다.
- `EpisodicMemory`
  - hash embedding 기반 local state-RAG index다.
  - record가 state delta/snapshot path를 가리킬 수 있게 되어 있다.
  - 기본 chat path에서 자동 retrieval/merge되지는 않는다.
- `TTTSidecarMemory`
  - hidden vector로 low-rank fast-weight memory를 업데이트하고 읽는 sidecar다.
  - base Qwen GDN state를 오염시키지 않는 별도 memory로 테스트되어 있다.
  - 기본 chat path에 residual로 자동 주입되지는 않는다.

## "262K 이상 긴 컨텍스트" 표현에 대한 해석

GDN recurrent state는 full attention KV처럼 token마다 cache를 계속 늘리지
않는다. 따라서 GDN layer만 보면 token 수가 길어져도 state 크기는 고정이다.
이 점 때문에 hybrid GDN architecture는 긴 입력을 훨씬 메모리 효율적으로
처리할 수 있다.

다만 LangBurst native의 현재 정확한 의미는 다음처럼 구분해야 한다.

- GDN compressed memory: token 수와 무관하게 고정 크기다.
- full attention recent KV: `recent_window`만큼 정확히 보존한다.
- window 밖 과거:
  - GDN recurrent state에는 압축되어 반영된다.
  - full attention KV처럼 원문 token별 K/V가 모두 남는 것은 아니다.
- exact retrieval:
  - long-context 원문을 정확히 다시 찾아야 하는 능력은 full attention KV와
    별개로 episodic/RAG 같은 외부 memory가 필요할 수 있다.

따라서 "무한 컨텍스트"라는 말은 "모든 과거 token을 full attention처럼
무손실 조회한다"가 아니라, "bounded exact KV + fixed recurrent state로
메모리를 고정한 채 긴 stream을 계속 처리한다"에 가깝다.

## 동작 신뢰도

현재 테스트로 확인되는 범위:

- GDN recurrent reference/CPU fallback parity
- CUDA GDN recurrent kernel parity
- GDN scan/batch kernel parity
- ring KV wrap boundary와 snapshot roundtrip
- state fork/branch isolation
- state delta apply
- warm boot snapshot continuation
- long streaming memory budget 고정
- arena slot recycle과 request별 state 분리
- explicit session이 같은 decode state를 재사용하고, stateless chat은 재사용하지 않음

아직 조심해야 할 범위:

- ring KV logical view는 일부 fallback path에서 `torch.cat` materialization이
  발생할 수 있어 성능 병목이 될 수 있다.
- CUDA graph decode는 아직 shipped path가 아니다.
- episodic memory/TTT sidecar는 research scaffold이며, product chat path에
  자동 semantic memory로 통합된 상태가 아니다.
- vLLM provider 경로에서는 Qwen/GDN recurrent semantics가 metadata/bridge
  방향으로만 표현된 부분이 있고, native와 같은 `DecodeState` lifecycle이
  완전히 동일하게 작동한다고 보면 안 된다.

## 운영 관점 요약

기본 native serving에서 실제로 믿고 써도 되는 상태 구조는 다음이다.

```text
request/session
  -> DecodeState
     -> GDN recurrent state   # fixed-size compressed long memory
     -> GDN conv state        # short local convolution history
     -> attention KV cache    # bounded exact recent memory
        -> ring/fp16/fp8/int4_bdr storage policy
```

명시적 session을 쓰는 경우:

```text
session_id
  -> SessionStateStore
     -> persistent DecodeState across turns
```

research 계층까지 확장하는 목표 구조:

```text
long stream / large corpus
  -> InfiniteStreamingRuntime
     -> DecodeState snapshots / deltas
     -> EpisodicMemory retrieval index
     -> optional TTT sidecar memory
```

현재 결론은 분명하다. Qwen GDN recurrent state와 conv state는 native Qwen
구현에서 실제로 사용 중이다. LangBurst의 차별점은 이 상태를 low-bit native
serving, session, ring KV, snapshot, speculative rollback, batch arena까지
하나의 runtime contract로 묶어낸 점이다. 반면 episodic memory와 TTT sidecar는
아직 기본 serving 의미론의 일부가 아니라 research scaffold로 보는 것이 맞다.
