# HERA-LM Architecture Proposal

**HERA = Hierarchical Event Recurrent Automaton**

목표는 기존 Transformer와 Mamba3를 능가할 가능성이 있는, 단순하지만 고성능이고 효율적이며 자연스러운 문장 생성을 잘하는 신규 아키텍처를 설계하는 것이다.

단, 중요한 전제는 다음과 같다.

> “Transformer와 Mamba3를 능가한다”는 것은 설계만으로 보장할 수 없고, 학습과 평가로 증명해야 한다. 이 문서는 그 목표를 달성하기 위한 연구 설계안이다.

---

## 1. 핵심 철학

HERA-LM의 핵심 철학은 다음과 같다.

```text
문장 생성은 local syntax path가 담당
긴 문맥 흐름은 recurrent delta memory가 담당
정확한 사실/대화 기억은 sparse fact slots가 담당
출력은 일반 LM head가 담당
memory head는 아주 약하게, 나중에만 켬
```

즉 Transformer처럼 모든 token이 모든 token을 보지 않고, Mamba처럼 하나의 연속 state에 모든 것을 밀어 넣지도 않는다. 문법, 흐름, 사실 기억을 분리한다.

---

## 2. 전체 구조

```text
Input tokens / patches
  ↓
Factorized Embedding
  ↓
HERA Block x N
  ├─ SyntaxMix        : 문장/구문/짧은 문맥
  ├─ DeltaFlow Memory : 긴 문맥 recurrent state
  ├─ FactBoard Slots  : sparse key-value memory
  └─ Router           : 세 경로를 동적으로 섞음
  ↓
RMSNorm
  ↓
Factorized tied LM head
```

한 블록은 다음과 같이 동작한다.

```text
h = h + SyntaxMix(h)
r = DeltaFlowRead(h, recurrent_state)
s = FactSlotRead(h, slots)
h = h + RouterMLP(h, r, s)
recurrent_state = DeltaFlowWrite(recurrent_state, h)
slots = FactSlotEdit(slots, h at chunk boundary)
```

핵심은 세 역할을 명확히 분리한다는 점이다.

---

## 3. Transformer와 다른 점

Transformer는 강하지만 비싸다.

```text
Transformer:
  local grammar 좋음
  exact recall 좋음
  하지만 긴 context에서 KV cache와 attention 비용 큼
```

HERA는 full attention을 쓰지 않는다.

```text
HERA:
  token-token full attention 없음
  짧은 문맥은 convolution/dynamic filter
  긴 문맥은 recurrent memory
  정확한 사실은 sparse slots
```

목표 복잡도는 다음과 같다.

```text
Transformer training: O(T²)
Transformer decode cache: O(T · layers · d)

HERA training: O(T)
HERA decode state: O(layers · recurrent_state + slots)
```

---

## 4. Mamba3와 다른 점

Mamba3류는 recurrent state가 강하지만, 정확한 key-value recall과 사실 overwrite/protect가 어렵다.

HERA는 state를 둘로 나눈다.

```text
DeltaFlow:
  연속적인 문맥 흐름, 스타일, 주제, 문장 진행

FactBoard:
  이름, 코드, 장소, 약속, 사용자 정보, 문서 근거
```

즉 Mamba식 “흐름 state”와 LUMA식 “편집 가능한 사실 memory”를 분리한다.

---

## 5. HERA Block 상세

### 5.1 SyntaxMix

문장 생성 품질은 SyntaxMix에서 나온다.

```text
SyntaxMix =
  RMSNorm
  → multi-kernel causal depthwise conv
  → gated pointwise MLP
  → residual
```

예시 구현 형태:

```python
z = RMSNorm(h)
a = DWConv3(z) + DWConv7(z) + DWConv15(z)
u, g = Linear(a).chunk(2)
syntax = u * sigmoid(g)
h = h + syntax
```

SyntaxMix의 역할은 다음과 같다.

```text
- 영어/한국어 문장 리듬
- 조사/전치사/어순
- 짧은 phrase 연결
- 반복 붕괴 방지
```

여기에는 full attention이 없다.

---

### 5.2 DeltaFlow Memory

DeltaFlow는 Mamba3를 대체하거나 보완하는 recurrent path다. 핵심은 channel-wise erase/write다.

각 layer는 작은 grouped matrix memory를 가진다.

```text
M_g: [d_key, d_value]  for each group g
```

매 token마다 다음을 수행한다.

```text
q, k, v, erase, write = projections(h)

read:
  r = M · q

update:
  M = M * (1 - erase ⊗ k) + write ⊗ v
```

더 안정적인 형태는 다음과 같다.

```text
M_g ← RMSNorm(
        M_g * (1 - sigmoid(e_g) · k_g)
      + sigmoid(w_g) · outer(k_g, v_g)
     )
```

DeltaFlow의 역할은 다음과 같다.

```text
- 현재 주제 유지
- 문장 흐름 유지
- 최근 대화 분위기 유지
- long-range dependency 압축
```

Mamba보다 해석이 쉬운 이유는 erase/write가 명시적이기 때문이다.

---

### 5.3 FactBoard Slots

FactBoard는 LUMA의 핵심을 더 안정화한 것이다.

각 layer에 sparse slot memory를 둔다.

```text
slots:
  key        [n_slots, d]
  value      [n_slots, d]
  confidence [n_slots]
  utility    [n_slots]
  age        [n_slots]
  lock       [n_slots]
```

읽기:

```text
score = q · slot_key + utility_bias - age_bias
idx = topk(score)
slot_read = weighted_sum(slot_value[idx])
```

쓰기:

```text
event = FactPool(chunk_hidden)

candidate = W[event, slot_read]

erase   = sigmoid(...)
write   = sigmoid(...)
protect = sigmoid(... + lock)

slot.value =
  protect * old
  + (1 - protect) * ((1 - erase) * old + write * candidate)
```

기존 LUMA와의 중요한 차이는 다음과 같다.

```text
기존 LUMA:
  chunk mean 또는 불안정 fact_pool
  slot_delta 폭주 가능

HERA:
  fact_pool entropy 제한
  slot value RMSNorm
  slot_delta clipping
  memory_scale warmup
```

---

### 5.4 Router

세 경로를 고정 비율로 섞으면 안 된다. token마다 다르게 섞어야 한다.

```python
router = sigmoid(W_router([h, delta_read, slot_read]))

h = h + SwiGLU(
    W_out([
      h,
      router_delta * delta_read,
      router_slot  * slot_read
    ])
)
```

일반 문장은 slot을 거의 안 쓴다.

```text
"The cat is on the table."
→ SyntaxMix + DeltaFlow 위주

"Mina's access code is AX-917."
→ FactBoard write 강함

"What is Mina's access code?"
→ FactBoard read 강함
```

모든 token에 memory를 강제로 섞으면 문장이 망가진다.

---

## 6. 출력부

현재 LUMA처럼 memory logits를 바로 크게 더하면 안 된다. 출력은 기본적으로 일반 LM head가 담당해야 한다.

```text
기본:
  logits = LMHead(h)

나중:
  logits += small_gate * MemoryHead(memory_signal)
```

권장 stage별 설정:

```text
Stage 1/2:
  memory_head off
  copy_head off

Stage 3:
  memory_head on, scale tiny
  memory_scale_init = -4.0

Stage 4:
  memory head가 ablation에서 도움될 때만 scale 증가
```

수식:

```python
lm_logits = lm_head(h)

memory_gate = sigmoid(memory_scale)   # 초기 거의 0
logits = lm_logits + memory_gate * memory_head(memory_h)
```

금지해야 하는 형태:

```python
logits = lm_logits + 1.0 * memory_logits  # 위험
```

---

## 7. Tokenizer / Embedding

최소 학습으로 문장을 잘 만들려면 bytepatch부터 하면 안 된다. 먼저 subword가 필요하다.

권장 tokenizer 경로:

```text
Stage 1~3:
  Qwen tokenizer 또는 Length-MAX류 tokenizer

Stage 4:
  bytepatch alignment

Final:
  dual route
    qwen/length-max → normal language
    bytepatch       → OOD/code/noisy text/proof span
```

Qwen vocab이 크면 embedding이 너무 커진다. 그래서 factorized embedding을 쓴다.

```text
token_id
  ↓
Embedding[vocab, d_embed=256 or 384]
  ↓
Project to d_model
```

출력도 tied factorized head를 쓴다.

```text
h
  ↓
Project d_model → d_embed
  ↓
tied vocab head
```

이렇게 하면 250k vocab도 작은 모델에서 버틸 수 있다.

---

## 8. 최종 블록 의사코드

```python
class HERABlock(nn.Module):
    def forward(self, h, state):
        # 1. local sentence path
        z = h + self.syntax_mix(self.norm1(h))

        # 2. recurrent flow memory
        delta_read, state.delta = self.delta_flow(
            self.norm2(z),
            state.delta
        )

        # 3. sparse fact slots
        slot_read = self.slot_read(self.norm3(z), state.slots)

        # 4. route memory into token stream
        route = torch.sigmoid(self.router(torch.cat([z, delta_read, slot_read], -1)))

        fused = self.fuse(torch.cat([
            z,
            route[..., 0:1] * delta_read,
            route[..., 1:2] * slot_read,
        ], -1))

        h = z + fused

        # 5. update slots once per chunk
        if state.end_of_chunk:
            event = self.fact_pool(h)
            state.slots = self.slot_edit(event, state.slots)

        return h, state
```

---

## 9. 추천 모델 크기

### 9.1 HERA-Speech-120M

RTX 4080 기준 현실적인 첫 실험 크기다.

```text
tokenizer: qwen
d_embed: 256
d_model: 640
layers: 10
ffn_mult: 2.5
delta_groups: 8
delta_key_dim: 32
slots: 128
slot_topk: 8
chunk_size: 64
copy_head: off
memory_head: off
```

목표:

```text
1000~3000 step 안에 자연스러운 짧은 문장
chat_sanity pass_rate >= 0.8
```

---

### 9.2 HERA-Base-300M

```text
d_embed: 384
d_model: 896 or 1024
layers: 14~16
delta_groups: 8~16
delta_key_dim: 48~64
slots: 256
chunk_size: 64
memory_head: weak
```

목표:

```text
작은 대화 모델
기본 instruction
간단 reasoning
slot memory proof
```

---

### 9.3 HERA-1B

```text
d_model: 1536
layers: 24
slots: 512
chunk_size: 64 or 128
dual tokenizer route
speculative/MTP head
```

목표:

```text
Transformer/Mamba3와 본격 비교
```

---

## 10. 학습 순서

아키텍처보다 학습 순서가 더 중요하다.

### Stage 0: Speech prior

```text
memory_head = off
copy_head = off
slot write = weak or frozen
delta memory = on
syntax path = on

data:
  TinyStories-style simple English
  short ChatML QA
  Korean short QA
  definitions
  simple arithmetic
  anti-repetition examples
```

목표:

```text
hi → 자연스러운 답
who are you → 자연스러운 자기소개
what is ML → 짧은 정의
한국어 질문 → 한국어 답
반복 붕괴 없음
```

---

### Stage 1: Teacher distillation

```text
teacher = Qwen / DeepSeek / existing strong model

loss:
  L_next_token
+ L_teacher_logits
+ L_hidden_align optional
```

랜덤 초기화만으로는 너무 비싸다. 최소 학습으로 문장 품질을 얻으려면 teacher가 필요하다.

---

### Stage 2: Chat SFT

```text
strict ChatML
answer-only loss
short answers
clean style
```

Stage2 gate:

```text
chat_sanity >= 0.8
repeat4_max < 5
replacement char 없음
```

이 gate를 통과하기 전에는 memory 학습을 금지한다.

---

### Stage 3: Memory proof

```text
slot write on
memory_head weak on
memory_scale tiny
ablation margin weak
```

목표:

```text
normal > no_slots + 20%
normal > random_slot_keys + 20%
no_copy에서도 유지
```

---

### Stage 4: Bytepatch alignment

```text
qwen route로 만든 hidden/slot state
↔
bytepatch route hidden/slot state alignment
```

이때 최종 tokenizer-free robustness를 얻는다.

---

## 11. 왜 기존 LUMA보다 나은가

현재 LUMA 문제:

```text
1. 말 배우기 전에 memory를 너무 세게 켬
2. memory_logits가 generation을 망침
3. slot_delta 폭주
4. random_slot_keys와 normal 차이가 없음
5. chat stage가 통과 안 됐는데 memory stage로 감
```

HERA 해결:

```text
1. speech path와 memory path 분리
2. memory head는 tiny gate로 나중에만 켬
3. DeltaFlow가 일반 문맥을 담당
4. FactBoard는 사실 기억만 담당
5. slot update를 RMSNorm/clip으로 안정화
6. chat gate 통과 전 memory 금지
```

---

## 12. 왜 Transformer보다 나을 수 있는가

목표상 장점:

```text
- full attention 없음
- KV cache 작음
- 긴 문맥 비용 낮음
- slot memory로 fact update/overwrite 가능
- bytepatch route로 OOD text 대응 가능
```

Transformer가 강한 local grammar는 `SyntaxMix`가 담당하고, Transformer가 비싼 long context는 `DeltaFlow + FactBoard`가 담당한다.

---

## 13. 왜 Mamba3보다 나을 수 있는가

Mamba3는 연속 state가 강하지만, 명시적 사실 저장/수정/보호는 약하다.

HERA는 다음을 분리한다.

```text
DeltaFlow:
  Mamba류 장점 흡수

FactBoard:
  key-value fact recall
  overwrite/protect
  user memory
  evidence pointer

SyntaxMix:
  문장 생성 안정화
```

즉 Mamba3의 state tracking 약점을 FactBoard로 보완한다.

---

## 14. 반드시 필요한 ablation

이 ablation들을 통과해야 “능가 가능성”을 말할 수 있다.

```text
normal
no_deltaflow
no_factboard
random_fact_slots
no_memory_head
no_syntaxmix
no_teacher_distill
no_bytepatch
```

성공 기준:

```text
문장:
  chat_sanity >= 0.9
  repetition collapse 없음

메모리:
  copy > 70%
  json_field > 80%
  recall > 80%
  update/protect > 70%

효율:
  same quality에서 Transformer보다 낮은 KV/cache
  Mamba3와 비슷하거나 빠른 decode
```

---

## 15. 한 문장 요약

**HERA-LM은 full attention 대신 `SyntaxMix`, Mamba식 단일 state 대신 `DeltaFlow + FactBoard`, 강한 memory logits 대신 `weak gated memory head`를 쓰는 하이브리드 recurrent memory 언어 모델이다.**

가장 중요한 설계 원칙은 다음과 같다.

```text
문장은 memory가 아니라 syntax path가 만들고,
흐름은 delta recurrent state가 유지하고,
사실은 sparse slots가 기억한다.
```

이 방향이 현재 LUMA를 “말이 안 되는 memory 실험”에서 대화도 되고, 빠르고, 장기 기억도 되는 신규 아키텍처로 바꾸는 가장 현실적인 설계다.
