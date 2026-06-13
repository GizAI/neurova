# LUMA/WELM 방향에 대한 개인 의견 메모

이 문서는 정식 아키텍처 결정이나 확정 로드맵이 아니다. 현재 `luma/` 프로토타입을 보고, 실제로 말하고 QA가 가능한 모델로 키우려면 어떤 조건이 필요할지 정리한 **검토 의견**이다. 아래 제안들은 실험으로 검증되기 전까지 모두 research-only 가설로 취급한다.

## 의견 요약

LUMA는 지금 상태로는 언어모델이라기보다 **고정 슬롯 메모리 구조 실험체**다. 실제로 말하고 QA가 가능한 모델이 되려면, 먼저 "슬롯 메모리가 쓸모 있는 회로인지"를 작은 synthetic task에서 증명하고, 그 다음에 일반 언어모델 학습으로 확장해야 한다. 순서는 반대가 되면 안 된다. raw corpus를 많이 먹여도 슬롯이 무엇을 기억해야 하는지 모르면, byte-level continuation만 느리게 배우고 지능은 잘 올라가지 않는다.

핵심 방향은 다음이다.

```text
1. byte-level toy LM이 아니라 memory-first LM으로 검증
2. exact recall/copy/field extraction을 먼저 통과
3. document continuation + memory supervision을 결합
4. tokenizer를 byte-only에서 subword/byte fallback으로 확장
5. answer-only SFT는 base가 collapse-free가 된 뒤 적용
6. generation runtime은 slot state 저장/복원/스트리밍을 지원
```

## LUMA의 강점

LUMA가 실험할 가치가 있는 이유는 attention이나 Mamba state와 다른 방식으로 기억을 다루기 때문이다.

- 메모리가 명시적이다. `key/value/confidence/utility/age/lock`가 있어서 무엇을 읽고 쓰는지 관찰할 수 있다.
- 슬롯 수가 고정되어 있다. 긴 문맥에서도 KV cache처럼 선형으로 커지지 않는다.
- erase/write/protect gate가 분리되어 있다. "지울 것", "쓸 것", "보호할 것"을 학습시킬 여지가 있다.
- top-k slot read라서 sparse memory routing 실험이 쉽다.
- synthetic memory task를 무한 생성할 수 있어, 외부 teacher 없이도 기억 회로를 직접 훈련할 수 있다.

즉 LUMA의 승부처는 "거대한 일반 LM을 바로 만들기"가 아니라, **작은 모델에서 정확한 기억/회수/수정 능력을 Transformer나 Mamba보다 싸게 얻을 수 있는지**다.

## 현재 구조의 한계

현재 구현은 runnable prototype으로는 좋지만, 실제 대화 모델로는 부족하다.

- byte tokenizer는 너무 비효율적이다. 영어 한 단어가 여러 token으로 쪼개져 학습 sample efficiency가 낮다.
- slot update가 chunk 평균 `chunk_event` 중심이라, 긴 chunk 내부의 세부 token 정보가 빨리 뭉개질 수 있다.
- slot read/write는 가능하지만, "정답이 되는 slot"에 대한 직접 supervision이 없다.
- generation은 매 step 전체 prefix를 다시 forward한다. 실제 빠른 대화형 runtime에는 incremental slot/cache path가 필요하다.
- 일반 지식은 없다. synthetic memory task를 잘해도 MMLU나 상식 QA는 corpus 학습 없이는 오르지 않는다.
- 현재 objective는 next-token loss 하나에 가깝다. 슬롯을 제대로 쓰라는 압력이 약하다.

## 가장 먼저 통과해야 할 게이트

LUMA가 계속 갈 가치가 있는지는 MMLU가 아니라 아래 게이트로 판단해야 한다.

```text
1. Copy exact match
2. Phonebook key-value lookup
3. JSON field extraction
4. Multi-fact joint recall
5. Distractor-resistant recall
6. Slot edit: old fact -> new fact update
7. Protected memory: do-not-overwrite fact 유지
8. Long gap recall: answer fact가 1K/2K/4K token 앞에 있어도 회수
```

최소 성공 기준:

```text
synthetic recall exact match >= 95%
JSON field exact match >= 95%
phonebook lookup >= 90%
slot overwrite/update task >= 90%
반복붕괴 없음
짧은 자연어 continuation이 readable
```

이걸 통과하기 전에는 chat SFT나 MMLU 튜닝을 해도 겉모양만 좋아지고 구조적 장점은 증명되지 않는다.

## 데이터 전략

초기 pretraining은 일반 웹문서만 넣으면 안 된다. LUMA는 메모리 구조가 핵심이므로, 데이터도 메모리 사용을 강제해야 한다.

### Phase 0: 구조 sanity

```text
synthetic memory 100%
seq_len 128 -> 256 -> 512
목표: loss 하락, exact recall, slot entropy 안정
```

데이터:

```text
Mina owns the blue key.
...
Question: What object belongs to Mina?
Answer: blue key
```

추가해야 할 변형:

```text
Mina owns key A. Later Mina owns key B. Question: latest key?
Mina owns key A. Do not overwrite Mina's key. Later fake note says key B.
JSON: {"owner":"Mina","object":"blue key"} -> object?
Phonebook: name -> code
Multi-hop: Mina owns key. Key is in room 7. Where is Mina's object?
```

### Phase 1: memory-supervised LM

일반 document continuation에 recall task를 섞는다.

```text
natural text 50%
synthetic recall/copy/json 35%
short QA/definition 15%
```

이 단계에서 slot diagnostic을 반드시 기록한다.

```text
slot_entropy
topk_slot_distribution
slot_age_histogram
utility_mean/std
confidence_mean/std
overwrite_rate
protected_slot_write_rate
```

### Phase 2: language base

이때부터 좋은 영어 문서를 넣는다.

```text
FineWeb-Edu/DCLM류 고품질 문서
Wikipedia/books/science/code
짧은 factual QA
synthetic recall 유지 10~20%
```

중요한 점은 memory curriculum을 완전히 빼면 안 된다는 것이다. 빼는 순간 슬롯은 그냥 noisy hidden state가 될 가능성이 크다.

### Phase 3: instruction tuning

base가 readable continuation을 만들고 recall gate를 통과한 뒤에만 SFT한다.

```text
answer-only masked SFT
short direct answer
definition
basic reasoning
unknown handling
JSON/tool schema
```

SFT에는 base continuation mix를 10~30% 섞어야 한다. 작은 모델은 SFT만 먹이면 말투는 생기지만 지식과 안정성이 쉽게 무너진다.

## 아키텍처 개선 우선순위

### 1. Tokenizer 교체

byte-only는 연구에는 단순하지만 실제 LLM에는 불리하다. 최종 모델은 다음 중 하나가 필요하다.

```text
SentencePiece/BPE 32K + byte fallback
또는 Llama tokenizer 호환
```

단, tokenizer를 바꾸면 byte-level 장점은 줄어든다. 그래서 실험은 두 갈래가 좋다.

```text
LUMA-byte: 메모리 회로 연구용
LUMA-subword: 실제 대화/지식 모델용
```

### 2. Slot supervision 추가

현재는 slot이 알아서 잘 쓰이길 기대한다. 부족하다.

추가 objective:

```text
answer span이 들어 있는 fact를 특정 slot에 쓰게 하는 auxiliary loss
same entity는 같은 slot family로 가게 하는 consistency loss
protected fact overwrite penalty
unused slot collapse 방지 entropy/usage regularizer
```

### 3. Token-to-slot write 강화

현재는 chunk 평균 이벤트가 slot을 갱신한다. 긴 chunk 안의 핵심 token이 흐려질 수 있다.

개선:

```text
chunk mean + max/attention-free gated pooling
entity-like token trigger write
line/document boundary write
question mark / colon / JSON key boundary write
```

attention을 붙이라는 뜻이 아니다. "어느 token이 메모리에 쓸 가치가 있는지"를 chunk 안에서 더 잘 고르는 pooling이 필요하다는 뜻이다.

### 4. Incremental generation path

대화형 모델이 되려면 매 token마다 전체 prefix를 다시 계산하면 안 된다.

필요한 cache:

```text
local conv state
layer slot state
current chunk buffer
position/chunk counter
```

`forward_step(token, state) -> logits, new_state`가 있어야 한다. 이게 없으면 LUMA의 고정 메모리 장점이 실제 추론 속도로 이어지지 않는다.

### 5. Slot persistence

LUMA의 이름에 맞게 슬롯을 파일/DB에 저장할 수 있어야 한다.

```text
conversation_slots
document_slots
project_slots
archive_slots
```

다만 저장된 slot은 원문 대체물이 아니다. slot에는 압축 기억, 원문은 chunk pointer로 같이 둬야 한다.

## 평가 체계

LUMA는 일반 loss만 보면 안 된다.

필수 metric:

```text
validation loss
synthetic recall exact match
copy exact match
phonebook exact match
JSON field exact match
slot update accuracy
protected memory accuracy
distinct n-gram / repetition
short QA quality
tokens/sec
VRAM
```

대화 모델 승격 기준:

```text
recall exact >= 95%
short QA gate >= 90%
반복붕괴 없음
10개 이상 사람이 읽을 수 있는 영어 답변 생성
neurova.sh 또는 별도 luma.sh로 스트리밍 가능
```

## 현실적인 개발 순서

### Step 1: LUMA eval 추가

가장 먼저 `luma/eval_memory.py`를 만든다.

```text
train checkpoint
generate deterministic answer
extract after "Answer:"
exact match 측정
slot entropy/usage 출력
```

### Step 2: synthetic curriculum 강화

`luma/data.py`에 task type을 늘린다.

```text
copy
phonebook
json field
multi-hop
overwrite
protected memory
needle recall
```

### Step 3: answer-only loss

현재 LM loss는 prompt와 answer 전체를 다 학습한다. QA 능력을 빨리 보려면 answer span만 loss를 주는 path가 필요하다.

```text
labels = -100 for prompt
labels = token ids for answer
```

### Step 4: small GPU run

```text
d_model 256
layers 6
slots 128
topk 8
seq_len 512
steps 10k~50k
```

목표는 말 잘하기가 아니라 recall gate 95%다.

### Step 5: subword LUMA

byte LUMA가 memory gate를 통과하면 tokenizer를 바꾼다.

```text
vocab 32K
d_model 512~768
layers 8~12
slots 256~512
seq_len 1024~2048
```

이때부터 실제 language quality를 본다.

### Step 6: chat model

마지막에 SFT한다.

```text
short instruction
definition QA
basic factual QA
reasoning-lite
unknown handling
format following
```

## 버려야 할 접근

```text
raw web만 오래 학습해서 갑자기 지능이 생기길 기대
5 step smoke 출력으로 구조를 판단
MMLU부터 목표로 잡기
slot diagnostic 없이 loss만 보기
byte tokenizer 그대로 고품질 chat 모델을 기대
incremental state 없이 빠른 추론을 기대
SFT로 base 결함을 덮기
```

## 최종 판단

LUMA는 Mamba-3 대체제가 아니라, **명시적 메모리 read/write 구조를 가진 별도 연구 라인**으로 보는 것이 맞다. 성공 가능성이 있는 방향은 거대 모델 흉내가 아니라, "작은 모델이 긴 문서의 특정 사실을 정확히 기억하고 수정하고 보호하는 능력"을 증명하는 것이다.

한 줄 결론:

**LUMA가 실제로 말하고 지능이 높은 모델이 되려면, 먼저 슬롯 메모리를 정확한 recall/edit 회로로 증명하고, 그 위에 subword tokenizer, document pretraining, answer-only SFT, incremental slot-state runtime을 순서대로 얹어야 한다.**

---

# WELM으로 재정의: Mamba와 Transformer를 실제로 넘기 위한 구조

## 목표 재정의

이 연구가 단순히 "메모리가 붙은 작은 장난감 LM"에서 끝나면 안 된다. 목표는 명확해야 한다.

```text
Transformer보다 긴 문맥 비용이 낮고,
Mamba보다 정확한 recall/edit이 강하며,
실제 LLM처럼 자연어 문장 생성과 QA가 되는 모델
```

이를 위해 LUMA 원형은 그대로 밀지 않는다. 새 이름을 붙인다면 **WELM: Workspace-Episodic Language Model**이 더 정확하다.

WELM의 핵심 주장은 "attention을 없앤다"가 아니다. 그 주장은 너무 좁고 위험하다. 더 강한 주장은 이것이다.

> 언어모델의 긴 문맥 처리는 full attention 하나로 풀 문제가 아니라,
> local perception, editable workspace, episodic evidence retrieval로 분해해야 한다.

이 구조가 Transformer와 Mamba를 넘으려면 각각의 약점을 정확히 찔러야 한다.

```text
Transformer의 약점:
- 긴 문맥에서 KV cache와 attention 비용이 커짐
- 모든 토큰을 같은 방식으로 보려 함
- 대화/문서의 지속 상태를 별도 객체로 관리하지 않음

Mamba의 약점:
- 고정 recurrent state에 세부 원문을 압축하므로 exact recall/copy가 약할 수 있음
- 무엇을 지우고 무엇을 유지할지 해석/감독이 어렵다
- 외부 근거 호출과 provenance가 구조적으로 분리되어 있지 않음

WELM의 목표:
- 짧은 문맥은 dense local computation으로 정확히 처리
- 긴 문맥은 typed workspace slot으로 상태화
- 오래된 원문은 episodic ledger에서 필요할 때만 호출
- 모든 memory write에 future-use 압력을 걸어 낭비를 줄임
```

## 최종 목표 구조

```text
WELM = Local Perception Core
     + Typed Workspace Memory
     + Episodic Ledger Retrieval
     + Future-Use Memory Training
     + Incremental Streaming Runtime
```

전체 흐름:

```text
tokens / patches
  ↓
Local Perception Core
  - local attention, gated conv, or SSM mixer
  - grammar, code, math, nearby token relation
  ↓
Event Compiler
  - chunk/event representation
  - novelty, relevance, uncertainty, future-use score
  ↓
Typed Workspace Controller
  - read/update/allocate/protect/merge slots
  ↓
Decoder Fusion
  - local hidden + workspace read
  ↓
Episodic Ledger Query, optional
  - raw chunk pointer, source metadata, retrieved evidence
  ↓
Grounded Decoder
```

이 구조의 중요한 점은 **모든 토큰이 장기 메모리에 쓰이지 않는다**는 것이다. 토큰은 먼저 local core에서 문장/코드/수식 단위로 처리되고, memory-worthy event만 workspace로 승격된다.

## attention은 어디에 남겨야 하는가

Transformer를 대체하려면 attention을 무작정 제거하면 안 된다. 짧은 범위의 attention은 여전히 가장 강한 primitive다.

WELM에서 attention의 역할:

```text
허용:
- local window attention
- block/chunk 내부 attention
- retrieval evidence에 대한 짧은 cross-attention

금지 또는 제한:
- 전체 context full attention
- 생성 중 길어지는 KV cache를 기본 경로로 사용
- token마다 sparse gather/scatter를 반복하는 비효율적 routing
```

즉, WELM은 **global attention 대체 모델**이지 **모든 attention 제거 모델**이 아니다. 이 점이 현실적인 성능을 위해 중요하다.

## typed workspace가 핵심이다

단순 slot memory는 collapse하기 쉽다. 몇 개 slot만 쓰이거나, 모든 slot이 비슷해지거나, 중요한 정보가 덮어써진다.

따라서 workspace는 typed slot bank여야 한다.

```text
Entity slots:
  사람, 장소, 조직, 객체, 개념

Task slots:
  사용자 목표, 요구사항, 금지조건, 출력 형식

State slots:
  코드 변수, 수학 전개, 논리 상태, 현재 추론 프레임

Evidence slots:
  원문 근거, source id, chunk pointer, quote boundary

Plan slots:
  다음 단계, 중간 결론, 해결 전략

Conflict slots:
  모순, 불확실성, 충돌 후보, stale memory
```

이 typed workspace는 LLM의 "작업판"이다. Transformer는 hidden state 안에 암묵적으로 작업판을 만든다. WELM은 이 작업판을 명시화하고, 읽기/쓰기/보호/충돌 처리를 학습시킨다.

## memory operation은 5개로 제한한다

처음부터 복잡한 differentiable database를 만들면 실패한다. operation은 작고 강하게 잡는다.

```text
READ:
  현재 chunk/event에 필요한 slot을 읽음

UPDATE:
  기존 slot을 새 정보로 조금 수정

ALLOCATE:
  새 entity/task/evidence/state slot 생성

PROTECT:
  사용자 지시, 원문 근거, 확정된 변수처럼 덮어쓰면 안 되는 정보 보호

MERGE:
  중복 slot을 합치거나 오래된 slot을 낮은 utility로 밀어냄
```

COMMIT은 모델 내부 operation이 아니라 runtime/system operation으로 둔다.

```text
COMMIT:
  workspace의 일부를 episodic ledger에 저장
```

이 분리가 중요하다. 모델은 workspace를 편집하고, 시스템은 ledger에 저장한다. 처음부터 ledger까지 end-to-end로 학습하려고 하면 연구가 너무 커진다.

## WELM block v0

제로베이스 첫 구현은 이 정도가 좋다.

```python
class WELMBlock(nn.Module):
    def __init__(self, d_model, n_slots, topk, window):
        super().__init__()
        self.local = LocalAttention(d_model, window=window)
        self.event = EventCompiler(d_model)
        self.router = TypedSlotRouter(d_model, n_slots, topk)
        self.editor = SlotEditor(d_model)
        self.fusion = GatedFusion(d_model)
        self.mlp = SwiGLU(d_model)

    def forward(self, x, workspace, boundaries):
        h = self.local(x)
        events = self.event(h, boundaries)
        reads, routes = self.router(events, workspace)
        workspace = self.editor(events, routes, workspace)
        h = self.fusion(h, reads, boundaries)
        h = h + self.mlp(norm(h))
        return h, workspace
```

여기서 `boundaries`는 chunk/document/line/function/QA boundary다. WELM은 token마다 memory를 편집하지 않는다. boundary 또는 chunk event 단위로만 memory를 편집한다.

## 실제 LLM이 되기 위한 tokenizer 전략

byte-only는 구조 연구에는 좋지만, 실제 지능 모델에는 비효율적이다. 목표가 "문장 생성과 지능"이면 tokenizer를 바꿔야 한다.

추천:

```text
v0 research:
  byte tokenizer 유지
  synthetic memory gate 검증

v1 LLM:
  32K~64K BPE/SentencePiece + byte fallback

v2 production:
  영어/코드/수학 중심 vocab
  document/chat/tool boundary special token
```

LUMA-byte와 WELM-subword는 분리해야 한다. byte 모델이 실패했다고 workspace memory가 실패한 것은 아니고, subword 모델이 잘 된다고 byte-patch가 증명된 것도 아니다.

## 학습 objective

WELM은 next-token loss만으로는 부족하다. 하지만 loss를 한 번에 너무 많이 넣으면 원인 분석이 안 된다.

v0 loss:

```text
L = L_next_token
  + 0.1 * L_answer_only
  + 0.05 * L_slot_future_use
  + 0.01 * L_slot_diversity
```

v1 loss:

```text
L = L_next_token
  + λ1 L_answer_only
  + λ2 L_future_use
  + λ3 L_retrieval_contrastive
  + λ4 L_grounding
  + λ5 L_overwrite_protection
```

v2 loss:

```text
L = base losses
  + verifier-guided RLVR
  + schema/tool correctness
  + citation/evidence consistency
  + calibrated abstention
```

처음부터 hallucination loss, proof loss, contradiction loss를 다 넣지 않는다. 먼저 slot이 필요한 정보를 기억하는지 증명한다.

## future-use supervision

이 연구의 가장 중요한 학습 신호는 future-use다.

정의:

> 현재 chunk/event가 미래 질문, 미래 코드 실행, 미래 수식 전개, 미래 대화 제약에 필요할 확률.

데이터 생성:

```text
document chunks: A B C D E
question Q is answered by chunk B

positive:
  B
  B의 entity definition
  B와 coreference되는 chunk

negative:
  같은 단어가 있지만 답에 필요 없는 chunk
  같은 문서의 무관 chunk
  distractor fact
```

모델은 모든 chunk를 기억하지 않고, 미래에 쓸 chunk를 workspace에 보존해야 한다. 이게 WELM이 Transformer/Mamba와 다르게 배워야 할 핵심이다.

## 데이터 커리큘럼

### Phase 0: memory circuit

```text
copy
phonebook
JSON field extraction
entity tracking
overwrite/update
protected memory
multi-hop recall
needle in distractors
```

목표:

```text
exact match >= 95%
slot usage collapse 없음
protected overwrite 실패율 낮음
```

### Phase 1: language base

```text
high-quality English documents
Wikipedia/books/science
code/docstrings
math explanations
memory curriculum 10~30%
```

목표:

```text
readable continuation
반복붕괴 없음
short QA 가능
slot diagnostics 안정
```

### Phase 2: instruction model

```text
answer-only SFT
definition QA
short reasoning
code explanation
tool/schema output
unknown handling
```

목표:

```text
대화형 답변
짧고 정확한 문장
기본 QA
형식 준수
```

### Phase 3: grounded model

```text
external ledger retrieval
evidence-grounded answer
citation consistency
conflict detection
abstention
```

목표:

```text
긴 문서 QA
근거 기반 답변
원문 pointer 복원
오래된 정보와 새 정보 충돌 처리
```

## 벤치마크 순서

WELM을 처음부터 MMLU로 때리면 안 된다. MMLU는 지식량, pretraining token 수, instruction format 영향을 크게 받는다.

먼저 봐야 할 것:

```text
1. synthetic memory exact match
2. passkey / needle retrieval
3. JSON exact extraction
4. code symbol tracking
5. multi-turn constraint retention
6. short natural QA
7. ARC/PIQA/HellaSwag
8. MMLU-Redux
```

Transformer/Mamba를 넘는다는 주장도 정확히 나눠야 한다.

```text
같은 파라미터
같은 token budget
같은 tokenizer
같은 FLOPs 또는 같은 wall-clock
같은 context length
같은 retrieval 사용 여부
```

이 조건을 맞추지 않으면 대체 주장으로 방어할 수 없다.

## GPU 효율 원칙

WELM이 망할 수 있는 가장 큰 이유는 GPU에서 느린 sparse 구조다.

반드시 지킬 것:

```text
chunk-level routing only
slot top-k는 batch matmul 기반
token-level scatter 금지
slot update는 contiguous tensor에 수행
workspace slot 수는 고정
decode-time shape 변화 금지
ledger retrieval은 prefill 또는 burst phase에서만 수행
```

나쁜 구조:

```text
for token:
  topk slot
  gather
  scatter update
```

좋은 구조:

```text
for chunk:
  event = compile(chunk)
  topk = matmul(event, slot_keys)
  update selected slots once
```

실제 빠른 모델은 이 차이에서 갈린다.

## incremental runtime

WELM이 실제 대화 모델이 되려면 runtime state가 있어야 한다.

```text
WELMState:
  local_window_cache
  chunk_buffer
  workspace_slots
  slot_age
  slot_utility
  ledger_query_state
```

필수 API:

```python
prefill(prompt) -> WELMState
decode_step(token, state) -> logits, new_state
commit(state) -> ledger_page
retrieve(query, ledger) -> evidence_chunks
```

이게 없으면 WELM은 긴 문맥 비용을 줄이는 모델이 아니라, 그냥 느린 custom Transformer가 된다.

## 모델 크기별 현실 목표

```text
50M~100M:
  memory circuit proof
  exact recall/edit benchmark

300M~500M:
  readable English
  short QA
  local reasoning
  strong synthetic memory

1B~2B:
  useful chat
  code/doc QA
  grounded retrieval
  MMLU-like evaluation

7B+:
  serious general intelligence target
  broad knowledge
  post-training and RLVR
```

작은 모델로도 WELM의 구조적 장점은 증명할 수 있다. 하지만 MMLU 60+ 같은 목표는 구조만으로 안 된다. 고품질 corpus, token budget, post-training, verifier가 필요하다.

## Transformer/Mamba를 이기는 방식

WELM이 전면적으로 모든 benchmark에서 이기기는 어렵다. 먼저 이겨야 할 전장은 따로 있다.

```text
WELM이 이겨야 할 곳:
- 긴 대화 상태 유지
- 오래된 제약조건 보존
- document QA에서 필요한 정보만 기억
- update/protect가 필요한 working memory
- retrieval evidence를 slot에 정리해 답변
- 같은 KV cache budget에서 긴 문맥 처리

Transformer가 여전히 강한 곳:
- 짧은 문맥 next-token modeling
- 대규모 dense pretraining
- 고도로 최적화된 GPU kernel 생태계

Mamba가 여전히 강한 곳:
- streaming sequence 처리
- constant-memory decode
- 긴 입력의 빠른 scan
```

따라서 WELM은 처음부터 "모든 것을 이긴다"가 아니라, **working-memory heavy task에서 압도적으로 이기고, 일반 언어 품질은 local Transformer core로 방어**해야 한다.

## 제로베이스 구현 로드맵

### Milestone 1: WELM-v0

```text
subword tokenizer는 아직 보류
byte tokenizer 유지
local mixer + typed workspace skeleton
memory exact-match eval
```

성공 기준:

```text
copy/phonebook/json >= 95%
slot collapse 없음
generate가 깨지지 않고 짧은 답 생성
```

### Milestone 2: WELM-v1

```text
32K tokenizer
local attention window 256/512
typed workspace slots
future-use loss
answer-only loss
```

성공 기준:

```text
readable continuation
short QA gate 통과
local Transformer baseline 대비 long recall 우위
```

### Milestone 3: WELM-v2

```text
external episodic ledger
retrieval query generator
evidence slot
grounded answer SFT
```

성공 기준:

```text
long document QA
source pointer recovery
conflict detection
retrieval 없을 때와 있을 때 성능 분리 평가
```

### Milestone 4: WELM-v3

```text
incremental decode
fixed-shape workspace cache
streaming chat shell
teacher/SFT/RLVR pipeline
```

성공 기준:

```text
실시간 대화
반복붕괴 없음
QA 가능
tokens/sec 측정 가능
```

## 최종 문장

WELM이 성공하려면 목표를 이렇게 잡아야 한다.

```text
Transformer처럼 모든 토큰을 다 보는 모델이 아니라,
Mamba처럼 모든 과거를 하나의 state에 압축하는 모델도 아니라,
짧은 문맥은 dense하게 이해하고,
미래에 쓸 정보는 typed workspace에 편집하고,
오래된 원문은 episodic ledger에서 근거로 다시 부르는 모델.
```

이 방향이면 "새롭다"와 "실제로 된다"를 동시에 노릴 수 있다. 핵심은 attention을 버리는 것이 아니라, **global attention을 작업기억 편집 문제로 바꾸는 것**이다.

---

# 2026-06-14 검증 결과와 현재 판정

## 요약

최근 로컬/원격 실험을 기준으로 보면, LUMA는 **runnable zero-base architecture prototype**이라는 의의가 분명히 있다. Transformer/Mamba 없이도 byte tokenizer, local mixer, sparse slot read/edit 구조만으로 loss가 안정적으로 내려가고 checkpoint 생성과 generation이 된다.

하지만 현재 결과만으로는 **슬롯 메모리가 실제 지능/기억 성능을 만든다**고 주장할 수 없다. loss 하락과 문장 생성은 확인됐지만, slot ablation에서 `normal`이 `no_slots`나 `random_slot_keys`를 압도하지 못했다.

따라서 현재 판정은 다음이다.

```text
1. 완전히 새로운 구조인가?        예.
2. 실제로 학습되는가?             예.
3. 문장 생성이 되는가?             예, 그러나 아직 품질 낮음.
4. 슬롯 메모리가 의미 있게 작동?   아직 미증명.
5. 기존 구조를 뛰어넘었다?         전혀 아님.
6. 논문화 가능성?                 있음. 단, memory proof 필요.
```

한 줄 결론:

**엄청난 출발은 맞지만, 아직 승리는 아니다. LUMA는 살아있는 새 아키텍처 prototype이고, 이제부터는 슬롯 메모리 회로를 강제로 증명해야 한다.**

## 확인된 성과

### 1. memory-circuit synthetic training

```text
runs/luma_memory_circuit_v1
loss: 5.59 -> 0.37
```

이건 중요하다. 기존 Transformer나 Mamba block 없이도 다음 구성만으로 학습이 안정적으로 진행됐다.

```text
byte tokenizer
local causal mixer
chunk event
sparse top-k slot read
erase/write/protect slot edit
LM head
```

즉 최소한 다음은 증명됐다.

```text
구조가 forward/backward 가능하다.
loss가 안정적으로 내려간다.
작은 문장/패턴 분포를 학습할 수 있다.
checkpoint 저장과 generation이 가능하다.
```

### 2. record SFT padding bug 수정

초기 `RecordTextDataset`는 짧은 record 뒤를 `eos_id`로 채웠다. 이 때문에 train loss가 비정상적으로 낮아지는 착시가 있었다. 모델이 문장을 잘 배운 것이 아니라 뒤쪽 EOS 반복을 맞추는 효과가 컸다.

수정 후:

```text
padding: eos_id -> pad_id
loss ignore 대상: pad_id
```

검증:

```text
수정 전 target_eos_frac ~= 0.418
수정 후 target_eos_frac ~= 0.004
수정 후 target_pad_frac ~= 0.414
```

따라서 `runs/luma_dmc8_record_sft`의 매우 낮은 loss는 폐기하고, `runs/luma_dmc8_record_sft_padfix` 이후 결과만 봐야 한다.

### 3. padfix SFT도 학습은 된다

```text
runs/luma_dmc8_record_sft_padfix
step 2000
loss ~= 0.200
lm_loss ~= 0.205
```

PAD 착시를 제거한 뒤에도 loss가 내려간 것은 긍정적이다. 다만 실제 generation은 아직 낮은 품질이다.

예시:

```text
Q: Who are you?
A: I am a ber finstraining repetitive or uns general resnors?
```

```text
Q: What is repetition collapse?
A: Collapse is when a modelue of field without is the time completed per unit of time.
```

즉 현재 SFT는 record 분포를 따라가는 능력은 보이지만, 일반적인 대화 모델 품질에는 도달하지 못했다.

## memory ablation 결과

### OOD memory QA

4지선다, chance는 약 25%.

```text
runs/luma_dmc8_proto:
normal            18.1%
no_slots          18.1%
random_slot_keys  20.8%

runs/luma_dmc8_sft:
normal            23.6%
no_slots          29.2%
random_slot_keys  25.0%

runs/luma_memory_circuit_v1:
normal            22.2%
no_slots          22.2%
random_slot_keys  22.2%
```

이 결과는 냉정하게 나쁘다. slot memory가 답을 고르는 데 실질적으로 도움이 된다는 증거가 없다. `no_slots`가 비슷하거나 더 좋은 경우도 있다.

### built-in memory QA

훈련 분포와 더 가까운 built-in memory format에서 다시 본 결과:

```text
Built-in memory QA, 160 cases, chance ~= 25%

runs/luma_memory_circuit_v1:
normal            28.1%
no_slots          21.9%
random_slot_keys  27.5%

runs/luma_dmc8_proto:
normal            24.4%
no_slots          25.0%
random_slot_keys  26.3%
```

여기서는 약한 긍정 신호가 있다. `memory_circuit_v1`에서 normal이 no-slots보다 높다.

특히 code recall:

```text
normal code acc   30.8%
no_slots code acc 10.3%
```

그러나 random slot keys가 전체 27.5%로 normal 28.1%와 거의 같다. 따라서 아직 "학습된 slot routing이 결정적이다"라고 말할 수 없다. 평균 margin도 음수라 정답 후보가 확실히 1등으로 뜨는 상태가 아니다.

## 현재 해석

현재 LUMA는 아마 다음 상태에 가깝다.

```text
local mixer + LM head가 분포를 학습하고 있다.
slot은 일부 신호를 제공할 수 있지만,
답을 보존/회수하는 명시적 memory object로는 아직 훈련되지 않았다.
```

가능한 원인:

```text
1. chunk_event = e.mean(dim=1)
   핵심 token 정보가 평균 pooling에서 뭉개진다.

2. slot target supervision이 없다.
   어떤 fact/entity를 어느 slot에 써야 하는지 직접 신호가 없다.

3. future-use loss가 없다.
   나중에 답에 필요한 정보만 보존하라는 압력이 없다.

4. slot routing collapse를 막는 metric/loss가 부족하다.
   일부 slot만 쓰거나, random key에도 비슷하게 동작할 수 있다.

5. generation은 전체 prefix 재계산이다.
   LUMA 철학인 persistent state runtime이 아직 없다.

6. byte tokenizer는 sample efficiency가 낮다.
   짧은 훈련으로 자연어 품질을 얻기 어렵다.
```

## 지금 더 큰 학습보다 먼저 해야 할 것

현재 필요한 것은 더 큰 학습이 아니라 **slot supervision**이다.

다음 목표는 train loss를 더 낮추는 것이 아니라, ablation에서 아래처럼 갈라지는 것이다.

```text
normal            70~90%+
no_slots          25~35%
random_slot_keys  25~40%
```

이때야 "LUMA memory 구조가 실제로 작동한다"고 말할 수 있다.

## 다음 실험 우선순위

### 1. answer-span / fact-slot supervision

정답이 들어 있는 fact를 특정 slot에 쓰게 하는 auxiliary loss가 필요하다.

예:

```text
Mina owns the blue key.
Question: What object belongs to Mina?
Answer: blue key
```

훈련 신호:

```text
entity = Mina
field = object
value = blue key
target slot type = entity/object slot
```

slot read가 answer candidate를 직접 복원하도록 보조 head를 둔다.

### 2. same-entity consistency

같은 entity는 같은 slot family로 가야 한다.

```text
Mina owns blue key.
Mina should go to seoul.
Mina's code is AX-123.
```

이 세 정보가 완전히 다른 slot에 흩어지면 QA가 어렵다. 최소한 같은 entity cluster나 linked slot group으로 묶여야 한다.

### 3. future-use loss

나중 질문에 필요한 chunk/event를 positive로 두고, distractor를 negative로 둔다.

```text
positive:
  answer fact
  entity definition
  variable definition

negative:
  같은 문서의 무관 fact
  비슷한 단어만 가진 distractor
```

목표:

```text
answer에 필요한 event는 높은 write_score
distractor는 낮은 write_score
```

### 4. better event compiler

현재 평균 pooling은 약하다.

대체:

```text
mean + max pooling
gated pooling
boundary-aware pooling
entity/key/value trigger pooling
```

attention을 전역으로 붙이라는 뜻이 아니다. chunk 내부에서 어떤 token이 memory-worthy인지 고르는 작은 local pooling이 필요하다.

### 5. slot usage diagnostics

학습 중 반드시 아래를 기록한다.

```text
slot entropy
top-k slot histogram
slot update frequency
slot overwrite rate
slot confidence mean/std
slot utility mean/std
normal vs no_slots online eval
random_slot_keys online eval
```

loss만 보면 안 된다. LUMA의 핵심은 slot이므로, slot이 실제로 쓰이는지 계속 봐야 한다.

### 6. exact-match eval first

다음 checkpoint 승격 기준:

```text
built-in memory QA normal >= 70%
OOD memory QA normal >= 50%
normal - no_slots >= 30 percentage points
normal - random_slot_keys >= 25 percentage points
generation does not collapse
```

MMLU, chat quality, long-context QA는 그 다음이다.

## 연구 주장 업데이트

지금 주장할 수 있는 것:

```text
LUMA is a runnable attention-free slot-edit language-model prototype.
It trains stably from zero on synthetic and record datasets.
It can generate text, although quality is still low.
```

아직 주장하면 안 되는 것:

```text
LUMA memory slots solve long-context recall.
LUMA beats Transformer or Mamba.
LUMA has proven grounded reasoning.
LUMA slots are learned meaningful memory objects.
```

다음 논문/문서에서 방어 가능한 claim은 이렇게 좁혀야 한다.

```text
Claim 1:
Slot-edit language models can be trained end-to-end without attention.

Claim 2:
Without explicit supervision, slot memories may not become useful memory objects.

Claim 3:
Future-use and fact-slot supervision are necessary to turn slot state into working memory.
```

이게 현재 결과에 가장 정직하고, 다음 실험을 가장 잘 이끄는 표현이다.
