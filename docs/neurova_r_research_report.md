# Neurova-R Research Report

이 문서는 SaneFlow 위에 올린 Neurova-R 계열 연구의 현재 상태를 정리한다.
목표는 구현한 것과 구현하지 않은 것을 섞지 않고, 실제 실험 결과 기준으로
다음 결정을 내리기 위한 단일 기록을 남기는 것이다.

## Executive Summary

Neurova-R은 SaneFlow의 순수 recurrent baseline을 reasoning-oriented hybrid로
확장하려는 연구 트랙이다. 현재 구현은 논문급 최종 구조가 아니라, 다음 아이디어가
실제로 학습 가능한지 확인하는 최소 검증체다.

```text
MultiKernel SyntaxMix
+ DeltaMatrix associative state
+ sparse local attention island with RoPE
+ thought slots
+ landmark chunk memory
+ Muon training
+ cache-step parity path
```

가장 중요한 판정은 다음과 같다.

1. `v2_fixed_sparse` 100M급은 현재 가장 실전적인 메인 후보다.
2. Neurova-R full 100M급은 현재 PyTorch loop 구현으로는 너무 느리거나 OOM이 난다.
3. Neurova-R small은 GPU1에서 학습 가능한 검증 트랙으로 낮춰서 돌리고 있다.
4. 최신 reasoning gate는 아직 실패다. 현재 모델은 base continuation 학습 단계이지
   QA/reasoning 모델이 아니다.
5. DeltaMatrix/KDA/MLA/mHC/MTP 등은 이름만 붙이면 안 된다. 논문급 구현과 현재
   간이 구현을 명확히 분리해야 한다.

## Implemented

### SaneFlow Config Extensions

`SaneFlowConfig`에 Neurova-R 실험용 옵션을 추가했다.

```text
thought_slots
landmark_interval
landmark_chunk
landmark_max
```

훈련 스크립트 `scripts/saneflow_train.py`도 동일 옵션을 받도록 확장했다.
checkpoint resume 시 구조가 달라졌는데 실수로 이어 학습하지 않도록 config mismatch
검사에도 해당 필드를 포함했다.

### v2_fixed State Mixer

`GatedStateMixer v2`에서 계산만 하고 쓰지 않던 `forget_delta` 경로를 제거한
`v2_fixed`를 추가했다.

의도:

```text
value, erase_delta, write_gate, out_gate
```

로 state update 의미를 명확히 하고, 불필요한 projection/연산을 줄인 순수
SaneFlow baseline을 확보하는 것이다.

### DeltaMatrix Associative State

`DeltaMatrixStateMixer`를 추가했다.

현재 형태:

```text
q, k, v, erase, write = proj(x)
state_t = erase * state_{t-1} + write * outer(k, v)
read_t = q @ state_t
```

이 구현의 목적은 key-value binding 가능성을 빠르게 검증하는 것이다. Kimi Linear /
KDA / DPLR kernel 급 구현이 아니다.

현재 한계:

```text
forward가 Python for t in range(length)
activation 보존 비용이 큼
긴 sequence와 큰 batch에서 OOM 또는 낮은 GPU utilization 발생
```

### Sparse Local Attention Island

`SparseAttentionIsland`를 추가/개선했다.

현재 구현:

```text
QKV local attention
RoPE 적용
chunked local window matmul
cache-step path
```

의도는 pure recurrent 모델의 copy/retrieval 약점을 최소 비용으로 보완하는 것이다.

현재 한계:

```text
MLA가 아님
compressed KV가 아님
Triton/fused banded kernel이 아님
relative bias/positional 분리 없음
```

### ThoughtSlotMixer

sequence별 latent working memory를 주기 위해 `ThoughtSlotMixer`를 추가했다.

현재 구현:

```text
slots: [B, M, D]
tokens read from previous slots
tokens write/update slots
cache-step path 있음
```

현재 Neurova-R small에서는 `thought_slots=8`로 켰다.

### LandmarkMemory

긴 입력을 chunk summary로 압축하는 `LandmarkMemory`를 추가했다.

현재 구현:

```text
landmark_chunk = 64
landmark_max = 64
completed chunk summaries only
token reads existing landmarks
cache-step path 있음
```

현재 Neurova-R small에서는 `landmark_interval=4`로 일부 block에만 켰다.

### Cache-Step Path

다음 경로에 step/cache API를 연결했다.

```text
SaneFlowLM.forward_step()
SaneFlowBlock.step()
GatedStateMixer.step()
DeltaMatrixStateMixer.step()
SparseAttentionIsland.step()
ThoughtSlotMixer.step()
LandmarkMemory.step()
```

작은 config에서 full-forward/cache-step parity를 확인했다. 이 경로는 대화형 추론과
향후 recurrent serving의 기반이다.

### Evaluation Tooling

`scripts/saneflow_reasoning_gate.py`를 추가했다.

현재 probe:

```text
apples 2 + 3
larger 9 vs 12
causal pushed cup
Paris -> France
2,4,6,8 pattern
Tom/Sam/Leo ordering
copy AX-917
```

출력:

```text
pass_rate
avg_tok_s
generated outputs
repeated_4gram_max
```

이 gate는 모델이 진짜 QA/reasoning을 시작했는지 보는 최소 실험이다.

### dmc9 Research Control Script

`scripts/saneflow_researchctl_dmc9.sh`를 추가했다.

지원 명령:

```bash
scripts/saneflow_researchctl_dmc9.sh status
scripts/saneflow_researchctl_dmc9.sh start-neurova-r-small
scripts/saneflow_researchctl_dmc9.sh eval-latest
```

긴 수동 SSH 명령으로 인한 quoting/PID/log 파편화를 줄이기 위한 관리 스크립트다.

## Current Experiments

### dmc9 GPU0: Main Candidate

현재 가장 실전적인 메인 후보는 `v2_fixed_sparse`다.

```text
run:
runs/saneflow_100m_research_v4/v2_fixed_sparse_100m_muon_b44_cuda

architecture:
v2_fixed state mixer
MultiKernel SyntaxMix v2, kernels 3/7/15
sparse local attention island every 4 blocks
attention window 64
d_model 672
layers 12
heads 8
d_ff 2016
seq_len 384
batch 44
optimizer Muon
```

최근 확인 상태:

```text
step around 1210
valid_loss around 4.72
GPU0 utilization near 100%
VRAM around 13GB / 16GB
```

판정:

```text
현재 가장 효율적으로 학습 중인 100M급 후보
Neurova-R full보다 훨씬 실전적
```

### dmc9 GPU1: Neurova-R Small

100M full Neurova-R이 OOM/저효율이라 small proof track으로 낮췄다.

```text
run:
runs/saneflow_neurova_r_v1_small/delta_kv_sparse_rope_thought_landmark_d384_l8_h32_b32_ga2_s256

architecture:
DeltaMatrix associative state
Sparse local attention + RoPE
Thought slots = 8
Landmark memory every 4 blocks
d_model 384
d_embed 256
layers 8
heads 32
d_ff 1152
seq_len 256
batch 32
grad_accum 2
optimizer Muon
```

왜 `heads=32`인가:

```text
DeltaMatrix state size는 head_dim^2에 민감함.
d_model 384, heads 16은 head_dim 24라 메모리/속도 부담이 큼.
heads 32는 head_dim 12로 state matrix를 작게 만들어 OOM을 피함.
```

최근 확인 상태:

```text
step 10 통과
VRAM around 10GB
GPU utilization around 20~35%
```

판정:

```text
학습 가능성은 확인
하지만 Python recurrent loop 병목으로 메인 후보라고 보기엔 이르다
```

### dmc8 Reference

기존 SaneFlow reference도 유지 중이다.

```text
run:
runs/saneflow_fineweb_edu_base_v3_100m_muon_mem

architecture:
SaneFlow reference
state_mixer_version v2
attention_interval 0
seq_len 384
optimizer Muon
```

최근 확인 상태:

```text
step around 1550
valid_loss around 4.48
```

판정:

```text
현 시점에서 language continuation은 이쪽이 가장 안정적
하지만 reasoning gate는 아직 실패
```

## Reasoning Gate Results

최신 checkpoint로 `scripts/saneflow_reasoning_gate.py`를 실행했다.

### dmc9 v2_fixed_sparse

```text
checkpoint: latest.pt
step: 1000
passed: 0 / 7
avg_tok_s: about 80
```

### dmc8 reference

```text
checkpoint: latest.pt
step: 1500
passed: 0 / 7
avg_tok_s: about 109
```

결론:

```text
현재 모델들은 영어 continuation은 배우고 있지만,
질문에 답하는 instruction/QA/reasoning 모델은 아직 아니다.
```

raw document continuation만 계속하면 자연스러운 문장 확률은 내려가지만,
MCQ, arithmetic, copy, variable binding, Work -> Answer reasoning은 자동으로 생긴다고
가정하면 안 된다.

## Tried And Rejected In This Round

### 100M Full Neurova-R

시도 방향:

```text
d_model 512
layers 10
heads 16
DeltaMatrix
sparse attention
thought slots
landmark memory
seq_len 384
```

결과:

```text
no activation checkpointing: OOM
activation checkpointing: step은 가능하지만 너무 느리고 GPU utilization 낮음
```

판정:

```text
현재 PyTorch loop 구현으로는 100M full Neurova-R을 메인 학습으로 밀면 비효율적
```

### 384x8 heads=16 Full Feature

시도 방향:

```text
d_model 384
layers 8
heads 16
seq_len 384 or 256
thought/landmark enabled
```

결과:

```text
batch 32 no-checkpoint: OOM
batch 16 activation checkpointing: step 가능하지만 GPU util 낮고 느림
```

판정:

```text
heads=16은 DeltaMatrix state size가 아직 큼
```

## Not Implemented Yet

아래 항목은 아직 구현하지 않았다. 문서나 코드에서 구현 완료처럼 주장하면 안 된다.

### Kimi-Style KDA / DPLR / Chunkwise Kernel

현재 DeltaMatrix는 단순 outer-product associative memory다. Kimi Linear/KDA급
finite-state memory update가 아니다.

안 한 이유:

```text
별도 수식/커널 설계가 필요함
현재 최소 구현의 효과부터 확인해야 함
```

### DeltaMatrix Fused / Parallel Scan / Triton Kernel

안 한 이유:

```text
현재 구조가 평가에서 이득을 내는지 아직 미확정
하지만 속도 병목의 핵심이므로 다음 고우선순위 후보
```

### MLA / Compressed KV

현재 local attention은 일반 QKV다.

안 한 이유:

```text
projection, cache, positional 처리까지 바뀌는 구조 변경
local attention island 자체가 유효한지 먼저 확인해야 함
```

### Fused Banded Window Attention

현재는 PyTorch chunked local matmul이다.

안 한 이유:

```text
attention island가 메인 대비 이기는지 먼저 판정 필요
Triton kernel은 이후 최적화 단계
```

### mHC-Lite Multi-Stream Residual

안 한 이유:

```text
block residual topology를 크게 바꿈
DeltaMatrix/thought/landmark 효과가 분리되기 전에는 원인 분석이 어려움
```

### Dynamic Compute / Thought Loop

안 한 이유:

```text
단일 forward 모델이 아직 reasoning gate를 통과하지 못함
base capability 없이 test-time compute만 늘리면 비용만 늘 가능성이 큼
```

### MTP / Speculative Head

안 한 이유:

```text
단일-token generation 품질이 먼저 안정돼야 함
MTP는 품질 안정 후 속도 최적화 단계
```

### Tiny MoE

안 한 이유:

```text
이전 MoE 실험에서 collapse/router 리스크가 컸음
현재는 dense/recurrent-hybrid baseline 안정화가 우선
```

### Reasoning Curriculum / Verifier / Work -> Answer SFT/RL

아직 본격 적용하지 않았다.

안 한 이유:

```text
현재는 base LM loss를 낮추는 단계
QA gate 0/7이라 바로 SFT/RL을 얹으면 템플릿 암기와 forgetting 위험이 큼
```

하지만 다음 단계에서는 반드시 필요하다. reasoning은 architecture만으로 생기지 않는다.

## Current Technical Diagnosis

### What Works

```text
v2_fixed_sparse trains efficiently at 100M scale.
SaneFlow reference continues to reduce validation loss.
Neurova-R components are wired and small full-feature track can step.
Cache-step paths exist.
Reasoning gate now gives a hard, repeatable failure signal.
```

### What Does Not Work Yet

```text
Neurova-R full 100M is not efficient with current DeltaMatrix Python loop.
Reasoning / QA is not present yet.
DeltaMatrix is not a production-speed state kernel.
Thought and landmark memory are not yet proven useful by ablation.
```

### Root Cause Of Neurova-R Slowness

The bottleneck is not only parameter count. It is the current update structure:

```text
for each token:
  update H small matrix states
  keep activations for backward
  optionally recompute under checkpointing
```

This creates poor GPU efficiency compared with dense matmul-heavy blocks.

Therefore, VRAM filling alone is not the correct objective. The correct objective is:

```text
use enough VRAM to keep large effective batch,
but avoid Python recurrent loops that leave SM utilization low.
```

## Recommended Next Actions

### 1. Let Current Runs Reach Checkpoints

Keep:

```text
dmc9 GPU0 v2_fixed_sparse
dmc9 GPU1 Neurova-R small
dmc8 reference
```

Evaluate at every saved checkpoint with:

```bash
ssh ml-dmc9 'cd ~/workspace/neurova && scripts/saneflow_researchctl_dmc9.sh eval-latest'
```

### 2. Compare With One Table

For each candidate:

```text
valid_loss
reasoning_gate pass_rate
generation examples
tok/s
VRAM
GPU utilization
repetition metrics
```

Do not promote by loss alone.

### 3. Add A Small Supervised Reasoning Dataset

Once base continuation is less chaotic, add clean verifiable tasks:

```text
copy exact match
arithmetic
comparison
entity binding
code/identifier recall
short Work -> Answer
MCQ with generated non-benchmark questions
```

This should be separate from raw document continuation and mixed carefully.

### 4. If Neurova-R Small Shows Signal, Optimize DeltaMatrix

Only if Neurova-R small improves reasoning/copy metrics:

```text
implement chunked scan
then Triton/fused DeltaMatrix update
then larger 100M variant
```

If it does not show signal, do not spend kernel time on it.

### 5. If Sparse Island Keeps Winning, Improve Attention Island

Next attention work should be:

```text
true banded/window kernel
larger window 128/256 ablation
compressed KV / MLA-like cache only after QKV baseline wins
```

## Current Verdict

Neurova-R is a meaningful research track, but it is not yet the main model.

The main trainable path today is:

```text
SaneFlow v2_fixed_sparse 100M
```

The research path is:

```text
Neurova-R small
  DeltaMatrix
  sparse local attention
  thought slots
  landmark memory
```

The promotion rule is strict:

```text
Do not promote Neurova-R unless it improves reasoning/copy/generation gates
without unacceptable speed loss.
```

The next real breakthrough is not another label. It is either:

```text
1. evidence that DeltaMatrix/thought/landmark improves reasoning metrics, or
2. evidence that sparse island alone is the best ROI and should become the main architecture.
```
