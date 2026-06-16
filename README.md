# Neurova Workspace

Neurova는 하나의 모델만 담은 저장소가 아니다. 저사양 GPU부터 대형 GPU까지
여러 환경에서 언어 모델을 실제로 서빙하기 위한 런타임, 자체 언어 모델
연구, Mamba-3 연구, 과거 메모리 실험, 배포 런북이 함께 들어 있는
작업공간이다.

현재 메인 프로젝트는 `langburst/`이다. 나머지 프로젝트는 LangBurst와
직접 이어질 수 있는 연구 재료이거나, 지금은 중단됐지만 왜 시도했는지
보존해야 하는 실험 기록이다.

## 큰 그림

| 프로젝트 | 한 줄 개념 | 현재 의미 |
| --- | --- | --- |
| `langburst/` | vLLM 같은 범용 저비트/stateful 모델 서빙 엔진 | 메인 프로젝트 |
| `saneflow/` | Neurova 자체 LM을 학습해보는 연구 라인 | 별도 학습 연구 |
| `neuromamba/` | Mamba-3 계열을 Neurova 모델 후보로 검증하는 연구 라인 | 별도 아키텍처 연구 |
| `deploy_exl3/` | Qwen EXL3/TabbyAPI 실서빙 운영 기록 | 배포 런북 |
| `luma/` | 외부 슬롯 메모리를 쓰는 attention-free LM 실험 | 보관된 연구 |
| `neurova/` | Qwen + embedding retrieval 개인 기억 CLI | 레거시 CLI |

루트는 작업공간 인덱스와 공통 런처만 담당한다. 새 문서, 스크립트, 설정,
데이터, 체크포인트, 논문 자료는 각 프로젝트 안에 둔다.

## `langburst/`: 메인 서빙 엔진

LangBurst는 이 저장소의 중심 프로젝트다. 목표는 16GB RTX 4080/4090급
GPU 하나에 갇힌 특화 런타임이 아니라, vLLM처럼 여러 모델 family를 같은
서버/스케줄러/런타임 계약 위에서 서빙하는 범용 엔진이다. 다만 방향은
“대형 서버 GPU 전용”이 아니라 저사양 GPU에서도 쓸 수 있는 저비트,
stateful, 긴 문맥, 멀티 요청 엔진이다.

핵심 아이디어는 “모델별 수학은 adapter가 담당하고, 서버와 디코드 루프는
공통 런타임이 담당한다”는 것이다. Qwen 전용 스크립트 더미가 아니라,
adapter registry, runtime engine, engine manager, request scheduler,
CUDA kernel, OpenAI-compatible server를 갖춘 제품형 런타임으로 정리되고
있다.

Qwen3.6-27B Q4 Marlin 경로는 현재 champion adapter이자 실측 기준일 뿐,
LangBurst의 정체성 자체가 Qwen 전용 또는 16GB 전용이라는 뜻은 아니다.
`hf-auto`/Gemma 계열 conformance path처럼 새 모델 family를 붙이는 경계가
이미 분리돼 있고, production adapter는 같은 `RuntimeEngine`과
`EngineManager`를 재사용해야 한다.

LangBurst가 실제로 담고 있는 것:

```text
langburst/langburst/core/
  RuntimeEngine: prefill, decode, sampling, state pool, generation contract
  EngineManager: lazy model load, LRU unload, model residency, health/status
  AdmissionController: active/queued request 제한과 timeout/reject 카운터
  ContinuousBatchScheduler: vLLM-style batching으로 가기 위한 scheduler 경계
  RuntimeFeatures/RuntimePlan: 기능 요청과 adapter capability를 합쳐 실행 계획 결정
  KVBlockTable / BatchStateStore: paged KV와 state arena를 위한 자원 경계

langburst/langburst/adapters/
  qwen36: Qwen3.6 hybrid GDN 모델 adapter
  qwen36-a3b: Qwen3.6 A3B 계열 adapter
  hf-auto / gemma4: 새 모델 family를 붙이기 위한 Transformers-backed conformance path
  qwen36_impl/: Qwen 전용 config/model/state 구현

langburst/csrc/
  Marlin W4A16 GEMM
  rowwise low-bit GEMV fallback
  RMSNorm
  Qwen Gated DeltaNet recurrence
  attention decode
  GPU sampling helpers

langburst/server.py
  OpenAI-compatible /v1/chat/completions
  SSE streaming
  /v1/langburst/health, /v1/langburst/models, /v1/langburst/features

langburst/tests/
  adapter/runtime/server/scheduler/state/cuda/speculation 단위 검증
```

LangBurst의 주요 기능 축:

```text
범용 서빙 구조
  adapter registry
  model-independent RuntimeEngine
  lazy multi-model load
  LRU unload
  declarative models-json
  OpenAI-compatible API

저사양 GPU / 메모리 스케일링
  low-bit checkpoint format
  Q4 Marlin fused projection path
  rowwise 2-8bit groupwise weights
  CPU/offload fallback for non-fitting checkpoints
  VRAM reserve based load admission
  prompt/generation token admission

stateful / 긴 문맥 기능
  kv_window_policy: error, shift, ring
  ring KV
  pooled DecodeState
  snapshots
  boundary_decay
  infinite streaming gate
  episodic memory gate
  TTT sidecar gate

서빙 성능 기능
  chunked block prefill
  GPU greedy sampling
  request admission and queue limits
  continuous batching baseline
  slot-indexed state arena
  paged KV / block table
  batch-state CUDA kernels
  native MTP/NEXTN speculative decoding with adaptive fallback
  CUDA Graph bucket scaffold

운영 안정성
  health/status endpoints
  model runtime status
  OOM 시 runtime pool cleanup
  request cancellation cleanup
  queue timeout/reject counters
```

아직 vLLM급 완성품이라고 말하면 안 된다. 현재 정체성은 “vLLM-class를
목표로 하는 범용 저비트/stateful 서빙 엔진”이다. Qwen3.6은 첫 번째 강한
adapter이고, continuous batching, paged/ring KV, state arena, speculative
decode, multi-model residency, 저사양 GPU 대응은 모두 LangBurst 본체의
핵심 방향이다.

대표 명령:

```bash
cd langburst
LANGBURST_SKIP_CUDA_EXT=1 python -m pip install -e .
./scripts/cuda_compile_and_test.sh
langburst-qwen-quantize /path/to/hf-model /path/to/converted-runtime-model --bits 4 --group-size 128
langburst-qwen-audit /path/to/converted-runtime-model --hf-model /path/to/hf-model
langburst-chat --adapter qwen36 --hf-model /path/to/hf-model --qb-model /path/to/qb-model --prompt "안녕"
langburst-server --adapter qwen36 --hf-model /path/to/hf-model --qb-model /path/to/qb-model --port 8008
```

먼저 읽을 문서:

```text
langburst/README.md
langburst/docs/DESIGN.md
langburst/docs/ADAPTER_ARCHITECTURE.md
langburst/docs/RUNTIME_FEATURES.md
langburst/docs/PERFORMANCE_LOG.md
langburst/docs/VLLM_PARITY_TODO.md
```

## `saneflow/`: 자체 LM 학습 연구

SaneFlow는 LangBurst처럼 기존 대형 모델을 서빙하는 런타임이 아니라,
Neurova 자체 언어 모델을 직접 학습해보는 연구 라인이다. 여기에는 모델
정의, 데이터 mix, tokenizer, 학습 profile, quality gate, autoresearch loop가
들어 있다.

개념적으로 SaneFlow는 “작고 통제 가능한 자체 LM을 만들어 언어 prior,
state mixer, dense Transformer baseline, ChatML SFT 품질 게이트를 연구하는
라인”이다. 루트의 `./neurova.sh`가 인자 없이 실행될 때 SaneFlow chat으로
들어가지만, 작업공간의 메인 프로젝트는 LangBurst다.

SaneFlow가 실제로 담고 있는 것:

```text
saneflow/model.py
  SaneFlowLM
  RMSNorm
  causal depthwise/multi-kernel syntax mix
  gated state mixer
  optional local attention / dense Transformer-style components

saneflow/profile_registry.py
  학습 profile의 single source of truth loader
  model size, data path, tokenizer, optimizer, dtype, checkpoint path를 config에서 생성

saneflow/data.py
  JSONL text stream -> token stream dataset
  causal loss와 ChatML assistant-only loss mask 지원
  token cache

saneflow/configs/
  active profiles
  research program
  practical pretrain mix
  ChatML SFT recipe
  dense scale ladder

saneflow/scripts/
  corpus prepare/build
  tokenizer train
  train/eval/generate/chat
  quality/reasoning gates
  dmc8/dmc9 fleet control
```

현재 연구 축은 dmc8 speak/chat, dmc9 practical base, dmc9 dense Transformer
baseline으로 나뉜다. 학습을 추가할 때는 스크립트에 모델 라인을 박지 말고
`saneflow/configs/saneflow_profiles.json`과
`saneflow/configs/saneflow_research_program.json`에 추가한다.

먼저 읽을 문서:

```text
saneflow/docs/saneflow_standardization.md
saneflow/docs/training_data_master_plan.md
```

## `neuromamba/`: Mamba-3 후보 연구

NeuroMamba는 Mamba-3 계열을 Neurova 모델 후보로 검증하기 위한 연구
프로젝트다. 이전 이름인 `mamba3_kr`은 버렸고, 현재 이름은 `neuromamba`다.
공식 `state-spaces/mamba` 코드는 수정 대상 메인 코드가 아니라
`neuromamba/vendor/mamba` 아래의 벤더 의존성으로 둔다.

개념적으로 NeuroMamba는 “Transformer가 아닌 Mamba-3 SISO/MIMO 계열로
언어 모델을 만들 수 있는지, 16GB 환경에서 어떤 크기와 커널 조합이 실제로
forward/backward/decode를 통과하는지 검증하는 실험장”이다.

NeuroMamba가 실제로 담고 있는 것:

```text
neuromamba/model.py
  NeuroMambaConfig
  official Mamba-3 MambaLMHeadModel wrapper
  transformer-tiny fallback
  SISO/MIMO, MoE, attention-hybrid, state-edit, meta-token preset support

neuromamba/presets.py
  mimo-r4-tiny
  16GB 120M/180M candidates
  MoE 260M-2.9B candidates
  paper-scale 180M/440M/880M/1.5B candidates
  siso/hybrid/transformer tiny candidates

neuromamba/state.py
  recurrent state save/load
  state summary metadata

neuromamba/cli.py
  model-info, train, eval, bench, state-prefill, quality gates, chat/server paths

neuromamba/scripts/
  governed corpus build
  clean document continuation corpus
  16GB train target probing
  Mamba-3 gate suite
  chat/speak SFT generation
  autonomous research autopilot
  MoE 100M-token block management
```

이 프로젝트는 완성된 한국어 모델이 아니다. 지금의 의미는 tiny smoke
training, Mamba-3 kernel/runtime 검증, recurrent state 저장/복구 검증,
continued pretraining과 instruction tuning 연구 기반이다.

먼저 읽을 문서:

```text
neuromamba/docs/MAMBA3_MASTER_PLAN.md
neuromamba/docs/NEUROMAMBA.md
neuromamba/docs/MAMBA3_TODO.md
neuromamba/docs/MAMBA3_KERNEL_BACKWARD_NOTES.md
neuromamba/docs/MAMBA3_OPTIMIZATION_RESEARCH.md
```

## `deploy_exl3/`: 실서빙 런북

`deploy_exl3/`는 모델 코드 프로젝트가 아니라 운영 기록이다. ml-dmc8에서
Qwen3.6-27B를 EXL3 + DFlash + TabbyAPI로 실제 OpenAI-compatible 서버로
띄우면서 생긴 설치, 패치, 장애 대응 절차를 보관한다.

개념적으로 이 폴더는 “LangBurst 이전/병행의 실전 배포 레퍼런스”다.
어떤 모델과 quantization이 16GB에서 돌아갔는지, TabbyAPI/ExLlamaV3에서
어떤 문제가 있었고 어떻게 고쳤는지 확인하는 곳이다.

담고 있는 내용:

```text
README_TABBYAPI_EXL3.md
  Qwen3.6-27B EXL3 3.08bpw
  DFlash draft model
  100K context 근처 운영값
  KV cache quantization
  ExLlamaV3 / TabbyAPI 문제 해결 기록

CODEX_QWEN_RUNBOOK.md
  Codex/CLI proxy 실사용 검증
  no-think template
  tool-call 종료 문제
  한글 byte-fallback streaming 문제 패치

templates/
  Qwen chat template
```

## `luma/`: 슬롯 메모리 LM 보관 연구

LUMA는 현재 주력 경로가 아니라 보관된 연구다. 하지만 단순 폐기물이
아니라, “LM 내부에 외부 슬롯 메모리를 붙이면 실제로 기억을 쓰는가”를
검증하려던 프로토타입이다.

개념적으로 LUMA는 attention-free 언어 모델에 sparse persistent memory
slots를 붙인 실험이다. 모델은 토큰 이벤트를 국소적으로 섞고, chunk event를
만들고, 관련 slot을 top-k로 읽고, gated write/erase/protect로 slot 값을
수정한다. 성공 기준은 loss가 낮은지가 아니라 slot을 끄거나 slot key를
무작위화했을 때 memory QA가 확실히 나빠지는지다.

LUMA가 실제로 담고 있는 것:

```text
luma/model.py
  LUMAConfig
  LUMALM
  LUMABlock
  SlotState
  local depthwise mixer
  optional local chunk attention
  top-k slot read/write
  erase/write/protect gates
  slot confidence/utility/age/lock diagnostics

luma/tokenizer.py
  ByteTokenizer: 259-token raw UTF-8 byte tokenizer
  QwenTokenizer: luma/tokenizers/qwen35 기반 BBPE route
  AdaptiveBytePatchTokenizer: byte-preserving patch tokenizer
  byte span metadata for raw evidence tracking

luma/ledger.py
  append-only JSONL evidence ledger scaffold
  raw byte evidence pointer

luma/eval_memory.py, eval_gate.py, eval_chat_sanity.py, eval_natural_sanity.py
  memory proof, chat sanity, natural continuation sanity

luma/data/
  raw continuation, ChatML SFT, memory curriculum, reasoning/dialogue stage data
  tracked small seed corpora and ignored generated corpora

luma/papers/luma_sane_lm/
  LUMA 설계 당시 참고한 BLT, Mamba3, Gated DeltaNet, Titans, RWKV 등 논문 묶음
```

실행은 명시적으로만 한다.

```bash
NEUROVA_ALLOW_LUMA=1 ./neurova.sh luma
python3 -m luma.train --help
python3 -m luma.eval_memory --help
```

## `neurova/`: V6 개인 기억 CLI

`neurova/`는 V6 Pure Embedding Memory CLI의 레거시 프로젝트다. LangBurst나
SaneFlow의 하위 모듈이 아니라, “기존 Qwen 모델에 학습 없이 개인 기억을
붙이는 가장 단순한 방법”을 실험한 독립 CLI다.

개념적으로 V6는 문법 규칙, 질문 유형 규칙, 1인칭/3인칭 변환 같은
하드코딩을 제거하고, 모델의 `embed_tokens` 평균 embedding으로 raw text를
USearch에 저장/검색한다. 검색된 원문 기억은 system prompt에 붙고, 실제
귀속과 표현은 Qwen 모델이 처리한다.

V6가 실제로 담고 있는 것:

```text
neurova/v6.py
  Qwen3.5-4B loader
  bf16/4bit mode
  TextIteratorStreamer 기반 streaming CLI
  Memory class
  MemSlot raw text storage
  embed_tokens mean pooling
  USearch cosine index
  per-user ~/.neurova_v6/users/<user>/ namespace
  /think, /nothink, /effort, /user, /clear, /status, remember:

neurova/docs/ARCHITECTURE_v6.md
  V6 구조 설명

neurova/scripts/deploy_v6.sh
  ml-dmc8 배포 스크립트
```

의도적으로 하지 않는 것:

```text
언어별 규칙
인칭 변환
topic keyword rule
question-type handler
template extractor
학습 루프
```

## 루트와 아티팩트 원칙

루트에는 작업공간 인덱스와 공통 런처만 둔다.

```text
README.md
neurova.sh
.gitignore
```

프로젝트별 문서, 스크립트, 설정, 데이터, 논문, 토크나이저, run 결과는
각 프로젝트 아래에 둔다.

```text
langburst/docs/
saneflow/configs/
neuromamba/scripts/
luma/data/
neurova/docs/
```

대용량 데이터, 다운로드 토크나이저, 논문 PDF, 학습 결과, 평가 결과는
기본적으로 git 추적 대상이 아니다. 예외는 재현에 필요한 작은 seed corpus나
설정 파일뿐이다.

## 기본 확인

정리나 이동 후 최소한 다음을 확인한다.

```bash
bash -n neurova.sh
find luma/scripts saneflow/scripts neuromamba/scripts neurova/scripts -maxdepth 1 -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
python -m compileall -q langburst luma saneflow neuromamba neurova
python saneflow/scripts/saneflow_run.py list
python -m neuromamba.cli model-info --mode mimo-r4-tiny --tokenizer byte --device cpu
```

GPU, gated tokenizer, 원격 호스트, 대형 checkpoint가 필요한 검증은 해당
환경에서 별도로 수행한다.
