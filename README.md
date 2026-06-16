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
| `langburst/` | external serving engine 같은 범용 저비트/stateful 모델 서빙 엔진 | 메인 프로젝트 |
| `saneflow/` | Neurova 자체 LM을 학습해보는 연구 라인 | 별도 학습 연구 |
| `neuromamba/` | Mamba-3 계열을 Neurova 모델 후보로 검증하는 연구 라인 | 별도 아키텍처 연구 |
| `luma/` | LUMA(Ledgered Universal Memory Automaton), 외부 슬롯 메모리를 쓰는 attention-free LM 실험 | 보관된 연구 |
| `qwen_exl3/` | Qwen3.6 EXL3/TabbyAPI 실서빙 운영 기록 | 배포 런북 |
| `qwen_memory/` | Qwen embed_tokens + USearch 기반 개인 기억 레이어 | 레거시 참고 구현 |

루트는 작업공간 인덱스와 공통 라우터만 담당한다. 새 문서, 스크립트, 설정,
데이터, 체크포인트, 논문 자료는 각 프로젝트 안에 둔다.

## `langburst/`: 메인 서빙 엔진

LangBurst는 이 저장소의 중심 프로젝트다. 이제 목표는 vLLM을 다시 만드는
것이 아니라, vLLM을 기본 엔진으로 쓰면서 SGLang, EXL3, 자체 native 엔진을
같은 계약으로 교체할 수 있는 서빙 extension layer가 되는 것이다. 방향은
“대형 서버 GPU 전용”이 아니라 저사양 GPU에서도 쓸 수 있는 정책, stateful
확장, custom model 실험을 얹는 것이다.

핵심 아이디어는 “서빙 엔진은 provider가 담당하고, LangBurst는 엔진 선택,
정책, stateful/session 확장, custom model 실험 경계를 담당한다”는 것이다.
기본 provider는 `vllm`이고, `sglang`, `exl3`, `native`가 같은
`EngineRegistry`에 등록된다.

Qwen3.6-27B Q4 Marlin 경로는 이제 LangBurst native engine plugin의
실험/레거시 경로다. Qwen3.6 GDN custom model이 vLLM custom model로
이식되기 전까지 보존하지만, 일반 HF/Gemma/Llama 계열 serving은 vLLM
provider로 보내는 것이 기준이다.

LangBurst가 실제로 담고 있는 것:

```text
langburst/langburst/engines/
  EngineDescriptor: 엔진 identity와 capability
  EngineModelSpec: 모델 경로/이름/dtype/quantization 등 공통 모델 선언
  EngineBackend: list_models, health, generate_chat, stream_chat 계약
  EngineRegistry: vllm, sglang, exl3, native provider의 단일 등록 지점
  vllm.py: 기본 실행 provider
  native.py: 기존 LangBurst 자체 런타임 wrapper

langburst/langburst/core/
  RuntimeFeatures/RuntimePlan: LangBurst-only stateful/research feature vocabulary
  adapter.py: native engine 내부에서 쓰는 legacy adapter registry

langburst/langburst/adapters/
  qwen36: native engine의 Qwen3.6 hybrid GDN 모델 adapter
  qwen36-a3b: Qwen3.6 A3B 계열 adapter
  qwen36_impl/: Qwen 전용 config/model/state 구현

langburst/csrc/
  Marlin W4A16 GEMM
  rowwise low-bit GEMV fallback
  RMSNorm
  Qwen Gated DeltaNet recurrence
  native engine 전용 attention/sampling helper

langburst/server.py
  OpenAI-compatible /v1/chat/completions
  EngineBackend 기반 라우팅
  /v1/langburst/health, /v1/langburst/features, /v1/langburst/engines

langburst/tests/
  engine registry/server/native/cuda 단위 검증
```

LangBurst의 주요 기능 축:

```text
엔진 교체 구조
  EngineRegistry
  vllm 기본 provider
  sglang optional provider target
  exl3 optional provider target
  native provider for Qwen3.6/GDN
  OpenAI-compatible API

저사양 GPU / 메모리 스케일링
  vLLM quantization/offload 활용
  low-bit checkpoint format
  Q4 Marlin fused projection path
  rowwise 2-8bit groupwise weights
  native engine fallback for custom checkpoints

stateful / 긴 문맥 기능
  engine capability 기반 기능 노출
  kv_window_policy: error, shift, ring
  ring KV
  pooled DecodeState
  snapshots
  boundary_decay
  infinite streaming gate
  episodic memory gate
  TTT sidecar gate

서빙 성능 기능
  vLLM continuous batching
  vLLM paged KV / prefix cache
  vLLM CUDA graph / speculative decoding
  native batch-state CUDA kernels only for Qwen3.6/GDN

운영 안정성
  health/status endpoints
  engine capability introspection
  backend별 교체 가능 배포
```

현재 정체성은 “vLLM 기본의 교체형 서빙 extension layer”다. 자체 native
엔진은 Qwen3.6/GDN custom kernel을 위한 plugin이고, generic serving 기능은
vLLM/SGLang/EXL3 provider가 맡는다.

대표 명령:

```bash
cd langburst
LANGBURST_SKIP_CUDA_EXT=1 python -m pip install -e .
./scripts/cuda_compile_and_test.sh
langburst-qwen-quantize /path/to/hf-model /path/to/converted-runtime-model --bits 4 --group-size 128
langburst-qwen-audit /path/to/converted-runtime-model --hf-model /path/to/hf-model
langburst-chat --engine vllm --model /path/or/hf-name --prompt "안녕"
langburst-server --engine vllm --model /path/or/hf-name --port 8008
langburst-chat --engine native --adapter qwen36 --hf-model /path/to/hf-model --qb-model /path/to/qb-model --prompt "안녕"
./neurova.sh langburst server --engine vllm --model /path/or/hf-name --port 8008
```

먼저 읽을 문서:

```text
langburst/README.md
langburst/docs/DESIGN.md
langburst/docs/ADAPTER_ARCHITECTURE.md
langburst/docs/RUNTIME_FEATURES.md
langburst/docs/PERFORMANCE_LOG.md
langburst/docs/SERVING_ENGINE_TODO.md
```

## `saneflow/`: 자체 LM 학습 연구

SaneFlow는 LangBurst처럼 기존 대형 모델을 서빙하는 런타임이 아니라,
Neurova 자체 언어 모델을 직접 학습해보는 연구 라인이다. 여기에는 모델
정의, 데이터 mix, tokenizer, 학습 profile, quality gate, autoresearch loop가
들어 있다.

개념적으로 SaneFlow는 “작고 통제 가능한 자체 LM을 만들어 언어 prior,
state mixer, dense Transformer baseline, ChatML SFT 품질 게이트를 연구하는
라인”이다. 실행은 `./neurova.sh saneflow ...` 또는
`saneflow/scripts/run.sh ...`로 명시한다. 작업공간의 메인 프로젝트는
LangBurst다.

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

## `qwen_exl3/`: Qwen EXL3 실서빙 런북

`qwen_exl3/`는 모델 코드 프로젝트가 아니라 운영 기록이다. ml-dmc8에서
Qwen3.6-27B를 EXL3 + DFlash + TabbyAPI로 실제 OpenAI-compatible 서버로
띄우면서 생긴 설치, 패치, 장애 대응 절차를 보관한다.

개념적으로 이 폴더는 “LangBurst 이전/병행의 실전 배포 레퍼런스”다.
어떤 모델과 quantization이 16GB에서 돌아갔는지, TabbyAPI/ExLlamaV3에서
어떤 문제가 있었고 어떻게 고쳤는지 확인하는 곳이다.

담고 있는 내용:

```text
qwen_exl3/README_TABBYAPI_EXL3.md
  Qwen3.6-27B EXL3 3.08bpw
  DFlash draft model
  100K context 근처 운영값
  KV cache quantization
  ExLlamaV3 / TabbyAPI 문제 해결 기록

qwen_exl3/CODEX_QWEN_RUNBOOK.md
  Codex/CLI proxy 실사용 검증
  no-think template
  tool-call 종료 문제
  한글 byte-fallback streaming 문제 패치

templates/
  Qwen chat template
```

## `luma/`: LUMA(Ledgered Universal Memory Automaton) 슬롯 메모리 LM 보관 연구

LUMA(Ledgered Universal Memory Automaton)는 현재 주력 경로가 아니라 보관된 연구다. 하지만 단순 폐기물이
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

## `qwen_memory/`: Qwen 기반 개인 기억 레이어

`qwen_memory/`는 기존 혼동되던 개인 기억 경로를 정리한 이름이다. 이 프로젝트는
독립 LLM도 아니고 새 symbolic LM도 아니다. 실제 성격은
`Qwen/Qwen3.5-4B`에 embedding/USearch 기반 개인 기억 레이어를 붙인
레거시 참고 구현이다.

이름을 `qwen_memory`로 둔 이유는 코드 성격이 그대로 드러나기 때문이다.
생성은 Qwen이 하고, 기억 embedding도 Qwen의 `embed_tokens`를 쓰며,
저장/검색은 USearch memory slot이 맡는다. CLI는 실행 표면일 뿐 프로젝트
정체성이 아니다.

개념적으로는 심볼릭한 언어획득 가설을 시도한 흔적이 있다. 문서상 핵심
prior는 `entity/event/relation exist`, `event/situation model`,
`memory slots`이고, 구현은 이를 명시적인 symbol table이나 grammar rule이
아니라 Qwen의 `embed_tokens` 평균 embedding과 USearch memory slot으로
대체한다.

즉 `qwen_memory`는 규칙 기반 symbolic parser가 아니다. 문법 규칙, 질문 유형 규칙,
1인칭/3인칭 변환 같은 하드코딩을 제거하고, “비슷한 사건/개체/관계는
embedding 공간에서 가깝다”는 prior만 둔 뒤 raw text를 저장/검색한다.
검색된 원문 기억은 system prompt에 붙고, 실제 귀속과 표현은 Qwen 모델이
처리한다.

`qwen_memory`가 실제로 담고 있는 것:

```text
qwen_memory/main.py
  Qwen3.5-4B loader
  bf16/4bit mode
  TextIteratorStreamer 기반 streaming CLI
  Memory class
  MemSlot raw text storage
  entity/event/relation prior를 USearch slot으로 표현
  embed_tokens mean pooling
  USearch cosine index
  per-user ~/.qwen_memory/users/<user>/ namespace
  /think, /nothink, /effort, /user, /clear, /status, remember:

qwen_memory/docs/ARCHITECTURE.md
  구조 설명

qwen_memory/scripts/deploy.sh
  ml-dmc8 배포 스크립트

qwen_memory/scripts/run.sh
  bf16/4bit 실행 모드 선택 후 qwen_memory/main.py 실행
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

루트에는 작업공간 인덱스와 얇은 라우터만 둔다. 프로젝트별 실행 로직은
루트 스크립트에 넣지 않고 각 프로젝트의 `scripts/run.sh`가 소유한다.
`./neurova.sh`를 인자 없이 실행하면 특정 레거시 프로젝트로 진입하지 않고
사용법만 보여준다.

```text
README.md
neurova.sh
.gitignore
```

프로젝트별 문서, 스크립트, 설정, 데이터, 논문, 토크나이저, run 결과는
각 프로젝트 아래에 둔다.

```text
langburst/docs/
langburst/scripts/run.sh
saneflow/configs/
saneflow/scripts/run.sh
neuromamba/scripts/
luma/data/
qwen_memory/docs/
qwen_memory/scripts/run.sh
```

대용량 데이터, 다운로드 토크나이저, 논문 PDF, 학습 결과, 평가 결과는
기본적으로 git 추적 대상이 아니다. 예외는 재현에 필요한 작은 seed corpus나
설정 파일뿐이다.

## 기본 확인

정리나 이동 후 최소한 다음을 확인한다.

```bash
bash -n neurova.sh
find langburst/scripts luma/scripts saneflow/scripts neuromamba/scripts qwen_memory/scripts -maxdepth 1 -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
./neurova.sh help
./neurova.sh langburst help
./neurova.sh qwen-memory help
python -m compileall -q langburst luma saneflow neuromamba qwen_memory
python saneflow/scripts/saneflow_run.py list
python -m neuromamba.cli model-info --mode mimo-r4-tiny --tokenizer byte --device cpu
```

GPU, gated tokenizer, 원격 호스트, 대형 checkpoint가 필요한 검증은 해당
환경에서 별도로 수행한다.
