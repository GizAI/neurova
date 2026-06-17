# AGENTS.md

이 저장소에서 작업할 때는 속도보다 먼저 정확한 계약을 잡고, 그 위에서 성능을 끌어올린다. 임시 우회, 옵션 끄기, 휴리스틱 패치로 문제를 숨기지 않는다.

## 기본 원칙

- 문제를 만나면 증상만 막지 말고 실제 호출 경로, 상태 전이, 로그, 테스트로 근본 원인을 확인한다.
- 기능을 끄거나 기본값을 낮추는 방식은 해결책이 아니다. 안전을 위해 임시로 막아야 하면 원인, 영향, 되돌릴 조건을 코드나 문서에 명확히 남기고 근본 수정까지 이어간다.
- 새 분기, 새 옵션, 새 helper를 만들기 전에 기존의 단일 책임 경계가 있는지 먼저 찾는다. 같은 정책을 두 곳 이상에서 판단하지 않는다.
- 하드코딩된 모델명, 경로, 컨텍스트 크기, thinking 정책, 엔진 선택, 포트, VRAM 가정은 피한다. 필요한 값은 설정, policy resolver, adapter capability, 또는 실행 인자로 모은다.
- 휴리스틱은 최후의 수단이다. 사용한다면 이름과 위치를 명확히 하고, 정답 계약을 대체하지 않도록 테스트 가능한 policy로 격리한다.
- 중복 구현, 호환 래퍼, v1/v2식 병렬 경로, 죽은 실험 코드는 남기지 않는다. 성능 이득이 검증되지 않은 경로는 기본 경로에 섞지 않는다.
- 사용자가 “정리정돈”, “파편화 제거”, “최종 정답구조”를 말하면 파일 하나가 아니라 관련 호출 경계 전체에서 중복과 불용 코드를 찾는다.

## 런타임/서빙 작업

- LangBurst의 기본 구조는 `RuntimeEngine` + adapter + engine provider + resolved policy다. 서버, CLI, 벤치, OpenAI-compatible API는 이 경계를 우회하지 않는다.
- Qwen 전용 로직은 Qwen adapter나 native engine 내부에 둔다. core에는 모델 이름이나 Qwen 전용 개념이 새지 않게 한다.
- vLLM, SGLang, EXL3, native engine은 교체 가능한 engine provider로 다룬다. 특정 엔진용 임시 코드를 서버 표면에 직접 박지 않는다.
- thinking, chat template, speculative decoding, KV dtype, context window, batching, prefix cache 같은 실행 정책은 한 resolver에서 결정하고 소비자는 그 결과만 읽는다.
- OOM은 서버를 failed 상태로 남기면 안 된다. admission control, queueing, pool cleanup, request cancellation, runtime recovery 계약으로 처리한다.
- 긴 입력은 무조건 큰 dense prompt로 밀어 넣지 않는다. chunked prefill, bounded exact KV, recurrent/state memory, ingestion/RAG/prefix cache 계약 중 어디로 들어가야 하는지 먼저 구분한다.
- OpenWebUI는 클라이언트다. 느린 요청의 생성 주체를 말할 때는 LangBurst API 요청, OpenWebUI backend task, follow-up/title generation 같은 실제 호출자를 정확히 구분한다.

## 성능 작업

- 성능 개선은 수치로 남긴다. 최소한 TTFT, prefill tok/s, decode tok/s, queue wait, VRAM, context window, batch size, dtype, engine을 같이 기록한다.
- 단일 사용자 속도와 멀티유저 throughput을 섞어 말하지 않는다.
- 기본값은 품질과 안정성 parity를 통과하고 실제 speedup이 검증된 경로만 켠다.
- prefill/decode/batch/speculative/paged-KV 경로를 따로 측정한다. 한 수치로 전체 병목을 추정하지 않는다.
- GPU hot path에서 CPU handoff, `.item()` 동기화, 반복 allocation, row-wise Python loop가 생기면 병목 후보로 먼저 의심한다.
- vLLM과 비교할 때는 개념만 베끼지 말고 scheduler, block table, cache, verifier, sampling, CUDA graph, metrics의 실제 계약 차이를 코드 레벨에서 확인한다.

## 정확성/품질 게이트

- 빠른 경로를 기본으로 켜기 전에 느린 기준 경로와 token identity, logits/top-k, state trajectory, continuation parity를 확인한다.
- paged KV, arena state, raw block prefill, batch decode, speculative verifier는 같은 commit contract를 공유해야 한다. 각자 rollback/commit 규칙을 따로 만들지 않는다.
- speculative decoding은 proposer가 live target state를 오염시키지 않아야 한다. target verifier만 state commit 권한을 가진다.
- reject가 있는 speculative 경로는 accepted prefix state만 commit하고 rejected candidate state를 폐기한다. replay 비용을 줄이려면 state trajectory/commit-prefix 버퍼로 해결한다.
- 한국어/UTF-8 깨짐, 조기 종료, 반복 출력, missing `<think>` prefix 같은 출력 품질 문제는 tokenizer, streaming, stop sequence, chat template, state corruption 중 실제 원인을 분해해 고친다.
- 테스트는 계약을 지키는 데 필요한 만큼만 만든다. 테스트 전용 우회나 생산 코드와 다른 가짜 계약을 만들지 않는다.

## 코드 정리 기준

- 한 정책을 바꾸려고 여러 파일의 상수나 조건문을 같이 고쳐야 하면 구조가 잘못된 것이다. resolver, registry, adapter capability, config schema 중 하나로 모은다.
- 새 기능을 추가한 뒤에는 기존 대체 경로가 여전히 필요한지 확인하고, 불필요하면 같은 변경에서 제거한다.
- 이름은 현재 역할을 반영한다. 과거 구현 세부사항인 `marlin`, `hot`, `v1`, `v2` 같은 이름은 외부 모델명이나 사용자 표면에 남기지 않는다.
- 기존 사용자가 만든 변경을 되돌리지 않는다. 충돌하면 먼저 읽고, 같은 계약 안에서 합친다.
- 배포 스크립트, 테스트, 문서는 같은 기본값을 공유해야 한다. 서로 다른 기본 context/window/dtype/model 경로를 만들지 않는다.

## 작업 방식

- 파일과 로그는 `rg`, `git status`, 실제 서버 로그, 실제 API 호출로 확인한다.
- 긴 검증은 자주 반복하지 말고, 빠른 계약 테스트와 대표 벤치로 좁힌 뒤 필요할 때만 전체 스트레스 테스트를 돌린다.
- ml-dmc8이나 OpenWebUI에 적용한 변경은 로컬 코드와 배포 상태를 혼동하지 말고 각각 명확히 보고한다.
- 완료 보고에는 무엇을 바꿨는지, 어떤 테스트를 통과했는지, 아직 남은 위험이 무엇인지 짧게 포함한다.
