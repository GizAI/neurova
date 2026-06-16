# Neurova 16GB VRAM 최적화 추론 환경 (ExLlamaV3 + DFlash + TabbyAPI)

2026년 5월 기준, 단일 16GB GPU (RTX 4080 등)에서 Qwen3.6-27B 모델을 EXL3 + DFlash로 안정 서빙하기 위한 구조입니다.

## 1. 아키텍처 및 성능
- **Target Model**: `UnstableLlama/Qwen3.6-27B-exl3-3.08bpw` (약 13GB)
- **Draft Model (초안 가속)**: `turboderp/Qwen3.6-27B-DFlash-exl3` (브랜치: `3.00bpw`)
- **Backend**: ExLlamaV3 (Pre-built wheel) + TabbyAPI (Source)
- **Context Limit**: `100352` 토큰 (ml-dmc8 실구동 안정값: DFlash 유지 + `max_batch_size: 2` + `draft_num_tokens: 6` 기준)
- **KV Cache**: `3,2` 비트 양자화 (메모리 절약)
- **출력 속도**: 짧은 응답은 약 35-58 tokens/sec, 70K급 Codex 컨텍스트에서는 프리필 병목으로 수 분까지 걸릴 수 있음. `draft_num_tokens: 8+`는 100K/batch 2에서 VRAM 여유가 부족함.

## 2. 발생했던 문제와 해결 방법 (Troubleshooting)

1. **ExLlamaV3 JIT 빌드 (C++ ABI 충돌) 무한 대기 에러**
   - **증상**: 파이썬에서 `import exllamav3` 실행 시 PyTorch C++ ABI 버전 충돌(`_GLIBCXX_USE_CXX11_ABI`)로 인해 백그라운드 컴파일(ninja)이 무한 대기하거나 `undefined symbol` 에러를 뱉음.
   - **해결**: pip 소스 빌드를 피하고, 릴리즈 페이지에서 PyTorch 버전에 맞는 **미리 빌드된 Wheel**(`exllamav3-0.0.38+cu128.torch2.10.0-cp310-cp310-linux_x86_64.whl`)을 직접 다운로드해 설치하여 원천 차단.

2. **DFlash 가중치 누락 이슈**
   - **증상**: `turboderp/Qwen3.6-27B-DFlash-exl3` 기본 `main` 브랜치에 `safetensors` 파일이 없음.
   - **해결**: `--revision "3.00bpw"` 브랜치를 명시하여 가중치를 다운로드함.

3. **모델 Config 호환성 문제 (Qwen3_5ForConditionalGeneration -> Qwen3_5ForCausalLM)**
   - **증상**: ExLlamaV3가 `config.json`의 architecture 속성을 엄격히 검사하여 Load 실패.
   - **해결**: `config.json` 내의 아키텍처를 `Qwen3_5ForCausalLM`로 수정하고 `text_config` 내의 파라미터(`hidden_size` 등)를 Root 노드로 복사하는 패치 적용.

4. **TabbyAPI 패키지 및 파라미터 변경**
   - **증상**: `pip install tabbyAPI` 버전에 치명적 오류(파일 누락)가 존재하며, ExLlamaV3 0.0.38부터 Cache 파라미터(`batch_size` -> `max_batch_size`)가 바뀜.
   - **해결**: TabbyAPI를 git clone 후 소스 설치(`pip install -e .`). `config.yml`의 draft_cache_mode 형식을 `Q4`로 지정.

5. **Codex/CLI Proxy에서 응답이 비어 보이는 이슈**
   - **증상**: `/v1/chat/completions`는 200을 반환하지만 `content`가 `null`이고 `reasoning_content`만 흘러 Codex CLI에서 응답이 없는 것처럼 보임.
   - **해결**: 모델의 `chat_template.jinja`와 `tokenizer_config.json`에서 `enable_thinking` 기본값을 no-think로 패치함. 명시적으로 `chat_template_kwargs: {"enable_thinking": true}`를 보낼 때만 thinking을 켬. 완성 템플릿은 `qwen_exl3/templates/qwen3_coder_neurova_chat_template.jinja`에 보관함.

6. **Codex에서 말만 하고 작업을 멈추는 이슈**
   - **증상**: Qwen이 `이제 cleanly 다시 시작합니다.` 같은 진행 멘트만 최종 응답으로 내고 도구 호출 없이 턴을 끝냄.
   - **원인**: TabbyAPI는 정상 완료했고 출력 토큰도 짧았으므로 truncation이 아니라 모델의 종료 판단 문제였음.
   - **해결**: 같은 템플릿 가드에 `Never end with a promise of future action; if work remains, call tools in the same reply, otherwise provide completed results.`를 추가하고, tool instruction block에도 “run/check/edit/verify/retry/execute를 말하면 같은 응답 안에 tool call을 포함”하도록 명시함.

7. **EXL3 한글 byte-fallback 깨짐 (`U+FFFD`)**
   - **증상**: direct TabbyAPI/proxy/Codex 경로에서 `토큰` 같은 일부 한글이 replacement character(`U+FFFD`)로 출력됨.
   - **원인**: 모델 token id는 정상이나 ExLlamaV3 스트리밍 경로가 byte-fallback 토큰 조각의 `result["text"]`를 너무 일찍 확정해 U+FFFD로 조립함.
   - **해결**: `backends/exllamav3/model.py`에서 생성 token id 누적분을 tokenizer로 재디코딩하고, U+FFFD가 남은 suffix는 다음 토큰까지 보류하도록 패치함.

## 3. 데몬 서비스 관리
이 서버는 systemd 데몬(`neurova-tabbyapi.service`)으로 등록되어 있습니다.

- **로그 확인**: `journalctl -u neurova-tabbyapi -f`
- **시작/중지/재시작**: `sudo systemctl [start|stop|restart] neurova-tabbyapi`
- **포트**: `http://127.0.0.1:5000/v1` (OpenAI 호환 API)

Codex/CLI Proxy 실사용 검증 절차와 원격 패치 목록은 `qwen_exl3/CODEX_QWEN_RUNBOOK.md`에 정리되어 있습니다.
