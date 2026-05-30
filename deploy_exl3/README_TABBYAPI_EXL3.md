# Neurova 16GB VRAM 최적화 추론 환경 (ExLlamaV3 + DFlash + TabbyAPI)

2026년 5월 기준, 단일 16GB GPU (RTX 4080 등)에서 Qwen3.6-27B 모델을 가장 넓은 컨텍스트와 가장 빠른 속도로 서빙하기 위한 최종 정답 구조입니다.

## 1. 아키텍처 및 성능
- **Target Model**: `UnstableLlama/Qwen3.6-27B-exl3-3.08bpw` (약 13GB)
- **Draft Model (초안 가속)**: `turboderp/Qwen3.6-27B-DFlash-exl3` (브랜치: `3.00bpw`)
- **Backend**: ExLlamaV3 (Pre-built wheel) + TabbyAPI (Source)
- **Context Limit**: `147456` 토큰 (16GB VRAM 한계상 DFlash 사용 시 134K가 안정적. DFlash 미사용 시 134K 가능)
- **KV Cache**: `4,4` 비트 양자화 (메모리 절약)
- **출력 속도**: 약 **79.2 tokens/sec** (DFlash 투기적 디코딩 적용 시 2배 가속)

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

## 3. 데몬 서비스 관리
이 서버는 systemd 데몬(`neurova-tabbyapi.service`)으로 등록되어 있습니다.

- **로그 확인**: `journalctl -u neurova-tabbyapi -f`
- **시작/중지/재시작**: `sudo systemctl [start|stop|restart] neurova-tabbyapi`
- **포트**: `http://127.0.0.1:5000/v1` (OpenAI 호환 API)
