# rsg_emergent_lm

CPU-only, no-hardcoded-intent language prototype using the uploaded Qwen-style `tokenizer.json`.

It removes the previous `detect_intent`, synonym table, and answer templates. The model learns only from JSONL pairs:

```json
{"source":"...", "target":"..."}
```

Generation flow:

```text
prompt
→ Qwen ByteLevel-BPE ids + unicode char n-gram features
→ sparse nearest-example retrieval
→ weighted target-side variable-order token LM
→ global target LM backoff
→ beam decode
```

Run:

```bash
./run_demo.sh  # 기본 토크나이저 자동 탐색
python3 emergent_rsg_lm.py --tokenizer /mnt/data/tokenizer.json generate \
  --pairs seed_pairs.jsonl \
  --prompt "오늘 일정 수립해줘" \
  --out demo_today_schedule.json
```

Interactive chat:

```bash
./run_chat.sh  # 기본 토크나이저 + seed_pairs 자동 탐색
```

캐시 디렉터리는 환경변수로 바꿀 수 있습니다.

```bash
RSG_CACHE_DIR=~/.cache/neurova_rsg_fast ./run_chat.sh
```

During chat:

- `user> ...` 입력 후 Enter: 모델 응답
- `/help` 도움말
- `/clear` 대화 이력 초기화
- `/stats` 모델 통계 출력
- `/exit` 종료

This is not a frontier LLM. Language ability appears only inside the distribution covered by the training examples. To scale it, add many source/target examples and raw-text-derived denoising pairs.

### Seed corpus

기본 `seed_pairs.jsonl`은 현재 Hugging Face의 `IkJun1/korean-qa-dataset`(한국어 QA 페어)에서 추출해 사용합니다.

```bash
./fetch_korean_seed_pairs.sh
```

환경변수로 크기/데이터셋을 바꿀 수 있습니다.

```bash
HF_DATASET=IkJun1/korean-qa-dataset HF_SPLIT=train MAX_PAIRS=50000 ./fetch_korean_seed_pairs.sh
```

수집 기준:
- `prompt` → `source`
- `response` → `target`
- 프롬프트 접두사(`Human:`, `GPT:`) 제거
- 중복 제거, 너무 긴 항목 필터링

### 성능(모델 캐시)

`generate`/`chat`는 기본적으로 디스크 캐시를 사용합니다.

```bash
python3 emergent_rsg_lm.py --tokenizer /mnt/data/tokenizer.json generate \
  --pairs seed_pairs.jsonl \
  --prompt "안녕?" \
  --cache-dir /tmp/rsg_cache
```

동일한 `tokenizer + seed_pairs + model 파라미터(order/beam/branch)` 조합이면 첫 빌드 후 다음 실행은 `fit_pairs`를 재실행하지 않습니다.

캐시 무효화 조건:
- tokenizer 파일/`seed_pairs` 원본이 변경
- `order`, `beam`, `branch` 변경
- `cache version` 변경(내부 호환성 갱신)

원하면 디버그 메시지를 확인할 수 있습니다:

```bash
RSG_CACHE_DEBUG=1 python3 emergent_rsg_lm.py ...
```

기본은 압축 없이 빠르게 저장/로드합니다. 저장공간을 아끼고 싶으면:

```bash
RSG_CACHE_COMPRESS=1 python3 emergent_rsg_lm.py ...
```

캐시를 비우고 새로 학습하려면:

```bash
python3 emergent_rsg_lm.py ... --no-cache
```
