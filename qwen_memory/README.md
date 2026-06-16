# Qwen Memory

Qwen Memory는 독립 LLM도, 새 symbolic LM도 아니다. `Qwen/Qwen3.5-4B`에
개인 기억 레이어를 붙인 레거시 실험이다. Qwen의 `embed_tokens`를 평균
pooling해서 사용자 발화를 embedding으로 만들고, USearch cosine index에
raw text 기억을 저장한 뒤, 다음 대화에서 관련 기억을 system prompt로
주입한다.

이름을 `qwen_memory`로 둔 이유는 코드 성격이 그대로 드러나기 때문이다.
핵심 의존성은 Qwen이고, 핵심 기능은 memory다. CLI는 실행 표면일 뿐
프로젝트 정체성이 아니다.

## Concept

```text
Qwen3.5-4B generation
+ Qwen embed_tokens mean pooling
+ USearch cosine memory index
+ raw text MemSlot storage
+ per-user persistent namespace
= Qwen-backed personal memory layer
```

이 프로젝트에는 심볼릭한 언어획득 가설도 남아 있다. `entity/event/relation`
같은 cognitive prior를 별도 symbol table, parser, grammar rule로 구현하지
않고, “비슷한 사건/개체/관계는 embedding 공간에서 가깝다”는 전제로
USearch memory slot에 표현한다.

## What It Is

- Qwen3.5-4B 기반 대화형 개인 기억 실험
- 별도 embedding model 없이 Qwen의 `embed_tokens`를 재사용
- raw user text를 변환 없이 저장
- per-user `~/.qwen_memory/users/<user>/` 저장소 사용
- `remember:`, `recall`, `/user`, `/think`, `/effort` 같은 대화형 명령 제공

## What It Is Not

- 새로 학습한 LLM
- 자체 LM architecture
- symbolic parser
- grammar/rule 기반 언어 모델
- fine-tuning 또는 continual training 시스템
- LangBurst/SaneFlow/NeuroMamba의 하위 모듈

## Quick Start

```bash
conda create -n qwen_memory python=3.10
conda activate qwen_memory
pip install torch transformers bitsandbytes usearch numpy sentencepiece

python3 qwen_memory/main.py
```

Optional speed dependencies:

```bash
pip install flash-linear-attention causal-conv1d
```

## Usage

```text
> My name is Alice.
[auto-store] remembers "My name is Alice."

> What is my name?
Alice, retrieved from memory via embedding similarity.

> /think
> /nothink
> /effort low|mid|high
> /user bob
> /clear
> /status
> remember: <text>
> recall
```

## Architecture

```text
user input
-> tokenize with Qwen tokenizer
-> mean-pool Qwen embed_tokens
-> USearch cosine top-K recall
-> put recalled raw memories into system prompt
-> Qwen model.generate()
-> stream response
-> auto-store user text after response
```

Memory details:

- `MemSlot`: text, source, timestamp, retrieval count
- embedding: `model.model.embed_tokens()` mean pool, 2560 dimensions
- index: USearch cosine index
- storage: `~/.qwen_memory/users/<user>/memory.{json,usearch}`
- dedup/update: near-exact duplicate skip, same-topic longer text update

## Environment

Preferred variables:

```text
QWEN_MEMORY_MODE=bf16|4bit
QWEN_MEMORY_EFFORT=low|mid|high
QWEN_MEMORY_CTX=16384
QWEN_MEMORY_MAX=4096
QWEN_MEMORY_K=7
QWEN_MEMORY_HIST=4
QWEN_MEMORY_AUTO=1
QWEN_MEMORY_USER=default
QWEN_MEMORY_HOME=~/.qwen_memory
```

## Files

```text
qwen_memory/main.py
qwen_memory/docs/ARCHITECTURE.md
qwen_memory/scripts/deploy.sh
```

## Deployment

```bash
bash qwen_memory/scripts/deploy.sh
```
