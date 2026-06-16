# Qwen Memory Architecture

## Core Idea

Qwen Memory is a Qwen-backed personal memory layer. It is not a standalone LLM
and not a symbolic LM implementation. It uses Qwen3.5-4B for generation and
uses the same model's `embed_tokens` layer as a cheap embedding model for memory
storage and recall.

The symbolic idea is present only as a cognitive prior:

```text
entities/events/relations exist
similar situations should be close in embedding space
memory persists across sessions
different users need isolated memory spaces
```

The implementation does not build explicit symbol tables, extraction templates,
or grammar rules. It stores raw text and retrieves nearby memories with USearch.

## Runtime Flow

```text
User Input
  -> Qwen tokenizer
  -> mean-pooled Qwen embed_tokens
  -> USearch cosine top-K memory recall
  -> Qwen chat template with recalled raw memories in system prompt
  -> Qwen model.generate()
  -> stream output
  -> store user text as raw memory
```

## Memory System

Storage:

- raw user text, stored verbatim
- `MemSlot`: text, source, timestamp, retrieval count
- embedding: mean-pooled `model.model.embed_tokens()`
- index: USearch cosine, 2560 dimensions
- namespace: `~/.qwen_memory/users/<user>/`

Retrieval:

- query text is embedded with the same Qwen embedding path
- top-K nearest memories are returned
- Qwen handles attribution and wording through generation

Dedup/update:

- `dist < 0.10`: near-exact duplicate, skip
- `dist < 0.45`: same topic, replace with longer/more specific text
- otherwise: append new memory slot

## LLM Boundary

LLM elements:

- `Qwen/Qwen3.5-4B`
- Qwen tokenizer and chat template
- `AutoModelForCausalLM.generate()`
- optional thinking mode through chat-template settings
- Qwen `embed_tokens` reused as memory embedding

Not included:

- new LM architecture
- training loop
- fine-tuning
- symbolic parser
- grammar rules
- continual model-weight updates

## Commands

```bash
python3 qwen_memory/main.py
QWEN_MEMORY_USER=alice python3 qwen_memory/main.py
QWEN_MEMORY_EFFORT=mid python3 qwen_memory/main.py
```

Interactive commands:

```text
/think
/nothink
/effort low|mid|high
/user <name>
/clear
remember: <text>
recall
status
```

## Design Decisions

1. No separate embedding model: Qwen's own embedding layer is reused.
2. No language rules: no Korean/English pattern extraction or person rewriting.
3. Raw text storage: the generator handles attribution and wording.
4. Per-user filesystem namespace: memory isolation stays simple.
5. No model training: memory changes are index/storage updates, not weight updates.
