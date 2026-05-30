# Neurova V6 — Architecture

## Core Philosophy

**Zero hardcoded language rules. Language-acquisition substrate only.**

```text
Don't program grammar rules.
Create a vessel that can learn language.

The vessel = 
  Embedding-based association (embed_tokens + USearch)
  + Event/situation model (memory slots)
  + Prediction via generation (Qwen3.5-4B)
  + Error-driven learning (auto-store + dedup)
  + Multi-layer memory (per-user persistent storage)
```

## Architecture

```
User Input
  → Tokenize
  → Embedding (mean-pooled embed_tokens, layer-0)
  → USearch cosine similarity → top-K memory recall
  → apply_chat_template() with enable_thinking
  → model.generate() → stream output
  → Background: auto-store raw text to memory
  → Response to user
```

## What's Hardcoded (Cognitive Priors Only)

```text
Entity/event/relation exist         → USearch index
Similar things are similar          → cosine embedding
Memory persists across sessions     → JSON + USearch on disk
Different users have separate memory → ~/.neurova_v6/users/<name>/
```

## What's NOT Hardcoded (Removed)

```text
detect_personal()      — English/Korean "I am..." patterns   ❌ removed
to_user_memory()       — 1st→3rd person string replacement   ❌ removed
_find_update()         — topic keyword list                  ❌ removed
_dedup() threshold     — magic number                        ❌ removed
Language-specific rules                                      ❌ removed
```

## Memory System

### Storage
- Raw user text stored verbatim (no conversion)
- `MemSlot`: text, source, timestamp, retrieval count
- Embedding: mean-pooled `model.model.embed_tokens()` → 2560-dim vector
- Index: USearch (cosine, 2560-dim)
- Per-user directory: `~/.neurova_v6/users/<name>/`

### Retrieval
- Pure cosine similarity (no thresholds)
- Always return top-K (default 7)
- Model handles attribution naturally via system prompt

### Dedup/Update
- `dist < 0.10` → near-exact duplicate, skip
- `dist < 0.45` → same topic, update with longer/more specific text
- Otherwise → new entry

## Performance (RTX 4080 16GB, Qwen3.5-4B, fla kernels)

| Metric | Value |
|--------|-------|
| Prefill | 5,767 tok/s |
| Generation | 32-33 tok/s |
| TTFT (warm) | 54-80ms |
| Total response | 1-6s |
| VRAM | 7.8GB (bf16) |
| Max context | 16,384 tokens |

## Thinking Control

| Effort | `enable_thinking` | Speed | Use Case |
|--------|-------------------|-------|----------|
| `low` (default) | False | 0.5-4s | General conversation |
| `mid` | True | 5-30s | Complex reasoning |
| `high` | True + prompt | 30-120s+ | Deep analysis |

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `neurova_v6.py` | Main engine | 331 |
| `neurova.sh` | Launcher | 5 |
| `ARCHITECTURE_v6.md` | This file | — |
| `deploy_v6.sh` | ml-dmc8 deploy | 40 |

## Dependencies

```bash
pip install torch transformers bitsandbytes usearch numpy sentencepiece
pip install flash-linear-attention causal-conv1d  # optional, for speed
```

## Usage

```bash
# Local
python3 neurova_v6.py

# Specific user
V6_USER=alice python3 neurova_v6.py

# Thinking mode
V6_EFFORT=mid python3 neurova_v6.py

# CLI commands
/think              → enable thinking (mid)
/nothink            → disable thinking (low)
/effort low|mid|high → set effort
/user <name>        → switch user (memory isolation)
/clear              → reset conversation history
remember: <text>    → manually store memory
recall              → browse recent memories
status              → system status
```

## Key Design Decisions

1. **No TTT** — embedding memory is faster and equally effective for personal info recall
2. **No separate embedding model** — uses model's own `embed_tokens` layer
3. **No background threads for memory** — synchronous after response (user sees no delay)
4. **Raw text storage** — no conversion, model handles attribution
5. **Per-user directories** — complete memory isolation via filesystem
