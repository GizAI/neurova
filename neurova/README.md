# Neurova V6 — Pure Embedding Memory CLI

**Zero hardcoded language rules. Language-acquisition substrate only.**

```text
Don't program grammar rules.
Create a vessel that can learn language.

The vessel =
  Embedding-based association (embed_tokens + USearch)
  + Memory slots (raw text, no conversion)
  + Prediction via generation (Qwen3.5-4B)
  + Error-driven learning (auto-store + dedup)
  + Per-user persistent memory
```

## Quick Start

```bash
# Requirements (conda env recommended)
conda create -n neurova python=3.10
conda activate neurova
pip install torch transformers bitsandbytes usearch numpy sentencepiece

# Run
python3 neurova/v6.py

# Optional: fast attention kernels (fla)
pip install flash-linear-attention causal-conv1d
```

## Usage

```text
> My name is Alice.
[auto-store] → remembers "My name is Alice."

> What is my name?
Alice — retrieved from memory via embedding similarity.

> /think              → enable thinking mode (deeper reasoning)
> /nothink            → disable thinking (faster)
> /effort low|mid|high → set reasoning effort
> /user bob           → switch to bob's memory space
> /clear              → reset conversation history
> /status             → system info (user, slots, VRAM)
> remember: <text>    → manually store memory
```

## Architecture

```
User Input
  → Tokenize
  → Embedding (mean-pooled embed_tokens)
  → USearch cosine similarity → top-K memory recall
  → apply_chat_template() with enable_thinking
  → model.generate() → stream output
  → Auto-store raw text to memory (after response)
```

### Memory System
- Raw user text stored **verbatim** — no conversion, no templates
- **MemSlot**: text, source, timestamp, retrieval count
- **USearch** index: 2560-dim cosine, per-user namespace
- **Dedup**: near-exact (dist < 0.10) skip / same-topic (dist < 0.45) update
- **Isolation**: `~/.neurova_v6/users/<name>/` per user

## What's Hardcoded (Cognitive Priors Only)

```text
Entity embedding exists        → embed_tokens layer
Similar things are similar     → cosine similarity
Memory persists                → JSON + USearch on disk
Different users separate       → filesystem namespaces
```

## What's NOT Hardcoded (All Removed)

```text
Language-specific patterns     ❌
1st→3rd person conversion      ❌
Topic keyword lists             ❌
Similarity thresholds           ❌
Question-type handlers          ❌
Grammar rules                   ❌
Signal maps                     ❌
Template extractors             ❌
```

## Performance (RTX 4080 16GB, Qwen3.5-4B)

| Metric | Value |
|--------|-------|
| Prefill | 5,767 tok/s |
| Generation | 32-33 tok/s |
| TTFT (warm) | 54-80ms |
| VRAM | 7.8GB (bf16) |
| Max context | 16,384 tokens |

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `neurova/v6.py` | Main engine | 331 |
| `neurova.sh` | Launcher | 9 |
| `docs/ARCHITECTURE_v6.md` | Architecture docs | — |
| `scripts/deploy_v6.sh` | ml-dmc8 deploy | 51 |

## Deployment

```bash
# To ml-dmc8
bash neurova/scripts/deploy_v6.sh

# Or manual
ssh ml-dmc8
cd /home/user/workspace/neurova
conda activate neurova_vsa
python3 neurova/v6.py
```

## Design Principles

1. **No TTT** — embedding memory is faster and equally effective for personal info
2. **No separate embedding model** — uses model's own `embed_tokens`
3. **No background threads** — memory store after response (user sees no delay)
4. **Raw text storage** — no conversion, model handles attribution
5. **Per-user directories** — complete memory isolation

## License

MIT — Giz Inc.
