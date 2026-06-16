# LUMA: Ledgered Universal Memory Automaton

Status: stopped archive. LUMA (Ledgered Universal Memory Automaton) is not the
active Neurova model line, not the default launcher target, and not eligible for
`luma/runs/luma_current` promotion. Run it only for explicit archive/debug work
with `NEUROVA_ALLOW_LUMA=1`.

This folder is a runnable zero-base prototype of the LUMA idea: events are mixed
locally, sparse persistent memory slots are read and edited, and every useful
checkpoint must prove that slot removal or slot-key randomization hurts memory
QA. These notes are preserved as an archive of the slot-memory validation path,
not as the current Neurova training plan.

Historical validation path:

1. `memory_proof`: prove the memory core first. Train only chunk-gap memory
   tasks and require `normal` to beat both `no_slots` and `random_slot_keys`.
2. `mixed_chat`: after memory proof passes, mix raw continuation, short chat,
   answer-only memory curriculum, and a smaller slot-proof stream.
3. `chat_candidate`: promote only when chat sanity and memory ablation gates
   both pass. `luma/runs/luma_current` is reserved for this stage only.

## What is implemented

- `byte`: original byte tokenizer with a 259-token vocabulary.
- `bytepatch`: exact byte-preserving adaptive patch route with all byte
  fallbacks, all byte-pair patches, common multi-byte latent patches,
  corpus-learned latent patches, and byte-span metadata.
- `qwen`: Qwen3.5 BBPE tokenizer route backed by `luma/tokenizers/qwen35`.
- `LUMALM`, an attention-free language model.
- `LUMABlock`, which performs event encoding, causal local mixing, sparse slot
  read, gated slot edit, slot utility/confidence updates, and token decoding.
- `forward(input_ids, slots_in=None, return_slots=True, ablation=...)` so chat,
  generation, document, and project memory can persist slot state outside a
  single forward call.
- Slot diagnostics in the train loop: entropy, usage entropy, update frequency,
  overwrite rate, slot delta, confidence, and utility.
- Recipe-based training: `memory_proof` first, then `mixed_chat`. `custom`
  remains available for experiments but is not the default research path.
- Memory ablations in `luma.eval_memory`: `normal`, `no_slots`, and
  `random_slot_keys`.
- `JsonlLedger`, an append-only raw-byte evidence ledger scaffold.

Not yet complete: learned byte-boundary routing, true dual bytepatch/Qwen
alignment, entity/fact-slot supervised objectives, future-use labels, and
proof-grounded decoding.

## Promotion gate

Do not promote a run to `luma/runs/luma_current` until it passes both surfaces:

- Chat sanity: `hi`, self-introduction, and simple QA produce readable Korean or
  English sentences without repetition collapse.
- Memory proof: exact copy > 50%, recall > 60%, json_field > 70%, and
  `normal` clearly beats both `no_slots` and `random_slot_keys`.

## Quick smoke train

```bash
python3 -m luma.train --steps 50 --batch-size 4 --d-model 96 --layers 2 --slots 32 --topk 4 --seq-len 192 --out luma/runs/luma-smoke
python3 -m luma.generate --ckpt luma/runs/luma-smoke/model.pt --prompt $'Memory page:\nMina owns the blue key.\nMina should go to seoul.\nQuestion: What object belongs to Mina?\nAnswer:'
python3 -m luma.eval_memory --ckpt luma/runs/luma-smoke/model.pt --cases 20 --compare-ablations
```

## Qwen3.5 tokenizer route

Download tokenizer-only files:

```bash
python3 luma/scripts/luma_download_qwen_tokenizer.py --repo Qwen/Qwen3.5-0.8B --out luma/tokenizers/qwen35
```

Train the same LUMA core with Qwen BBPE tokens instead of raw bytes:

```bash
python3 -m luma.train \
  --tokenizer-backend qwen \
  --qwen-tokenizer-path luma/tokenizers/qwen35 \
  --dataset-mode records \
  --data luma/README.md \
  --steps 50 \
  --batch-size 2 \
  --seq-len 128 \
  --d-model 96 \
  --layers 2 \
  --slots 32 \
  --topk 4 \
  --out luma/runs/luma-qwen-smoke
```

Checkpoints store `tokenizer_backend`, tokenizer paths, vocabulary size, and a
tokenizer fingerprint, so
`luma.generate` and `luma/scripts/luma_chat.py` automatically reopen the correct
front-end. The event/slot core is shared; embeddings and output heads remain
backend-specific through each checkpoint's vocabulary size.

## Adaptive byte-latent patch route

The `bytepatch` route is tokenizer-free at the input boundary and has no unknown
tokens. It uses:

- Special tokens: `pad`, `bos`, `eos`
- 256 single-byte fallback tokens
- 65,536 byte-pair patch tokens
- Common multi-byte latent patches for chat, English, code, JSON, and Korean
- Optional corpus-learned latent patches from
  `luma/tokenizers/luma_bytepatch/bytepatch_vocab.json`
- `encode_with_spans()` metadata carrying `byte_start`, `byte_end`, and patch
  source, so ledger/proof memory can store raw-byte provenance instead of
  tokenizer ids.

Learn the corpus-adaptive patch vocabulary first:

```bash
python3 luma/scripts/luma_train_bytepatch_tokenizer.py \
  --data luma/README.md \
  --out luma/tokenizers/luma_bytepatch/bytepatch_vocab.json \
  --max-patches 8192 \
  --min-count 2 \
  --min-len 3 \
  --max-len 12
```

Smoke train:

```bash
python3 -m luma.train \
  --tokenizer-backend bytepatch \
  --bytepatch-vocab-path luma/tokenizers/luma_bytepatch/bytepatch_vocab.json \
  --dataset-mode records \
  --data luma/README.md \
  --steps 50 \
  --batch-size 2 \
  --seq-len 128 \
  --d-model 96 \
  --layers 2 \
  --slots 32 \
  --topk 4 \
  --out luma/runs/luma-bytepatch-smoke
```

## Archived Training Recipe

```bash
python3 -m luma.train \
  --recipe memory_proof \
  --slot-proof-gap-lines 8 \
  --steps 2000 \
  --batch-size 6 \
  --seq-len 384 \
  --d-model 384 \
  --layers 8 \
  --slots 192 \
  --topk 8 \
  --out luma/runs/luma_memory_proof_v1
```

Remote DMC8 run:

```bash
ssh ml-dmc8 'cd /home/user/workspace/neurova && source ~/miniconda3/etc/profile.d/conda.sh && conda activate mamba3_siso && ./luma/scripts/luma_train_dmc8.sh'
```

The DMC8 script defaults to `RECIPE=memory_proof` and writes
`memory_ablation_eval.json`. A low loss without ablation separation is a failed
memory architecture experiment, not a deployable chat model.
