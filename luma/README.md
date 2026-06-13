# LUMA: Ledgered Universal Memory Automaton

This folder is a runnable zero-base prototype of the LUMA idea: byte events are
mixed locally, each chunk reads a sparse set of persistent memory slots, and the
selected slots are edited with erase/write/protect gates. It is intentionally
small and dependency-light so the architecture can be trained and inspected
before adding custom kernels, external ledgers, or byte-patch compression.

## What is implemented

- Byte tokenizer with a 259-token vocabulary.
- `LUMALM`, an attention-free language model.
- `LUMABlock`, which performs event encoding, causal local mixing, sparse slot
  read, gated slot edit, slot utility/confidence updates, and token decoding.
- Synthetic long-memory training data where facts appear before distractors and
  answers require recalling the right fact.
- Training and generation entrypoints.

## Quick smoke train

```bash
python3 -m luma.train --steps 50 --batch-size 4 --d-model 96 --layers 2 --slots 32 --topk 4 --seq-len 192 --out runs/luma-smoke
python3 -m luma.generate --ckpt runs/luma-smoke/model.pt --prompt $'Memory page:\nMina owns the blue key.\nMina should go to seoul.\nQuestion: What object belongs to Mina?\nAnswer:'
```

## Larger local run

```bash
python3 -m luma.train --steps 2000 --batch-size 16 --d-model 256 --layers 6 --slots 128 --topk 8 --seq-len 384 --out runs/luma-256m-proto
```

The current machine reports no CUDA device from PyTorch, so smoke training will
run on CPU unless the environment changes. A real "말 잘하고 지능 좋은" model will
need a much larger corpus, GPU training, and the next objectives: future-use
supervision, ledger-page retrieval, proof grounding, and abstention loss.
