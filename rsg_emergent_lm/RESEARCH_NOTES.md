# Research notes

Goal: remove hand-written intent routing, synonym tables, and answer templates.

Implemented two CPU-only data-driven variants:

1. ExemplarTransducer
   - Qwen ByteLevel-BPE tokenization
   - Unicode char n-gram + token feature retrieval
   - weighted target-side variable-order language model
   - global target LM backoff
   - beam decoding

2. DataDrivenComposer
   - retrieves nearest source examples
   - splits their target responses into sentence fragments
   - selects fragments by query relevance, corpus centrality, and diversity
   - recombines fragments without task-specific if/else

The code now uses a larger Korean QA corpus (`IkJun1/korean-qa-dataset`), converted to
`{"source","target"}` JSONL format under 30k pairs, so it handles a much broader
Korean input range than the initial 35-row toy seed. 

A performance hotspot in the first CLI demo was post-generation overlap scanning across
all training pairs. It is now limited to retrieved neighbors, so generation latency drops
back to retrieval cost (~2~3s for 30k pairs on this CPU baseline).

Disk cache now wraps training (`fit_pairs`) with a signature-based version check:
- cache key includes tokenizer file signature (size/mtime/hash), pair-file signature list,
  and model params (`order/beam/branch`)
- cache is reused when those sources are unchanged
- cache payload is stored as primitives, so loading works whether `emergent_rsg_lm.py` is run
  as `python3 emergent_rsg_lm.py` or imported as module
