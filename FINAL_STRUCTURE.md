# Final Structure — V17 Learned Semantic Compiler

```text
brainos/
  ir.py                       # Cognitive IR dataclasses and proof objects
  compiler.py                 # Hybrid compiler: grammar → learned semantic parser → phrase parser → regex fallback
  cognitive_model.py          # No-LM cognitive scorer: IR/evidence/proof/world/memory operations
  memory.py                   # Versioned evidence graph memory
  reasoner.py                 # Active memory reasoner: taxonomy, temporal, causal, comparison, exception
  agent.py                    # FinalCognitiveOS runtime
  semantic/
    ir_grammar.py             # Meaning Atom Table
    phrase_segmenter.py       # Surface discourse segmentation
    slot_tagger.py            # Tiny structured IR type/slot learner
    fragment_parser.py        # Learned phrase-to-IR parser and composer
    candidate_assembler.py    # IR graph candidate assembly
    verifier.py               # Candidate schema verification
    active_teacher.py         # Hard-case active learning queue
    dataset_generator.py      # IR-first seed corpus builder
    train_parser.py           # JSONL → LearnedSemanticParser loader
    eval_parser.py            # parser accuracy evaluation
```

Principle:

```text
No next-token language model.
No text generation objective.
Language is UI; Cognitive IR is thought.
```
