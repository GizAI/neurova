# Neurova V40 — Developmental Neuro-Symbolic Cognitive Engine

A **genuine learning system** that acquires language and reasoning through conversation — no hardcoded templates, no question-type-specific handlers.

## Architecture

```
Utterance → Perception Cortex → Knowledge Graph → Universal Query → Answer
                                ↓
                           Learning through prediction error
```

### Key Components

- **V40CleanEngine** (`neurova/architecture/v40_engine_clean.py`): Single synchronous inference engine. No threads, no global workspace — pure fact storage + universal query.
- **SensoryPerceptionCortex** (`neurova/architecture/perception_cortex.py`): spaCy + Kiwi NLP with singleton model loading. Handles deep preposition structures ("went back to the bathroom").
- **KnowledgeGraph**: Universal fact store with inheritance, negation tracking, location overwrite, and object possession tracking.
- **Universal Query Interface**: Single `query()` method handles all question types — no `if/elif` chains for grow/need/fly/where/what.

### Learning Architecture

- All utterances become facts in the graph
- Inheritance is derived from is-a relations (not hardcoded)
- Negation blocks inheritance naturally
- Location tracking uses overwrite (last update wins)
- Object possession tracked through pick/drop/put/give verbs
- Multi-hop reasoning: object → holder → holder's location

## Quick Start

```bash
# Requirements
pip install spacy usearch numpy requests
python -m spacy download en_core_web_sm

# Run CLI
PYTHONPATH=. python -m neurova.v40_cli

# Or via rsync to ml-dmc8 (GPU server)
./rsync_deploy.sh
ssh ml-dmc8 "cd ~/workspace/neurova && PYTHONPATH=. python -m neurova.v40_cli"
```

## Benchmarks

### Elementary School Science Curriculum — 8/8 (100%)
```
Does a sunflower grow?  → Yes, it grows!
What does a dog need?   → It needs water.
Does a rock grow?       → No, it cannot grow.
Can a robin fly?        → Yes, it can fly!
Can a penguin fly?      → No, it cannot fly.
Does a penguin need water? → It needs water.
Does a penguin grow?    → Yes, it grows!
```

### bAbI Tasks
- **Task 1 (Single Supporting Fact):** 1000/1000 (100.0%) SOLVED
- **Task 2 (Two Supporting Facts):** 668/1000 (66.8%)
- *More tasks being implemented...*

### Natural Language
```
> I am Kyungtae.
Got it.
> Kyungtae is the CEO of Giz Inc.
Got it.
> What is Giz?
I recall: Its be ceo of giz inc.
```

## bAbI Evaluation

```bash
# Download bAbI data and run full 20-task evaluation
PYTHONPATH=. python eval_babi_full.py
```

## Deployment

```bash
# Local
PYTHONPATH=. python -m neurova.v40_cli

# Remote (ml-dmc8 GPU server)
./rsync_deploy.sh
```

## Dependencies

- Python 3.10+
- spaCy + en_core_web_sm (NLP parsing)
- kiwipiepy (Korean NLP, optional)
- usearch (vector search, optional)
- numpy

## License

MIT — Giz Inc.
