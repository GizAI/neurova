from __future__ import annotations
from pathlib import Path
from typing import Iterable
from ..datasets import generate_nl_ir_examples, write_jsonl

def build_seed_corpus(path: Path, n: int = 2000, seed: int = 17) -> list[dict]:
    rows = generate_nl_ir_examples(n, seed=seed)
    write_jsonl(path, rows)
    return rows
