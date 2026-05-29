from __future__ import annotations
import json
from pathlib import Path
from .fragment_parser import LearnedSemanticParser

def train_from_jsonl(path: Path) -> LearnedSemanticParser:
    rows = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return LearnedSemanticParser(rows)
