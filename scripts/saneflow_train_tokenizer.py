#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a compact byte-level BPE tokenizer for SaneFlow.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", default="tokenizers/saneflow_tinystories_16k")
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--min-frequency", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text_files: list[str] = []
    tmp = out / "_train_text.txt"
    with tmp.open("w", encoding="utf-8") as w:
        for name in args.input:
            with Path(name).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{"):
                        text = json.loads(line).get("text", "")
                    else:
                        text = line
                    text = text.strip()
                    if text:
                        w.write(text.replace("\n", " ") + "\n")
    text_files.append(str(tmp))
    special = ["<pad>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"]
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=text_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special,
    )
    tok.save_model(str(out))
    (out / "saneflow_tokenizer_config.json").write_text(
        json.dumps(
            {
                "type": "byte_level_bpe",
                "pad_token": "<pad>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "additional_special_tokens": ["<|im_start|>", "<|im_end|>"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    vocab = json.loads((out / "vocab.json").read_text(encoding="utf-8"))
    tmp.unlink(missing_ok=True)
    print(json.dumps({
        "out": str(out),
        "vocab_size": len(vocab),
        "special_tokens": special,
    }, indent=2))


if __name__ == "__main__":
    main()
