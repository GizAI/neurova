#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|tool|>",
    "<|im_start|>",
    "<|im_end|>",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a multilingual tokenizer for SaneFlow/Neurova.")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", default="tokenizers/neurova_spm_unigram_64k")
    p.add_argument("--kind", choices=["sentencepiece_unigram", "byte_bpe"], default="sentencepiece_unigram")
    p.add_argument("--vocab-size", type=int, default=65536)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--character-coverage", type=float, default=0.99995)
    p.add_argument("--max-sentence-length", type=int, default=16384)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
    if args.kind == "sentencepiece_unigram":
        import sentencepiece as spm

        prefix = out / "tokenizer"
        user_symbols = [x for x in SPECIAL_TOKENS if x not in {"<pad>", "<bos>", "<eos>", "<unk>"}]
        spm.SentencePieceTrainer.train(
            input=str(tmp),
            model_prefix=str(prefix),
            model_type="unigram",
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            byte_fallback=True,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            split_digits=True,
            allow_whitespace_only_pieces=True,
            train_extremely_large_corpus=True,
            hard_vocab_limit=False,
            max_sentence_length=args.max_sentence_length,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<bos>",
            eos_piece="<eos>",
            user_defined_symbols=user_symbols,
        )
        vocab_size_actual = sum(1 for _ in (out / "tokenizer.vocab").open(encoding="utf-8"))
    else:
        from tokenizers import ByteLevelBPETokenizer

        tok = ByteLevelBPETokenizer()
        tok.train(
            files=[str(tmp)],
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            special_tokens=SPECIAL_TOKENS,
        )
        tok.save_model(str(out))
        vocab_size_actual = len(json.loads((out / "vocab.json").read_text(encoding="utf-8")))
    manifest = {
        "type": args.kind,
        "target": "ko_en_code",
        "vocab_size_requested": args.vocab_size,
        "min_frequency": args.min_frequency,
        "byte_fallback": args.kind == "sentencepiece_unigram",
        "character_coverage": args.character_coverage if args.kind == "sentencepiece_unigram" else None,
        "normalization": "identity" if args.kind == "sentencepiece_unigram" else "byte_level_bpe_default",
        "inputs": [str(Path(x)) for x in args.input],
        "pad_token": "<pad>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "additional_special_tokens": [x for x in SPECIAL_TOKENS if x not in {"<pad>", "<bos>", "<eos>", "<unk>"}],
    }
    (out / "saneflow_tokenizer_config.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.unlink(missing_ok=True)
    manifest["vocab_size_actual"] = vocab_size_actual
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "kind": args.kind,
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": vocab_size_actual,
        "special_tokens": SPECIAL_TOKENS,
    }, indent=2))


if __name__ == "__main__":
    main()
