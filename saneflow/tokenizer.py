from __future__ import annotations

import json
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer


class SaneFlowBPETokenizer:
    def __init__(self, path: str | Path) -> None:
        root = Path(path)
        vocab = root / "vocab.json"
        merges = root / "merges.txt"
        spm_model = root / "tokenizer.model"
        config = root / "saneflow_tokenizer_config.json"
        self.path = str(root)
        self.backend = "sentencepiece" if spm_model.exists() else "byte_bpe"
        self.sp = None
        self.tokenizer = None
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        self.additional_special_tokens: list[str] = []
        if config.exists():
            raw = json.loads(config.read_text(encoding="utf-8"))
            self.pad_token = raw.get("pad_token", self.pad_token)
            self.bos_token = raw.get("bos_token", self.bos_token)
            self.eos_token = raw.get("eos_token", self.eos_token)
            self.unk_token = raw.get("unk_token", self.unk_token)
            self.additional_special_tokens = list(raw.get("additional_special_tokens", []))
        if self.backend == "sentencepiece":
            import sentencepiece as spm

            self.sp = spm.SentencePieceProcessor(model_file=str(spm_model))
            self.pad_token_id = self.sp.piece_to_id(self.pad_token)
            self.bos_token_id = self.sp.piece_to_id(self.bos_token)
            self.eos_token_id = self.sp.piece_to_id(self.eos_token)
            self.unk_token_id = self.sp.piece_to_id(self.unk_token)
        else:
            if not vocab.exists() or not merges.exists():
                raise FileNotFoundError(f"missing tokenizer files under {root}")
            self.tokenizer = ByteLevelBPETokenizer(str(vocab), str(merges))
            if self.additional_special_tokens:
                self.tokenizer.add_special_tokens(self.additional_special_tokens)
            self.pad_token_id = self.tokenizer.token_to_id(self.pad_token)
            self.bos_token_id = self.tokenizer.token_to_id(self.bos_token)
            self.eos_token_id = self.tokenizer.token_to_id(self.eos_token)
            self.unk_token_id = self.tokenizer.token_to_id(self.unk_token)
        if self.eos_token_id is None:
            raise ValueError(f"tokenizer at {root} has no eos token {self.eos_token!r}")

    @property
    def name_or_path(self) -> str:
        return self.path

    def __len__(self) -> int:
        if self.backend == "sentencepiece":
            assert self.sp is not None
            return int(self.sp.vocab_size())
        assert self.tokenizer is not None
        return self.tokenizer.get_vocab_size()

    @property
    def special_token_ids(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for token in [self.pad_token, self.bos_token, self.eos_token, self.unk_token, *self.additional_special_tokens]:
            token_id = self.token_to_id(token)
            if token_id is not None:
                out[token] = int(token_id)
        return out

    def token_to_id(self, token: str) -> int | None:
        if self.backend == "sentencepiece":
            assert self.sp is not None
            token_id = self.sp.piece_to_id(token)
            return int(token_id) if token_id >= 0 else None
        assert self.tokenizer is not None
        token_id = self.tokenizer.token_to_id(token)
        return int(token_id) if token_id is not None else None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if self.backend == "sentencepiece":
            assert self.sp is not None
            ids = list(self.sp.encode(text, out_type=int))
        else:
            assert self.tokenizer is not None
            ids = self.tokenizer.encode(text).ids
        if add_special_tokens:
            if self.bos_token_id is not None:
                ids = [self.bos_token_id] + ids
            if self.eos_token_id is not None:
                ids = ids + [self.eos_token_id]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        if self.backend == "sentencepiece":
            assert self.sp is not None
            special_by_id = {v: k for k, v in self.special_token_ids.items()}
            if skip_special_tokens:
                ids = [x for x in ids if x not in special_by_id]
                return self.sp.decode(ids)
            parts: list[str] = []
            run: list[int] = []
            for token_id in ids:
                if token_id in special_by_id:
                    if run:
                        parts.append(self.sp.decode(run))
                        run = []
                    parts.append(special_by_id[token_id])
                else:
                    run.append(token_id)
            if run:
                parts.append(self.sp.decode(run))
            return "".join(parts)
        if skip_special_tokens:
            specials = set(self.special_token_ids.values())
            ids = [x for x in ids if x not in specials]
        assert self.tokenizer is not None
        return self.tokenizer.decode(ids)
