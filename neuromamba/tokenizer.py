from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOKENIZER_ROOT = REPO_ROOT / "neuromamba" / "tokenizers"


class Tokenizer(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class ByteTokenizer:
    """UTF-8 byte tokenizer for Mamba-3 smoke training.

    This is intentionally small and dependency-free. It is not the final Korean
    production tokenizer, but it preserves Korean text losslessly for pipeline
    validation on a 16GB GPU.
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3

    @property
    def vocab_size(self) -> int:
        return 259

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(b + self.byte_offset for b in text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            if token_id in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if self.byte_offset <= token_id < self.byte_offset + 256:
                raw.append(token_id - self.byte_offset)
        return raw.decode("utf-8", errors="replace")


class HFTokenizer:
    """Wrapper around a real subword tokenizer for language-model training."""

    def __init__(self, name: str | list[str]):
        from transformers import AutoTokenizer

        names = [name] if isinstance(name, str) else name
        errors: list[str] = []
        self.name = names[0]
        self.tokenizer = None
        for candidate in names:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(candidate, clean_up_tokenization_spaces=False)
                self.name = candidate
                if candidate != names[0]:
                    print(
                        f"warning: loaded tokenizer from fallback repo {candidate}; primary {names[0]} was unavailable",
                        file=sys.stderr,
                        flush=True,
                    )
                break
            except Exception as exc:
                errors.append(f"{candidate}: {exc.__class__.__name__}: {exc}")
        if self.tokenizer is None:
            if any(item.startswith("meta-llama/") for item in names):
                raise RuntimeError(
                    "failed to load Llama-3.1 tokenizer. Accept the Meta license and set HF_TOKEN, "
                    "run `huggingface-cli login`, or allow the configured public tokenizer mirror. "
                    "Errors: " + " | ".join(errors)
                )
            raise RuntimeError("failed to load tokenizer. Errors: " + " | ".join(errors))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.bos_id = int(self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.eos_id)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return [int(i) for i in ids]

    def decode(self, ids: list[int]) -> str:
        skip = {self.pad_id, self.bos_id, self.eos_id}
        cleaned = [
            int(i)
            for i in ids
            if 0 <= int(i) < self.vocab_size and int(i) not in skip
        ]
        return self.tokenizer.decode(cleaned, skip_special_tokens=True)


def build_tokenizer(kind: str) -> Tokenizer:
    if kind == "byte":
        return ByteTokenizer()
    if kind in {"llama31", "llama-3.1", "llama3.1"}:
        return HFTokenizer([
            str(LOCAL_TOKENIZER_ROOT / "llama31"),
            "meta-llama/Meta-Llama-3.1-8B",
            "meta-llama/Llama-3.1-8B",
            "NousResearch/Meta-Llama-3.1-8B",
        ])
    if kind in {"llama31-instruct", "llama-3.1-instruct", "llama3.1-instruct"}:
        return HFTokenizer([
            str(LOCAL_TOKENIZER_ROOT / "llama31-instruct"),
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "NousResearch/Meta-Llama-3.1-8B-Instruct",
        ])
    if kind == "gpt-neox":
        return HFTokenizer("EleutherAI/gpt-neox-20b")
    if kind.startswith("hf:"):
        return HFTokenizer(kind[3:])
    raise ValueError(f"unknown tokenizer: {kind}")
