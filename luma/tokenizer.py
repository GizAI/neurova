from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Protocol


class LUMATokenizer(Protocol):
    pad_id: int
    bos_id: int
    eos_id: int

    @property
    def vocab_size(self) -> int: ...

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class TokenSpan:
    token_id: int
    byte_start: int
    byte_end: int
    source: str


@dataclass(frozen=True)
class ByteTokenizer:
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
        ids.extend(byte + self.byte_offset for byte in text.encode("utf-8", errors="replace"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            if self.byte_offset <= int(token_id) < self.byte_offset + 256:
                raw.append(int(token_id) - self.byte_offset)
        return raw.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class QwenTokenizer:
    path: str = "tokenizers/qwen35"
    pad_id: int = 248044
    bos_id: int = 248044
    eos_id: int = 248044

    def __post_init__(self) -> None:
        root = Path(self.path)
        tokenizer_json = root / "tokenizer.json"
        if not tokenizer_json.exists():
            raise FileNotFoundError(
                f"Qwen tokenizer not found at {tokenizer_json}; run scripts/luma_download_qwen_tokenizer.py"
            )
        from tokenizers import Tokenizer

        object.__setattr__(self, "_tokenizer", Tokenizer.from_file(str(tokenizer_json)))
        config_path = root / "tokenizer_config.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            added = cfg.get("added_tokens_decoder", {})
            end_ids = [int(k) for k, v in added.items() if v.get("content") == "<|endoftext|>"]
            if end_ids:
                object.__setattr__(self, "pad_id", end_ids[0])
                object.__setattr__(self, "bos_id", end_ids[0])
                object.__setattr__(self, "eos_id", end_ids[0])

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(int(item) for item in self._tokenizer.encode(text).ids)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        clean = [int(item) for item in ids if int(item) not in {self.pad_id, self.bos_id, self.eos_id}]
        return self._tokenizer.decode(clean)


def _build_common_bytepatches() -> tuple[bytes, ...]:
    words = [
        " the", " and", " to", " of", " in", " that", " is", " for", " on", " with", " as", " are",
        " this", " from", " by", " be", " or", " an", " at", " not", " it", " can", " should",
        "User:", "Assistant:", "System:", "Question:", "Answer:", "Instruction:", "Memory page:",
        "```", "\n\n", "\n- ", "\n1. ", "\n2. ", "\n3. ", ": ", ", ", ". ", "; ", " = ", " -> ",
        "true", "false", "null", "json", "python", "function", "return", "class", "import", "const",
        "def ", "if ", "else", "elif", "for ", "while", "try", "except", "raise", "torch", "model",
        "한국", "한국어", "질문", "답변", "사용자", "시스템", "모델", "학습", "추론", "메모리", "문서",
    ]
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    punct = ["()","[]","{}","<>","==","!=","<=",">=","//","::","--","__","##","**","&&","||"]
    patches: list[bytes] = []
    seen: set[bytes] = set()
    for item in words + chars + punct:
        raw = item.encode("utf-8")
        if len(raw) >= 3 and raw not in seen:
            patches.append(raw)
            seen.add(raw)
    return tuple(patches)


@dataclass(frozen=True)
class AdaptiveBytePatchTokenizer:
    """Exact byte-preserving adaptive patch tokenizer.

    The vocabulary has three layers:
    1. special tokens: pad/bos/eos
    2. all single-byte and all byte-pair tokens, guaranteeing exact fallback
    3. common multi-byte latent patches, reducing sequence length for language,
       code, chat templates, JSON, and common Korean spans.

    This is the tokenizer-side final shape for LUMA-DTF. A learned boundary
    predictor can later replace the greedy router while keeping token ids,
    byte spans, and exact reconstruction semantics stable.
    """

    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    byte_offset: int = 3
    pair_offset: int = 259
    common_offset: int = 65795
    max_patch_bytes: int = 24
    vocab_path: str = "tokenizers/luma_bytepatch/bytepatch_vocab.json"

    def __post_init__(self) -> None:
        common = self._load_common_patches()
        object.__setattr__(self, "_common", common)
        patch_to_id = {patch: self.common_offset + idx for idx, patch in enumerate(common)}
        object.__setattr__(self, "_patch_to_id", patch_to_id)
        object.__setattr__(self, "_id_to_patch", {token_id: patch for patch, token_id in patch_to_id.items()})
        by_first: dict[int, list[bytes]] = {}
        for patch in common:
            by_first.setdefault(patch[0], []).append(patch)
        for key, values in by_first.items():
            values.sort(key=len, reverse=True)
        object.__setattr__(self, "_by_first", by_first)

    def _load_common_patches(self) -> tuple[bytes, ...]:
        patches: list[bytes] = []
        seen: set[bytes] = set()
        for patch in _build_common_bytepatches():
            if patch not in seen:
                patches.append(patch)
                seen.add(patch)
        path = Path(self.vocab_path)
        if path.exists():
            data = json.loads(path.read_text())
            for item in data.get("patches", []):
                raw = base64.b64decode(item["bytes_b64"])
                if len(raw) >= 3 and raw not in seen:
                    patches.append(raw)
                    seen.add(raw)
        patches.sort(key=lambda item: (-len(item), item))
        return tuple(patches)

    @property
    def vocab_size(self) -> int:
        return self.common_offset + len(self._common)

    def _single_id(self, byte: int) -> int:
        return self.byte_offset + int(byte)

    def _pair_id(self, a: int, b: int) -> int:
        return self.pair_offset + int(a) * 256 + int(b)

    def _decode_id_to_bytes(self, token_id: int) -> bytes:
        token_id = int(token_id)
        if self.byte_offset <= token_id < self.pair_offset:
            return bytes([token_id - self.byte_offset])
        if self.pair_offset <= token_id < self.common_offset:
            pair = token_id - self.pair_offset
            return bytes([pair // 256, pair % 256])
        return self._id_to_patch.get(token_id, b"")

    def _match_common(self, raw: bytes, pos: int) -> bytes | None:
        candidates = self._by_first.get(raw[pos], ())
        remaining = raw[pos : pos + self.max_patch_bytes]
        for patch in candidates:
            if len(patch) <= len(remaining) and remaining.startswith(patch):
                return patch
        return None

    def encode_with_spans(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[TokenSpan]:
        raw = text.encode("utf-8", errors="replace")
        spans: list[TokenSpan] = []
        if add_bos:
            spans.append(TokenSpan(self.bos_id, 0, 0, "control"))
        pos = 0
        while pos < len(raw):
            common = self._match_common(raw, pos)
            if common is not None:
                spans.append(TokenSpan(self._patch_to_id[common], pos, pos + len(common), "bytepatch_common"))
                pos += len(common)
                continue
            if pos + 1 < len(raw):
                spans.append(TokenSpan(self._pair_id(raw[pos], raw[pos + 1]), pos, pos + 2, "bytepatch_pair"))
                pos += 2
            else:
                spans.append(TokenSpan(self._single_id(raw[pos]), pos, pos + 1, "bytepatch_byte"))
                pos += 1
        if add_eos:
            spans.append(TokenSpan(self.eos_id, len(raw), len(raw), "control"))
        return spans

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        return [span.token_id for span in self.encode_with_spans(text, add_bos=add_bos, add_eos=add_eos)]

    def decode(self, ids: list[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            token_id = int(token_id)
            if token_id in {self.pad_id, self.bos_id, self.eos_id}:
                continue
            raw.extend(self._decode_id_to_bytes(token_id))
        return bytes(raw).decode("utf-8", errors="replace")


def learn_bytepatch_vocab(
    texts: list[str],
    *,
    max_patches: int = 8192,
    min_count: int = 2,
    min_len: int = 3,
    max_len: int = 12,
) -> list[bytes]:
    counts: Counter[bytes] = Counter()
    for text in texts:
        raw = text.encode("utf-8", errors="replace")
        for start in range(len(raw)):
            stop = min(len(raw), start + max_len)
            for end in range(start + min_len, stop + 1):
                piece = raw[start:end]
                if b"\x00" in piece:
                    continue
                counts[piece] += 1
    builtins = set(_build_common_bytepatches())
    scored = []
    for piece, count in counts.items():
        if count < min_count or piece in builtins:
            continue
        score = count * (len(piece) - 1)
        scored.append((score, count, len(piece), piece))
    scored.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3]))
    selected: list[bytes] = []
    seen: set[bytes] = set()
    for _, _, _, piece in scored:
        if piece in seen:
            continue
        selected.append(piece)
        seen.add(piece)
        if len(selected) >= max_patches:
            break
    return selected


def write_bytepatch_vocab(path: str | Path, patches: list[bytes], *, source: str = "corpus") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "luma-bytepatch-v1",
        "source": source,
        "patch_count": len(patches),
        "patches": [
            {
                "bytes_b64": base64.b64encode(patch).decode("ascii"),
                "byte_len": len(patch),
                "preview": patch.decode("utf-8", errors="replace"),
            }
            for patch in patches
        ],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build_tokenizer(
    backend: str = "byte",
    qwen_path: str = "tokenizers/qwen35",
    bytepatch_vocab_path: str = "tokenizers/luma_bytepatch/bytepatch_vocab.json",
) -> LUMATokenizer:
    if backend == "byte":
        return ByteTokenizer()
    if backend == "bytepatch":
        return AdaptiveBytePatchTokenizer(vocab_path=bytepatch_vocab_path)
    if backend in {"qwen", "qwen35"}:
        return QwenTokenizer(path=qwen_path)
    raise ValueError(f"unknown LUMA tokenizer backend: {backend}")


def tokenizer_fingerprint(
    backend: str,
    *,
    qwen_path: str = "tokenizers/qwen35",
    bytepatch_vocab_path: str = "tokenizers/luma_bytepatch/bytepatch_vocab.json",
) -> str:
    if backend == "bytepatch":
        path = Path(bytepatch_vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"bytepatch vocab manifest is required: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if backend in {"qwen", "qwen35"}:
        path = Path(qwen_path) / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"Qwen tokenizer file is required: {path}")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if backend == "byte":
        return "byte-v1"
    raise ValueError(f"unknown LUMA tokenizer backend: {backend}")


def assert_tokenizer_contract(config: dict, tokenizer: LUMATokenizer) -> None:
    expected_vocab = int(config["vocab_size"])
    if tokenizer.vocab_size != expected_vocab:
        raise RuntimeError(f"tokenizer/model vocab mismatch: tokenizer={tokenizer.vocab_size} model={expected_vocab}")
    expected_fingerprint = config.get("tokenizer_sha256")
    if not expected_fingerprint:
        raise RuntimeError("checkpoint is missing tokenizer_sha256; retrain with the current tokenizer contract")
    actual_fingerprint = tokenizer_fingerprint(
        config["tokenizer_backend"],
        qwen_path=config.get("qwen_tokenizer_path", "tokenizers/qwen35"),
        bytepatch_vocab_path=config.get("bytepatch_vocab_path", "tokenizers/luma_bytepatch/bytepatch_vocab.json"),
    )
    if actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"tokenizer fingerprint mismatch: current={actual_fingerprint} checkpoint={expected_fingerprint}"
        )
