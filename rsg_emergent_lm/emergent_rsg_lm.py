#!/usr/bin/env python3
"""No-hardcoded-intent CPU-only emergent language prototype.

Core idea:
  - Use the uploaded Qwen-style tokenizer.json as the symbol layer.
  - Learn from examples and raw text only.
  - No intent if/else, no synonym table, no hand-written answer templates.
  - Response generation is a conditional, memory-augmented variable-order token LM:
      query -> nearest training prompts -> weighted target-token continuation model
            -> global token LM backoff -> beam decode.

This is deliberately small and transparent. Real quality needs large corpora.
"""
from __future__ import annotations

import argparse
import gzip
import json
import hashlib
import math
import os
import random
import re
import sys
import pickle
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

try:
    import regex as ure
except Exception as exc:  # pragma: no cover
    raise SystemExit("The small 'regex' module is required.") from exc

BOS = -1
EOS = -2

# ---------------------------------------------------------------------------
# Qwen/HF ByteLevel BPE tokenizer implementation copied as a standalone layer.
# ---------------------------------------------------------------------------

def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

@dataclass
class Tokenized:
    ids: List[int]
    tokens: List[str]

class QwenTokenizer:
    def __init__(self, tokenizer_json: str | Path):
        self.path = Path(tokenizer_json)
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.data = data
        self.normalizer_type = (data.get("normalizer") or {}).get("type")
        self.pre_tokenizer = data.get("pre_tokenizer") or {}
        self.decoder_type = (data.get("decoder") or {}).get("type")
        self.model = data["model"]
        if self.model.get("type") != "BPE":
            raise ValueError(f"Only BPE tokenizers are supported, got {self.model.get('type')}")
        self.vocab: dict[str, int] = dict(self.model["vocab"])
        self.id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.added_tokens = list(data.get("added_tokens") or [])
        self.added_by_content = {t["content"]: int(t["id"]) for t in self.added_tokens}
        self.id_to_added = {int(t["id"]): t["content"] for t in self.added_tokens}
        self.special_contents = sorted([t["content"] for t in self.added_tokens], key=len, reverse=True)
        self.merge_rank: dict[tuple[str, str], int] = {}
        for i, item in enumerate(self.model.get("merges") or []):
            a, b = item.split(" ", 1)
            self.merge_rank[(a, b)] = i
        pattern = None
        if self.pre_tokenizer.get("type") == "Sequence":
            for p in self.pre_tokenizer.get("pretokenizers", []):
                if p.get("type") == "Split":
                    pattern = (p.get("pattern") or {}).get("Regex")
                    break
        if not pattern:
            pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        self.split_re = ure.compile(pattern)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.cache: dict[str, List[str]] = {}

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFC", text) if self.normalizer_type == "NFC" else text

    def _split_special(self, text: str) -> List[tuple[str, bool]]:
        parts: List[tuple[str, bool]] = []
        i = 0
        buf: List[str] = []
        while i < len(text):
            matched = None
            for s in self.special_contents:
                if text.startswith(s, i):
                    matched = s
                    break
            if matched:
                if buf:
                    parts.append(("".join(buf), False)); buf.clear()
                parts.append((matched, True))
                i += len(matched)
            else:
                buf.append(text[i]); i += 1
        if buf:
            parts.append(("".join(buf), False))
        return parts

    def pretokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        out: List[str] = []
        for segment, is_special in self._split_special(text):
            if not segment:
                continue
            if is_special:
                out.append(segment)
            else:
                out.extend(m.group(0) for m in self.split_re.finditer(segment))
        return out

    def _bytelevel(self, piece: str) -> str:
        return "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))

    def _bpe(self, byte_piece: str) -> List[str]:
        cached = self.cache.get(byte_piece)
        if cached is not None:
            return cached[:]
        if not byte_piece:
            return []
        if byte_piece in self.vocab:
            self.cache[byte_piece] = [byte_piece]
            return [byte_piece]
        word = tuple(byte_piece)
        while len(word) > 1:
            ranked = []
            for i in range(len(word) - 1):
                p = (word[i], word[i + 1])
                r = self.merge_rank.get(p)
                if r is not None:
                    ranked.append((r, p))
            if not ranked:
                break
            _rank, (first, second) = min(ranked, key=lambda x: x[0])
            new_word: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second); i += 2
                else:
                    new_word.append(word[i]); i += 1
            word = tuple(new_word)
        result = list(word)
        self.cache[byte_piece] = result
        return result[:]

    def encode(self, text: str) -> Tokenized:
        ids: List[int] = []
        tokens: List[str] = []
        for piece in self.pretokenize(text):
            if piece in self.added_by_content:
                tokens.append(piece); ids.append(self.added_by_content[piece]); continue
            for tok in self._bpe(self._bytelevel(piece)):
                tid = self.vocab.get(tok)
                if tid is None:
                    for ch in tok:
                        if ch in self.vocab:
                            tokens.append(ch); ids.append(self.vocab[ch])
                else:
                    tokens.append(tok); ids.append(tid)
        return Tokenized(ids=ids, tokens=tokens)

    def decode_tokens(self, tokens: Sequence[str]) -> str:
        chunks: List[str] = []
        buf: List[int] = []
        def flush() -> None:
            nonlocal buf
            if buf:
                chunks.append(bytes(buf).decode("utf-8", errors="replace")); buf = []
        for tok in tokens:
            if tok in self.added_by_content:
                flush(); chunks.append(tok); continue
            for ch in tok:
                b = self.byte_decoder.get(ch)
                if b is None:
                    flush(); chunks.append(ch)
                else:
                    buf.append(b)
        flush()
        return "".join(chunks)

    def decode(self, ids: Sequence[int]) -> str:
        toks = []
        for i in ids:
            if i >= 0:
                toks.append(self.id_to_token.get(i, self.id_to_added.get(i, "")))
        return self.decode_tokens(toks)

    def analyze(self) -> dict:
        return {
            "normalizer": self.normalizer_type,
            "pre_tokenizer_type": self.pre_tokenizer.get("type"),
            "decoder": self.decoder_type,
            "model_type": self.model.get("type"),
            "vocab_size": len(self.vocab),
            "merges": len(self.model.get("merges") or []),
            "added_tokens": len(self.added_tokens),
        }

# ---------------------------------------------------------------------------
# Learned features and models: no manual intent, no synonym tables.
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    source: str
    target: str
    source_ids: List[int]
    target_ids: List[int]
    features: Counter[str]

@dataclass
class Neighbor:
    index: int
    score: float
    source: str
    target: str

@dataclass
class GenerationResult:
    prompt: str
    output: str
    neighbors: List[Neighbor]
    exact_training_target_match: bool
    longest_training_target_substring_chars: int
    timings: dict

MODEL_CACHE_VERSION = "exemplar-transducer-v2"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_signature(path: str | Path) -> str:
    p = Path(path)
    st = p.stat()
    return f"{p.resolve()}|{st.st_size}|{int(st.st_mtime_ns)}|{_sha256_file(p)}"


def cache_key(tokenizer_path: str | Path, pair_paths: Sequence[str], order: int, beam: int, branch: int) -> str:
    h = hashlib.sha256()
    h.update(f"version={MODEL_CACHE_VERSION}\n".encode())
    h.update(f"tokenizer={_file_signature(tokenizer_path)}\n".encode())
    h.update(f"order={order}|beam={beam}|branch={branch}\n".encode())
    for pair_path in pair_paths:
        h.update(f"pairs={_file_signature(pair_path)}\n".encode())
    return h.hexdigest()


def cache_paths(cache_dir: str | Path, sig: str) -> Path:
    p = Path(cache_dir).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    ext = ".pkl.gz" if os.environ.get("RSG_CACHE_COMPRESS", "").lower() in {"1", "true", "yes", "on"} else ".pkl"
    return p / f"exemplar_transducer_cache_{sig}{ext}"


def _open_cache_file(path: Path, mode: str):
    if os.environ.get("RSG_CACHE_COMPRESS", "").lower() in {"1", "true", "yes", "on"}:
        return gzip.open(path, mode)
    return open(path, mode)


def load_cached_model(cache_file: Path, tokenizer: QwenTokenizer) -> "ExemplarTransducer | None":
    if not cache_file.exists():
        return None
    try:
        with _open_cache_file(cache_file, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict) or payload.get("version") != MODEL_CACHE_VERSION:
            return None
        if payload.get("tokenizer_signature") is None:
            return None
        if payload["tokenizer_signature"] != _file_signature(tokenizer.path):
            return None
        model_payload = payload["model"]
        if not isinstance(model_payload, dict):
            return None
        return ExemplarTransducer.from_cache_payload(tokenizer, model_payload)
    except Exception:
        if os.environ.get("RSG_CACHE_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
            print(f"cache-load-failed path={cache_file}", file=sys.stderr)
        return None


def save_cached_model(cache_file: Path, tokenizer_signature: str, model_payload: dict) -> None:
    payload = {
        "version": MODEL_CACHE_VERSION,
        "tokenizer_signature": tokenizer_signature,
        "model": model_payload,
    }
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    try:
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with _open_cache_file(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, cache_file)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


class Featureizer:
    def __init__(self, tok: QwenTokenizer):
        self.tok = tok

    @staticmethod
    def _compact_for_features(text: str) -> str:
        # Remove punctuation and symbols so query variants like "안녕?", "오늘 일정은?" map stably
        # to their lexical content.
        lowered = text.lower()
        # Keep unicode word characters and Korean syllables/spaces only.
        compact = re.sub(r"[^\w\s가-힣]", " ", lowered)
        return re.sub(r"\s+", " ", compact).strip()

    def features(self, text: str) -> Counter[str]:
        text = self._compact_for_features(self.tok.normalize(text))
        f: Counter[str] = Counter()
        # Unicode char n-grams provide Korean/English surface semantics without dictionaries.
        compact = text
        for n in range(2, 7):
            for i in range(0, max(0, len(compact) - n + 1)):
                gram = compact[i:i+n]
                if gram.strip():
                    f[f"c{n}:{gram}"] += 1.0 / n
        # ByteLevel-BPE token IDs and token-ID bigrams anchor exact technical/code symbols.
        ids = self.tok.encode(text).ids
        for tid in ids:
            f[f"t:{tid}"] += 1.0
        for a, b in zip(ids, ids[1:]):
            f[f"tb:{a},{b}"] += 1.5
        return f

class SparseRetriever:
    def __init__(self, pairs: List[Pair]):
        self.pairs = pairs
        self.df: Counter[str] = Counter()
        for p in pairs:
            self.df.update(set(p.features))
        self.N = max(1, len(pairs))

    def idf(self, k: str) -> float:
        return math.log((self.N + 1.0) / (self.df.get(k, 0) + 1.0)) + 1.0

    def score(self, q: Counter[str], d: Counter[str]) -> float:
        if not q or not d:
            return 0.0
        dot = 0.0
        for k, qv in q.items():
            dv = d.get(k)
            if dv:
                w = self.idf(k)
                dot += (qv * w) * (dv * w)
        qn = math.sqrt(sum((v * self.idf(k)) ** 2 for k, v in q.items())) or 1.0
        dn = math.sqrt(sum((v * self.idf(k)) ** 2 for k, v in d.items())) or 1.0
        return dot / (qn * dn)

    def retrieve(self, q: Counter[str], k: int = 8) -> List[Neighbor]:
        rows = []
        for i, p in enumerate(self.pairs):
            s = self.score(q, p.features)
            if s > 0:
                rows.append(Neighbor(i, round(float(s), 6), p.source, p.target))
        rows.sort(key=lambda n: n.score, reverse=True)
        return rows[:k]

class VariableOrderLM:
    def __init__(self, order: int = 6):
        self.order = order
        self.counts: DefaultDict[Tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def add(self, seq: Sequence[int]) -> None:
        xs = [BOS] + list(seq) + [EOS]
        for i in range(1, len(xs)):
            nxt = xs[i]
            lo = max(0, i - self.order)
            for j in range(lo, i + 1):
                ctx = tuple(xs[j:i])
                self.counts[ctx][nxt] += 1

    def dist(self, context: Sequence[int]) -> Counter[int]:
        ctx = tuple(context[-self.order:])
        for l in range(min(self.order, len(ctx)), -1, -1):
            sub = ctx[-l:] if l else tuple()
            c = self.counts.get(sub)
            if c:
                return c.copy()
        return Counter()

class ExemplarTransducer:
    def __init__(self, tokenizer: QwenTokenizer, order: int = 6, beam: int = 8, branch: int = 16):
        self.tok = tokenizer
        self.feat = Featureizer(tokenizer)
        self.order = order
        self.beam = beam
        self.branch = branch
        self.pairs: List[Pair] = []
        self.pair_target_set: set[str] = set()
        self.retriever: SparseRetriever | None = None
        self.global_lm = VariableOrderLM(order=order)
        self.target_lms: List[VariableOrderLM] = []

    @staticmethod
    def _lm_to_payload(lm: "VariableOrderLM") -> dict:
        return {
            "order": lm.order,
            "counts": [
                [list(ctx), [[int(tid), int(cnt)] for tid, cnt in c.items()]]
                for ctx, c in lm.counts.items()
            ],
        }

    @staticmethod
    def _lm_from_payload(payload: dict) -> "VariableOrderLM":
        lm = VariableOrderLM(order=int(payload.get("order", 6)))
        counts: DefaultDict[Tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for ctx, c in payload.get("counts", []):
            counts[tuple(int(x) for x in ctx)] = Counter(
                {int(tid): int(cnt) for tid, cnt in c}
            )
        lm.counts = counts
        lm.order = int(payload.get("order", 6))
        return lm

    def to_cache_payload(self) -> dict:
        pairs_payload = [
            {
                "source": p.source,
                "target": p.target,
                "source_ids": p.source_ids,
                "target_ids": p.target_ids,
                "features": {str(k): float(v) for k, v in p.features.items()},
            }
            for p in self.pairs
        ]
        retriever_payload = None
        if self.retriever is not None:
            retriever_payload = {
                "df": {str(k): float(v) for k, v in self.retriever.df.items()},
                "N": int(self.retriever.N),
            }
        return {
            "order": self.order,
            "beam": self.beam,
            "branch": self.branch,
            "pairs": pairs_payload,
            "pair_target_set": self.pair_target_set,
            "retriever": retriever_payload,
            "global_lm": self._lm_to_payload(self.global_lm),
            "target_lms": [self._lm_to_payload(lm) for lm in self.target_lms],
        }

    @classmethod
    def from_cache_payload(cls, tokenizer: QwenTokenizer, payload: dict) -> "ExemplarTransducer":
        order = int(payload.get("order", 6))
        beam = int(payload.get("beam", 8))
        branch = int(payload.get("branch", 16))
        model = cls(tokenizer, order=order, beam=beam, branch=branch)
        model.pairs = [
            Pair(
                source=str(item.get("source", "")),
                target=str(item.get("target", "")),
                source_ids=[int(x) for x in item.get("source_ids", [])],
                target_ids=[int(x) for x in item.get("target_ids", [])],
                features=Counter({str(k): float(v) for k, v in item.get("features", {}).items()}),
            )
            for item in payload.get("pairs", [])
        ]
        model.pair_target_set = set(payload.get("pair_target_set", []))
        retr = SparseRetriever.__new__(SparseRetriever)
        retr.pairs = model.pairs
        retr.df = Counter({str(k): float(v) for k, v in (payload.get("retriever") or {}).get("df", {}).items()})
        retr.N = int((payload.get("retriever") or {}).get("N", len(model.pairs)))
        model.retriever = retr
        model.global_lm = cls._lm_from_payload(payload.get("global_lm", {}))
        model.target_lms = [cls._lm_from_payload(item) for item in payload.get("target_lms", [])]
        return model

    def fit_pairs(self, raw_pairs: Sequence[tuple[str, str]]) -> "ExemplarTransducer":
        self.pairs.clear()
        self.pair_target_set.clear()
        self.target_lms.clear()
        self.global_lm = VariableOrderLM(order=self.order)
        for source, target in raw_pairs:
            source = source.strip()
            target = target.strip()
            if not source or not target:
                continue
            src_ids = self.tok.encode(source).ids
            tgt_ids = self.tok.encode(target).ids
            p = Pair(source, target, src_ids, tgt_ids, self.feat.features(source))
            self.pairs.append(p)
            lm = VariableOrderLM(order=self.order)
            lm.add(tgt_ids)
            self.target_lms.append(lm)
            self.global_lm.add(tgt_ids)
            self.pair_target_set.add(target)
        self.retriever = SparseRetriever(self.pairs)
        return self

    @staticmethod
    def _softmax_scores(neighbors: List[Neighbor]) -> Dict[int, float]:
        if not neighbors:
            return {}
        # Data-dependent normalization; no task labels or intent branches.
        vals = [n.score for n in neighbors]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
        scale = math.sqrt(var) or (max(vals) - min(vals)) or 1.0
        exps = [math.exp((v - mu) / scale) for v in vals]
        z = sum(exps) or 1.0
        return {n.index: e / z for n, e in zip(neighbors, exps)}

    def _next_counts(self, context: Sequence[int], weights: Dict[int, float]) -> Counter[int]:
        mixed: Counter[int] = Counter()
        # Exemplar-conditioned distribution.
        for idx, w in weights.items():
            c = self.target_lms[idx].dist(context)
            total = sum(c.values()) or 1
            for tid, cnt in c.items():
                mixed[tid] += w * cnt / total
        # Data-only backoff: global LM mass shrinks automatically when exemplar mass is high.
        local_mass = sum(mixed.values())
        g = self.global_lm.dist(context)
        if g:
            gt = sum(g.values()) or 1
            backoff_mass = 1.0 / (1.0 + local_mass)
            for tid, cnt in g.items():
                mixed[tid] += backoff_mass * cnt / gt
        return mixed

    def generate(self, prompt: str, max_new_tokens: int = 120, k: int = 8) -> GenerationResult:
        if self.retriever is None:
            raise RuntimeError("fit_pairs must be called before generate")
        t0 = time.perf_counter()
        qf = self.feat.features(prompt)
        neighbors = self.retriever.retrieve(qf, k=k)
        weights = self._softmax_scores(neighbors)
        t1 = time.perf_counter()

        beams: List[tuple[float, List[int]]] = [(0.0, [])]
        finished: List[tuple[float, List[int]]] = []
        for _step in range(max_new_tokens):
            new_beams: List[tuple[float, List[int]]] = []
            for score, seq in beams:
                if seq and seq[-1] == EOS:
                    finished.append((score, seq[:-1])); continue
                ctx = [BOS] + seq
                counts = self._next_counts(ctx, weights)
                if not counts:
                    finished.append((score, seq)); continue
                total = sum(counts.values()) or 1.0
                # Top branch by learned probability only.
                for tid, val in counts.most_common(self.branch):
                    p = max(val / total, 1e-12)
                    new_seq = seq + [tid]
                    # Average log-probability length normalization prevents one-token endings from always winning.
                    new_score = score + math.log(p)
                    new_beams.append((new_score, new_seq))
            if not new_beams:
                break
            # Rank by normalized logprob, not by any task-specific rule.
            new_beams.sort(key=lambda x: x[0] / max(1, len([t for t in x[1] if t != EOS])), reverse=True)
            beams = new_beams[: self.beam]
            if len(finished) >= self.beam and all((b[1] and b[1][-1] == EOS) for b in beams):
                break
        finished.extend((s, seq[:-1] if seq and seq[-1] == EOS else seq) for s, seq in beams)
        finished.sort(key=lambda x: x[0] / max(1, len(x[1])), reverse=True)
        best_ids = finished[0][1] if finished else []
        output = self.tok.decode([i for i in best_ids if i >= 0]).strip()
        t2 = time.perf_counter()
        neighbor_indices = [n.index for n in neighbors]
        neighbor_targets = [self.pairs[i].target for i in neighbor_indices if 0 <= i < len(self.pairs)]
        exact = output in self.pair_target_set
        longest = max((longest_common_substring_len(output, t) for t in neighbor_targets), default=0) if neighbor_targets else 0
        return GenerationResult(
            prompt=prompt,
            output=output,
            neighbors=neighbors,
            exact_training_target_match=exact,
            longest_training_target_substring_chars=longest,
            timings={"retrieve_seconds": round(t1 - t0, 6), "decode_seconds": round(t2 - t1, 6)},
        )

    def stats(self) -> dict:
        return {
            "pairs": len(self.pairs),
            "global_contexts": len(self.global_lm.counts),
            "order": self.order,
            "beam": self.beam,
            "branch": self.branch,
        }


# ---------------------------------------------------------------------------
# IO and diagnostics
# ---------------------------------------------------------------------------

def longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    # O(n*m), fine for tiny diagnostics.
    prev = [0] * (len(b) + 1)
    best = 0
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                v = prev[j-1] + 1
                best = max(best, v)
                cur.append(v)
            else:
                cur.append(0)
        prev = cur
    return best


def load_pairs(paths: Sequence[str]) -> List[tuple[str, str]]:
    pairs: List[tuple[str, str]] = []
    for path in paths:
        p = Path(path)
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if "source" not in obj or "target" not in obj:
                raise ValueError(f"{path}:{line_no} needs source and target")
            pairs.append((str(obj["source"]), str(obj["target"])))
    return pairs


def ensure_model(
    tokenizer_path: str,
    pair_paths: Sequence[str],
    order: int,
    beam: int,
    branch: int,
    cache_dir: str,
    use_cache: bool = True,
) -> tuple[QwenTokenizer, ExemplarTransducer]:
    tok = QwenTokenizer(tokenizer_path)
    if not pair_paths:
        raise ValueError("pairs must include at least one file")
    if use_cache:
        sig = cache_key(tokenizer_path, pair_paths, order=order, beam=beam, branch=branch)
        cpath = cache_paths(cache_dir, sig)
        model = load_cached_model(cpath, tok)
        if model is not None:
            return tok, model
    pairs = load_pairs(pair_paths)
    model = ExemplarTransducer(tok, order=order, beam=beam, branch=branch).fit_pairs(pairs)
    if use_cache:
        try:
            save_cached_model(cache_paths(cache_dir, cache_key(tokenizer_path, pair_paths, order=order, beam=beam, branch=branch)), _file_signature(tokenizer_path), model.to_cache_payload())
        except Exception:
            if os.environ.get("RSG_CACHE_DEBUG", "").lower() in {"1", "true", "yes", "on"}:
                print(f"cache-save-failed token={tokenizer_path}", file=sys.stderr)
    return tok, model


def cmd_train_generate(args: argparse.Namespace) -> dict:
    tok, model = ensure_model(
        tokenizer_path=args.tokenizer,
        pair_paths=args.pairs,
        order=args.order,
        beam=args.beam,
        branch=args.branch,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    result = model.generate(args.prompt, max_new_tokens=args.max_new_tokens, k=args.k)
    return {
        "engine": "No-hardcoded-intent Qwen-BPE Exemplar Transducer",
        "tokenizer": tok.analyze(),
        "model": model.stats(),
        "result": asdict(result),
    }


def _chat_prompt_with_history(
    prompt: str, history: list[tuple[str, str]], context_limit: int, system_prompt: str | None
) -> str:
    if context_limit <= 0 or not history:
        return prompt
    clipped = history[-context_limit:]
    parts: List[str] = []
    if system_prompt:
        parts.append(system_prompt.strip())
    # 사용자 발화만 히스토리에 반영해, 이전 응답 텍스트가 다음 라운드 retrieval을 오염시키지 않도록 함.
    parts.append("최근 사용자 발화:")
    for user, _assistant in clipped:
        if user:
            parts.append(f"- {user}")
    parts.append(f"현재 질문: {prompt}")
    return "\n".join(parts)


def _print_help() -> None:
    sys.stdout.write(
        "\n입력 명령어:\n"
        "  /help          이 도움말\n"
        "  /clear         대화 이력 초기화\n"
        "  /stats         모델 통계 출력\n"
        "  /exit 또는 /quit  대화 종료\n"
        "  비어있지 않은 빈 줄 입력은 무시됩니다.\n\n"
    )
    sys.stdout.flush()


def cmd_chat(args: argparse.Namespace) -> int:
    tok, model = ensure_model(
        tokenizer_path=args.tokenizer,
        pair_paths=args.pairs,
        order=args.order,
        beam=args.beam,
        branch=args.branch,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
    )
    sys.stdout.write("RSG chat 시작: 종료는 /exit 또는 /quit\n")
    if args.system_prompt:
        sys.stdout.write(f"[system_prompt]\n{args.system_prompt}\n")
    _print_help()
    sys.stdout.flush()

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_text = input("user> ").strip()
        except EOFError:
            return 0
        except KeyboardInterrupt:
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return 0
        if user_text == "/help":
            _print_help()
            continue
        if user_text == "/clear":
            history.clear()
            sys.stdout.write("대화 이력이 초기화되었습니다.\n")
            sys.stdout.flush()
            continue
        if user_text == "/stats":
            info = model.stats()
            sys.stdout.write(json.dumps({"status": "ok", "model": info}, ensure_ascii=False, indent=2) + "\n")
            sys.stdout.flush()
            continue

        prompt = _chat_prompt_with_history(
            user_text,
            history,
            context_limit=args.history_turns,
            system_prompt=args.system_prompt,
        )
        result = model.generate(prompt, max_new_tokens=args.max_new_tokens, k=args.k)
        output = result.output.strip() or "(empty output)"

        if args.json:
            payload = {
                "status": "ok",
                "result": asdict(result),
            }
            sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        else:
            sys.stdout.write(f"assistant> {output}\n")
            if args.show_neighbors:
                sys.stdout.write(f"neighbors={len(result.neighbors)}\n")
            if args.show_timings:
                sys.stdout.write(f"timings={result.timings}\n")
            sys.stdout.flush()
        history.append((user_text, output))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="No-hardcoded-intent Qwen-BPE emergent language prototype")
    ap.add_argument("--tokenizer", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--pairs", nargs="+", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--k", type=int, default=8)
    g.add_argument("--order", type=int, default=6)
    g.add_argument("--beam", type=int, default=8)
    g.add_argument("--branch", type=int, default=16)
    g.add_argument("--max-new-tokens", type=int, default=120)
    g.add_argument("--cache-dir", default="~/.cache/neurova_rsg", help="모델 캐시 디렉터리")
    g.add_argument("--no-cache", action="store_true", help="캐시를 사용하지 않음")
    g.add_argument("--out", default="")

    c = sub.add_parser("chat", help="chat loop from the same model")
    c.add_argument("--pairs", nargs="+", required=True)
    c.add_argument("--k", type=int, default=8)
    c.add_argument("--order", type=int, default=6)
    c.add_argument("--beam", type=int, default=8)
    c.add_argument("--branch", type=int, default=16)
    c.add_argument("--max-new-tokens", type=int, default=120)
    c.add_argument("--cache-dir", default="~/.cache/neurova_rsg", help="모델 캐시 디렉터리")
    c.add_argument("--no-cache", action="store_true", help="캐시를 사용하지 않음")
    c.add_argument("--history-turns", type=int, default=0, help="최근 사용자 발화 몇 턴을 프롬프트에 반영할지(기본 0: 히스토리 미반영)")
    c.add_argument("--system-prompt", default="", help="대화 프롬프트 앞에 붙일 시스템 문구")
    c.add_argument("--json", action="store_true", help="JSON 결과로 출력")
    c.add_argument("--show-neighbors", action="store_true", help="JSON 아닌 출력에서 후보 유사도 개수 출력")
    c.add_argument("--show-timings", action="store_true", help="JSON 아닌 출력에서 timings 출력")
    c.add_argument("--out", default="", help="chat은 /stats, /help 외 출력 저장은 미지원")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        payload = cmd_train_generate(args)
    elif args.cmd == "chat":
        return cmd_chat(args)
    else:
        raise ValueError(args.cmd)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if getattr(args, "out", ""):
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
