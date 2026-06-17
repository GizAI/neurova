#!/usr/bin/env python3
"""Tiny CPU GRU seq2seq over compacted Qwen token IDs.
No intent rules, no templates: source->target behavior is learned from pairs.
"""
from __future__ import annotations

import argparse, json, random, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from emergent_rsg_lm import QwenTokenizer, load_pairs, longest_common_substring_len

PAD, BOS, EOS, UNK = 0, 1, 2, 3

torch.set_num_threads(1)

@dataclass
class Seq2SeqResult:
    prompt: str
    output: str
    exact_training_target_match: bool
    longest_training_target_substring_chars: int
    loss_final: float
    epochs: int
    timings: dict

class TinyGRUSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 96, hidden: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.enc = nn.GRU(d_model, hidden, batch_first=True)
        self.dec = nn.GRU(d_model, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_len = src.ne(PAD).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), src_len, batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        y, _ = self.dec(self.emb(tgt_in), h)
        return self.out(y)

    def encode_hidden(self, src: torch.Tensor) -> torch.Tensor:
        src_len = src.ne(PAD).sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(src), src_len, batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return h

class LearnedQwenGRUTransducer:
    def __init__(self, tok: QwenTokenizer, max_len: int = 160):
        self.tok = tok
        self.max_len = max_len
        self.qid_to_cid = {}
        self.cid_to_qid = {}
        self.model: TinyGRUSeq2Seq | None = None
        self.targets_text: List[str] = []

    def _build_vocab(self, pairs: Sequence[Tuple[str, str]]):
        ids = set()
        for s, t in pairs:
            ids.update(self.tok.encode(s).ids)
            ids.update(self.tok.encode(t).ids)
        for cid, qid in enumerate(sorted(ids), start=4):
            self.qid_to_cid[qid] = cid
            self.cid_to_qid[cid] = qid

    def _enc(self, text: str) -> List[int]:
        raw = self.tok.encode(text).ids[: self.max_len - 2]
        return [BOS] + [self.qid_to_cid.get(i, UNK) for i in raw] + [EOS]

    @staticmethod
    def _pad(batch: List[List[int]]) -> torch.Tensor:
        m = max(len(x) for x in batch)
        return torch.tensor([x + [PAD] * (m - len(x)) for x in batch], dtype=torch.long)

    def fit(self, pairs: Sequence[Tuple[str, str]], epochs: int = 500, lr: float = 0.003, seed: int = 7) -> float:
        random.seed(seed); torch.manual_seed(seed)
        self.targets_text = [t for _, t in pairs]
        self._build_vocab(pairs)
        vocab_size = max(self.cid_to_qid.keys(), default=3) + 1
        self.model = TinyGRUSeq2Seq(vocab_size=vocab_size)
        src = self._pad([self._enc(s) for s, _ in pairs])
        tgt = self._pad([self._enc(t) for _, t in pairs])
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        last = 0.0
        for _ in range(epochs):
            self.model.train(); opt.zero_grad(set_to_none=True)
            logits = self.model(src, tgt_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=PAD)
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step()
            last = float(loss.detach())
        return last

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 140) -> str:
        assert self.model is not None
        self.model.eval()
        src = self._pad([self._enc(prompt)])
        h = self.model.encode_hidden(src)
        cur = torch.tensor([[BOS]], dtype=torch.long)
        out_ids: List[int] = []
        for _ in range(max_new_tokens):
            y, h = self.model.dec(self.model.emb(cur), h)
            logits = self.model.out(y[:, -1])
            cid = int(logits.argmax(dim=-1).item())
            if cid == EOS:
                break
            out_ids.append(cid)
            cur = torch.tensor([[cid]], dtype=torch.long)
        qids = [self.cid_to_qid.get(cid) for cid in out_ids if cid >= 4]
        return self.tok.decode([q for q in qids if q is not None]).strip()

def run(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()
    tok = QwenTokenizer(args.tokenizer)
    pairs = load_pairs(args.pairs)
    learner = LearnedQwenGRUTransducer(tok, max_len=args.max_len)
    loss = learner.fit(pairs, epochs=args.epochs, lr=args.lr, seed=args.seed)
    t1 = time.perf_counter()
    out = learner.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    t2 = time.perf_counter()
    result = Seq2SeqResult(
        prompt=args.prompt,
        output=out,
        exact_training_target_match=any(out == t for _, t in pairs),
        longest_training_target_substring_chars=max((longest_common_substring_len(out, t) for _, t in pairs), default=0),
        loss_final=round(loss, 6),
        epochs=args.epochs,
        timings={"train_seconds": round(t1-t0, 4), "decode_seconds": round(t2-t1, 4)},
    )
    return {
        "engine": "Tiny CPU GRU seq2seq over compacted Qwen-BPE IDs",
        "tokenizer": tok.analyze(),
        "training_pairs": len(pairs),
        "compact_vocab_size": max(learner.cid_to_qid.keys(), default=3) + 1,
        "result": asdict(result),
    }

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--max-new-tokens", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    payload = run(args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
