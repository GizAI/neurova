"""
Neurova Tiny LLM — local byte-level GPT with test-time training.

This is intentionally small and dependency-light:
- no external generation API
- byte-level tokenizer, works for English/Korean/any UTF-8 text
- train from plain text, dialogue logs, Neurova memories, or Wikipedia snippets
- test-time update from corrections/examples

It will NOT be smart before training. It is a minimal local generator substrate
that can be trained and adapted by Neurova.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any
import argparse
import json
import math
import os
import random
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# Byte vocab: 0..255 raw UTF-8 bytes, plus specials.
PAD = 256
BOS = 257
EOS = 258
VOCAB_SIZE = 259


class ByteTokenizer:
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8", errors="replace"))
        if add_bos:
            ids = [BOS] + ids
        if add_eos:
            ids = ids + [EOS]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        bs = bytearray()
        for i in ids:
            if 0 <= int(i) < 256:
                bs.append(int(i))
        return bs.decode("utf-8", errors="replace")


@dataclass
class TinyGPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.05


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size:]
            if targets is not None:
                targets = targets[:, -self.cfg.block_size:]
            T = self.cfg.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 80, stop_ids: Optional[set] = None):
        stop_ids = stop_ids or {EOS}
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if int(next_id[0, 0]) in stop_ids:
                break
        return idx


def _require_torch():
    if torch is None:
        raise RuntimeError(f"PyTorch is required for tiny_llm.py: {_TORCH_IMPORT_ERROR}")


def format_dialogue(user: str, assistant: str) -> str:
    return f"<user> {user.strip()}\n<assistant> {assistant.strip()}\n"


class TextDataset:
    def __init__(self, texts: List[str], tok: ByteTokenizer, block_size: int):
        ids: List[int] = []
        for t in texts:
            if t.strip():
                ids.extend(tok.encode(t, add_bos=True, add_eos=True))
        if len(ids) < 2:
            ids = tok.encode("<user> hello\n<assistant> hello\n", add_bos=True, add_eos=True)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def sample_batch(self, batch_size: int, device: str):
        max_start = max(0, len(self.data) - self.block_size - 1)
        starts = torch.randint(0, max_start + 1, (batch_size,)) if max_start > 0 else torch.zeros(batch_size, dtype=torch.long)
        x = torch.stack([self.data[s:s+self.block_size] for s in starts])
        y = torch.stack([self.data[s+1:s+self.block_size+1] for s in starts])
        return x.to(device), y.to(device)


def build_bootstrap_corpus(extra_texts: Optional[List[str]] = None) -> List[str]:
    base = [
        format_dialogue("Hello.", "Hello. I am Neurova."),
        format_dialogue("Who are you?", "I am Neurova."),
        format_dialogue("I am Kyungtae.", "Got it. You are Kyungtae."),
        format_dialogue("Who am I?", "You are Kyungtae."),
        format_dialogue("SpaceX was founded by Elon Musk in 2002.", "Got it. SpaceX was founded by Elon Musk in 2002."),
        format_dialogue("Who founded SpaceX?", "SpaceX was founded by Elon Musk."),
        format_dialogue("When was SpaceX founded?", "SpaceX was founded in 2002."),
        format_dialogue("Korea is in East Asia.", "Got it. Korea is in East Asia."),
        format_dialogue("Where is Korea?", "Korea is in East Asia."),
        format_dialogue("John gave Mary the apple.", "Got it. Mary has the apple."),
        format_dialogue("Who has the apple?", "Mary has the apple."),
        format_dialogue("I don't know.", "Teach me with a correction and I will adapt."),
    ]
    if extra_texts:
        base.extend(extra_texts)
    return base


class TinyLLMRuntime:
    def __init__(self, model_dir: str, device: Optional[str] = None, cfg: Optional[TinyGPTConfig] = None):
        _require_torch()
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = ByteTokenizer()
        if (self.model_dir / "config.json").exists() and (self.model_dir / "model.pt").exists():
            meta = json.loads((self.model_dir / "config.json").read_text())
            self.cfg = TinyGPTConfig(**meta["config"])
            self.model = TinyGPT(self.cfg).to(self.device)
            self.model.load_state_dict(torch.load(self.model_dir / "model.pt", map_location=self.device))
        else:
            self.cfg = cfg or TinyGPTConfig()
            self.model = TinyGPT(self.cfg).to(self.device)
        self.model.eval()

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.model_dir / "model.pt")
        (self.model_dir / "config.json").write_text(json.dumps({"config": asdict(self.cfg), "saved_at": time.time()}, indent=2))

    def train_texts(self, texts: List[str], steps: int = 500, batch_size: int = 16, lr: float = 3e-4, log_every: int = 50):
        ds = TextDataset(texts, self.tok, self.cfg.block_size)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.model.train()
        last_loss = None
        for step in range(1, steps + 1):
            x, y = ds.sample_batch(batch_size, self.device)
            _, loss = self.model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.item())
            if log_every and step % log_every == 0:
                print(f"step {step}/{steps} loss={last_loss:.4f}")
        self.model.eval()
        return last_loss

    def ttt_update_dialogue(self, user: str, assistant: str, steps: int = 20, lr: float = 1e-4):
        """Test-time training from one correction/example."""
        return self.train_texts([format_dialogue(user, assistant)], steps=steps, batch_size=1, lr=lr, log_every=0)

    @torch.no_grad()
    def complete(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.8, top_k: int = 80) -> str:
        self.model.eval()
        ids = self.tok.encode(prompt, add_bos=True)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
        gen = self.tok.decode(out[0].tolist()[len(ids):])
        return gen

    def chat(self, user_text: str, context: str = "", max_new_tokens: int = 160, temperature: float = 0.8) -> str:
        prompt = ""
        if context.strip():
            prompt += f"<context> {context.strip()}\n"
        prompt += f"<user> {user_text.strip()}\n<assistant>"
        gen = self.complete(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        # Stop at next user/context marker to keep turns clean.
        for stop in ["<user>", "<context>", "\n<user", "\n<context"]:
            if stop in gen:
                gen = gen.split(stop, 1)[0]
        return gen.strip() or "..."


def load_text_files(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            continue
        if path.is_dir():
            for q in sorted(path.rglob("*.txt")):
                out.append(q.read_text(encoding="utf-8", errors="ignore"))
            for q in sorted(path.rglob("*.md")):
                out.append(q.read_text(encoding="utf-8", errors="ignore"))
        else:
            if path.suffix.lower() == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    out.append(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    out.append(path.read_text(encoding="utf-8", errors="ignore"))
            else:
                out.append(path.read_text(encoding="utf-8", errors="ignore"))
    return out


def main(argv: Optional[List[str]] = None):
    _require_torch()
    ap = argparse.ArgumentParser(description="Neurova local TinyGPT")
    sub = ap.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--model-dir", default=".neurova_tiny_llm")
    train.add_argument("--data", nargs="*", default=[])
    train.add_argument("--steps", type=int, default=500)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--block-size", type=int, default=512)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--embd", type=int, default=256)

    chat = sub.add_parser("chat")
    chat.add_argument("--model-dir", default=".neurova_tiny_llm")
    chat.add_argument("--temperature", type=float, default=0.8)

    args = ap.parse_args(argv)
    if args.cmd == "train":
        cfg = TinyGPTConfig(block_size=args.block_size, n_layer=args.layers, n_head=args.heads, n_embd=args.embd)
        rt = TinyLLMRuntime(args.model_dir, cfg=cfg)
        texts = build_bootstrap_corpus(load_text_files(args.data))
        loss = rt.train_texts(texts, steps=args.steps, batch_size=args.batch_size, lr=args.lr)
        rt.save()
        print(f"saved {args.model_dir}; final_loss={loss}")
    elif args.cmd == "chat":
        rt = TinyLLMRuntime(args.model_dir)
        print("TinyLLM chat. Commands: /exit, /learn <assistant-answer>")
        last_user = ""
        while True:
            try:
                u = input(">>> ").strip()
            except EOFError:
                break
            if not u:
                continue
            if u in {"/exit", "exit", "quit"}:
                break
            if u.startswith("/learn "):
                ans = u[len("/learn "):].strip()
                if last_user and ans:
                    rt.ttt_update_dialogue(last_user, ans, steps=25)
                    rt.save()
                    print("[learned]")
                else:
                    print("No previous user turn.")
                continue
            last_user = u
            print(rt.chat(u, temperature=args.temperature))


if __name__ == "__main__":
    main()
