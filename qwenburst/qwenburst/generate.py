from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import time

import torch

from .config import Qwen36_27B_TextConfig
from .loader import QuantizedStore
from .model import QwenBurstModel
from .state import DecodeState


@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_k: int = 0
    eos_token_ids: tuple[int, ...] = ()


def sample_next(logits: torch.Tensor, cfg: GenerationConfig) -> int:
    if cfg.temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    scores = logits.float() / max(cfg.temperature, 1e-6)
    if cfg.top_k and cfg.top_k > 0:
        vals, idx = torch.topk(scores, min(cfg.top_k, scores.numel()))
        probs = torch.softmax(vals, dim=-1)
        return int(idx[torch.multinomial(probs, 1)].item())
    probs = torch.softmax(scores, dim=-1)
    return int(torch.multinomial(probs, 1).item())


class QwenBurstGenerator:
    """Minimal first-real-chat generator.

    This is intentionally correctness-first and single-sequence.  It gives the
    engine a real tokenizer → prefill → decode → EOS path before CUDA/MTP speed
    work resumes.
    """

    def __init__(self, model: QwenBurstModel, state: DecodeState):
        self.model = model
        self.state = state

    def forward_one(self, token: int, *, return_logits: bool = True):
        try:
            return self.model.forward_one(int(token), self.state, use_mtp=False, return_logits=return_logits)
        except TypeError:
            return self.model.forward_one(int(token), self.state, use_mtp=False)

    @torch.no_grad()
    def prefill(self, input_ids: Iterable[int]) -> torch.Tensor:
        ids = [int(tid) for tid in input_ids]
        if not ids:
            raise ValueError("prefill requires at least one token")
        logits: torch.Tensor | None = None
        for i, tid in enumerate(ids):
            logits = self.forward_one(tid, return_logits=(i == len(ids) - 1))
        assert logits is not None
        return logits

    @torch.no_grad()
    def generate_ids(self, prompt_ids: Iterable[int], gen_cfg: GenerationConfig) -> Iterator[int]:
        logits = self.prefill(prompt_ids)
        next_id = sample_next(logits, gen_cfg)
        for _ in range(gen_cfg.max_new_tokens):
            if gen_cfg.eos_token_ids and next_id in gen_cfg.eos_token_ids:
                break
            yield next_id
            logits = self.forward_one(next_id, return_logits=True)
            next_id = sample_next(logits, gen_cfg)


def load_tokenizer(model_dir: str | Path):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("transformers is required for qwenburst-chat") from exc
    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


def build_chat_ids(tokenizer, prompt: str, system: str | None = None) -> list[int]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if isinstance(encoded, dict):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.reshape(-1).tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            encoded = encoded[0]
        return [int(t) for t in encoded]
    text = (system + "\n" if system else "") + prompt
    return tokenizer.encode(text)


def choose_weight_device(qb_model: Path, requested: str, runtime_device: str) -> str:
    if requested != "auto":
        return runtime_device if requested == "cuda" else "cpu"
    import json

    index = json.loads((qb_model / "qwenburst_index.json").read_text(encoding="utf-8"))
    if any(meta.get("kind") == "lowbit_marlin_groupwise" for meta in index.get("tensors", {}).values()):
        return runtime_device
    bits = {
        int(meta["bits"])
        for meta in index.get("tensors", {}).values()
        if meta.get("kind") == "lowbit_symmetric_groupwise"
    }
    return runtime_device if bits and max(bits) <= 3 else "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="First-real-chat greedy runner for QwenBurst low-bit checkpoints")
    ap.add_argument("--hf-model", required=True, type=Path, help="Original HF model dir, used for tokenizer/config")
    ap.add_argument("--qb-model", required=True, type=Path, help="Converted QwenBurst low-bit model dir")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=8192)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print prefill/decode timing to stderr")
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto", help="where low-bit layer weights live between matvecs")
    ap.add_argument("--gpu-embed-head", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cpu-embed", action="store_true", help="offload only token embeddings to CPU if 16GB VRAM is too tight")
    args = ap.parse_args()

    cfg = Qwen36_27B_TextConfig.from_hf_config(args.hf_model) if (args.hf_model / "config.json").exists() else Qwen36_27B_TextConfig()
    tokenizer = load_tokenizer(args.hf_model)
    weight_device = choose_weight_device(args.qb_model, args.weight_device, args.device)
    store = QuantizedStore(args.qb_model, device=weight_device)
    embed_store = QuantizedStore(args.qb_model, device="cpu") if args.cpu_embed else None
    head_store = None
    model = QwenBurstModel(store, cfg=cfg, device=args.device, embed_store=embed_store, head_store=head_store)
    state = DecodeState.allocate(cfg, max_seq_len=args.recent_window, device=args.device, kv_window_policy="ring")
    runner = QwenBurstGenerator(model, state)

    prompt_ids = build_chat_ids(tokenizer, args.prompt, args.system)
    eos = []
    for name in ("eos_token_id", "pad_token_id"):
        val = getattr(tokenizer, name, None)
        if isinstance(val, int):
            eos.append(val)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_ids=tuple(set(eos)),
    )

    out_ids: list[int] = []
    t0 = time.perf_counter()
    for tid in runner.generate_ids(prompt_ids, gen_cfg):
        out_ids.append(tid)
        if args.stream:
            print(tokenizer.decode([tid], skip_special_tokens=False), end="", flush=True)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    if args.stream:
        print()
    else:
        print(tokenizer.decode(out_ids, skip_special_tokens=True))
    if args.stats:
        import sys
        dt = max(t1 - t0, 1e-9)
        print(f"[qwenburst] generated={len(out_ids)} elapsed={dt:.3f}s tok/s={len(out_ids)/dt:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
