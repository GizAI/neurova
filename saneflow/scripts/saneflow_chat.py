#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.model import SaneFlowConfig, SaneFlowLM
from saneflow.tokenizer import SaneFlowBPETokenizer
from saneflow.chat_format import DEFAULT_SYSTEM, IM_END, format_chatml_user_prompt, strip_chatml_tail
from saneflow.decoding import filter_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive streaming chat for SaneFlowLM.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--max-new", type=int, default=160)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--decode", choices=["full", "cache"], default="cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--prompt", default="")
    p.add_argument("--chatml", action="store_true", help="Wrap each user prompt as a ChatML user turn.")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    return p.parse_args()

@torch.no_grad()
def stream_generate(model, tokenizer, prompt: str, args: argparse.Namespace) -> None:
    prompt_text = format_chatml_user_prompt(prompt, args.system) if args.chatml else prompt
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.bos_token_id]
    generated_text = ""
    printed_len = 0
    im_end_id = tokenizer.token_to_id(IM_END)

    def emit_token(token_id: int) -> bool:
        nonlocal generated_text, printed_len
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        if not text:
            return False
        generated_text += text
        visible = strip_chatml_tail(generated_text) if args.chatml else generated_text
        if args.chatml and IM_END not in generated_text:
            visible = visible[: max(0, len(visible) - len(IM_END) + 1)]
        if len(visible) > printed_len:
            print(visible[printed_len:], end="", flush=True)
            printed_len = len(visible)
        return args.chatml and IM_END in generated_text

    cache = None
    if args.decode == "cache":
        x = torch.tensor([ids[-args.context :]], dtype=torch.long, device=args.device)
        logits, cache = model.forward_step(x, None)
        logits = logits[0]
        generated_ids: list[int] = []
        for _ in range(args.max_new):
            logits = filter_logits(
                logits,
                top_k=args.top_k,
                top_p=args.top_p,
                generated_ids=generated_ids,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            if next_id == tokenizer.eos_token_id or (args.chatml and im_end_id is not None and next_id == im_end_id):
                break
            ids.append(next_id)
            generated_ids.append(next_id)
            if emit_token(next_id):
                break
            logits, cache = model.forward_step(torch.tensor([next_id], dtype=torch.long, device=args.device), cache)
            logits = logits[0]
    else:
        generated_ids: list[int] = []
        for _ in range(args.max_new):
            x = torch.tensor([ids[-args.context :]], dtype=torch.long, device=args.device)
            logits = model(x)[0, -1]
            logits = filter_logits(
                logits,
                top_k=args.top_k,
                top_p=args.top_p,
                generated_ids=generated_ids,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            if next_id == tokenizer.eos_token_id or (args.chatml and im_end_id is not None and next_id == im_end_id):
                break
            ids.append(next_id)
            generated_ids.append(next_id)
            if emit_token(next_id):
                break
    print(flush=True)


def main() -> None:
    args = parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    payload = torch.load(Path(args.ckpt), map_location="cpu", weights_only=True)
    cfg = SaneFlowConfig(**payload["config"])
    tokenizer = SaneFlowBPETokenizer(cfg.tokenizer_path)
    model = SaneFlowLM(cfg).to(device=args.device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()

    if args.prompt:
        stream_generate(model, tokenizer, args.prompt, args)
        return

    print("SaneFlow streaming chat. Ctrl-D to exit.", flush=True)
    while True:
        try:
            prompt = input("\nuser> ").strip()
        except EOFError:
            print()
            return
        if not prompt:
            continue
        print("saneflow> ", end="", flush=True)
        stream_generate(model, tokenizer, prompt, args)


if __name__ == "__main__":
    main()
