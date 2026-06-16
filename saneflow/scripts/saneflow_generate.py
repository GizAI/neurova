#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
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
    p = argparse.ArgumentParser(description="Generate from SaneFlowLM.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new", type=int, default=120)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--no-repeat-ngram-size", type=int, default=0)
    p.add_argument("--decode", choices=["full", "cache"], default="cache")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--chatml", action="store_true", help="Wrap --prompt as a ChatML user turn and strip the ChatML tail.")
    p.add_argument("--plain-chat", action="store_true", help="Wrap --prompt as 'User: ...\\nAssistant:' and stop on a new User turn.")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    payload = torch.load(Path(args.ckpt), map_location="cpu", weights_only=True)
    cfg = SaneFlowConfig(**payload["config"])
    tokenizer = SaneFlowBPETokenizer(cfg.tokenizer_path)
    im_end_id = tokenizer.token_to_id(IM_END)
    model = SaneFlowLM(cfg).to(device=args.device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()

    prompt_text = f"User: {args.prompt.strip()}\nAssistant:" if args.plain_chat else (format_chatml_user_prompt(args.prompt, args.system) if args.chatml else args.prompt)
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not ids:
        ids = [tokenizer.bos_token_id]
    generated: list[int] = []
    start = time.perf_counter()
    with torch.inference_mode():
        cache = None
        if args.decode == "cache":
            prompt = torch.tensor([ids[-args.context :]], dtype=torch.long, device=args.device)
            logits, cache = model.forward_step(prompt, None)
            logits = logits[0]
            for _ in range(args.max_new):
                logits = filter_logits(
                    logits,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    generated_ids=generated,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                if next_id == tokenizer.eos_token_id or (args.chatml and im_end_id is not None and next_id == im_end_id):
                    break
                ids.append(next_id)
                generated.append(next_id)
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                if args.chatml and IM_END in decoded:
                    break
                if args.plain_chat and "\nUser:" in decoded:
                    break
                logits, cache = model.forward_step(torch.tensor([next_id], dtype=torch.long, device=args.device), cache)
                logits = logits[0]
        else:
            for _ in range(args.max_new):
                x = torch.tensor([ids[-args.context :]], dtype=torch.long, device=args.device)
                logits = model(x)[0, -1]
                logits = filter_logits(
                    logits,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    generated_ids=generated,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                )
                probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
                next_id = int(torch.multinomial(probs, 1).item())
                if next_id == tokenizer.eos_token_id or (args.chatml and im_end_id is not None and next_id == im_end_id):
                    break
                ids.append(next_id)
                generated.append(next_id)
                decoded = tokenizer.decode(generated, skip_special_tokens=True)
                if args.chatml and IM_END in decoded:
                    break
                if args.plain_chat and "\nUser:" in decoded:
                    break
    elapsed = max(time.perf_counter() - start, 1e-9)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if args.chatml:
        text = strip_chatml_tail(text)
    if args.plain_chat:
        text = text.split("\nUser:", 1)[0].strip()
    if text:
        print(text, end="", flush=True)
    print(f"\n({len(generated) / elapsed:.1f} tok/s)")


if __name__ == "__main__":
    main()
