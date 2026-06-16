from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import LUMAConfig, LUMALM
from .tokenizer import assert_tokenizer_contract, build_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained LUMA checkpoint.")
    p.add_argument("--ckpt", default="luma/runs/luma-smoke/model.pt")
    p.add_argument("--prompt", default="Memory page:\nMina owns the blue key.\nQuestion: What object belongs to Mina?\nAnswer:")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-repeat-ngram", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    payload = torch.load(Path(args.ckpt), map_location=args.device, weights_only=True)
    raw_cfg = payload["config"]
    cfg = LUMAConfig(**raw_cfg)
    tokenizer = build_tokenizer(cfg.tokenizer_backend, cfg.qwen_tokenizer_path, cfg.bytepatch_vocab_path)
    assert_tokenizer_contract(raw_cfg, tokenizer)
    model = LUMALM(cfg).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids[-512:]], dtype=torch.long, device=args.device)
    out = model(x, return_slots=True)
    slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
    for _ in range(args.max_new):
        logits = out.logits[0, -1]
        for special_id in {tokenizer.pad_id, tokenizer.bos_id} - {tokenizer.eos_id}:
            if 0 <= int(special_id) < logits.numel():
                logits[int(special_id)] = -torch.inf
        if args.no_repeat_ngram > 0 and len(ids) >= args.no_repeat_ngram - 1:
            prefix = tuple(ids[-(args.no_repeat_ngram - 1) :]) if args.no_repeat_ngram > 1 else tuple()
            banned = set()
            for start in range(0, len(ids) - args.no_repeat_ngram + 1):
                ngram = tuple(ids[start : start + args.no_repeat_ngram])
                if args.no_repeat_ngram == 1 or ngram[:-1] == prefix:
                    banned.add(ngram[-1])
            for token_id in banned:
                if 0 <= token_id < logits.numel() and token_id != tokenizer.eos_id:
                    logits[token_id] = -torch.inf
        if args.greedy:
            next_id = int(torch.argmax(logits).item())
        else:
            probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        x = torch.tensor([[next_id]], dtype=torch.long, device=args.device)
        out = model(x, slots_in=slots, return_slots=True)
        slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
