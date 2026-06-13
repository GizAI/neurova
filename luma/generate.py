from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .model import LUMAConfig, LUMALM
from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with a trained LUMA checkpoint.")
    p.add_argument("--ckpt", default="runs/luma-smoke/model.pt")
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
    tokenizer = ByteTokenizer()
    payload = torch.load(Path(args.ckpt), map_location=args.device, weights_only=True)
    model = LUMALM(LUMAConfig(**payload["config"])).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    for _ in range(args.max_new):
        x = torch.tensor([ids[-512:]], dtype=torch.long, device=args.device)
        logits = model(x).logits[0, -1]
        logits[tokenizer.pad_id] = -torch.inf
        logits[tokenizer.bos_id] = -torch.inf
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
    print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
