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

from luma.model import LUMAConfig, LUMALM
from luma.tokenizer import LUMATokenizer, assert_tokenizer_contract, build_tokenizer
from luma.chat_format import IM_END, IM_START


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive streaming chat for a trained LUMA checkpoint.")
    parser.add_argument("--ckpt", default="luma/runs/luma_current/model.pt")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new", type=int, default=160)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram", type=int, default=4)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--system", default="You are LUMA, a concise helpful assistant. Answer directly.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ablation", choices=["normal", "no_slots", "random_slot_keys", "no_copy", "no_slots_no_copy", "no_local_attention"], default="normal")
    return parser.parse_args()


def stop_ids(tokenizer: LUMATokenizer) -> set[int]:
    ids = {int(tokenizer.eos_id)}
    for text in (IM_END, f"{IM_START}user", f"{IM_START}system"):
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return ids


def pick_dtype(name: str, device: str) -> torch.dtype:
    if name == "fp32" or device == "cpu":
        return torch.float32
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def apply_sampling_filters(
    logits: torch.Tensor,
    generated: list[int],
    tokenizer: LUMATokenizer,
    *,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram: int,
) -> torch.Tensor:
    logits = logits.clone()
    logits[tokenizer.pad_id] = -torch.inf
    logits[tokenizer.bos_id] = -torch.inf

    if repetition_penalty > 1.0:
        for token_id in set(generated[-256:]):
            if 0 <= token_id < logits.numel() and token_id != tokenizer.eos_id:
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

    if no_repeat_ngram > 0 and len(generated) >= no_repeat_ngram - 1:
        prefix = tuple(generated[-(no_repeat_ngram - 1) :]) if no_repeat_ngram > 1 else tuple()
        banned: set[int] = set()
        for start in range(0, len(generated) - no_repeat_ngram + 1):
            ngram = tuple(generated[start : start + no_repeat_ngram])
            if no_repeat_ngram == 1 or ngram[:-1] == prefix:
                banned.add(ngram[-1])
        for token_id in banned:
            if 0 <= token_id < logits.numel() and token_id != tokenizer.eos_id:
                logits[token_id] = -torch.inf

    if top_k > 0 and top_k < logits.numel():
        cutoff = torch.topk(logits, top_k).values[-1]
        logits[logits < cutoff] = -torch.inf

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
        keep[0] = True
        filtered = torch.full_like(logits, -torch.inf)
        filtered[sorted_idx[keep]] = logits[sorted_idx[keep]]
        logits = filtered

    return logits


@torch.no_grad()
def generate_stream(
    model: LUMALM,
    tokenizer: LUMATokenizer,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[str, int, float, list | None]:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids[-args.context :]], dtype=torch.long, device=args.device)
    out = model(x, return_slots=True, ablation=args.ablation)
    slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
    return generate_from_state(model, tokenizer, ids, out, slots, args)


@torch.no_grad()
def generate_from_state(
    model: LUMALM,
    tokenizer: LUMATokenizer,
    ids: list[int],
    out,
    slots,
    args: argparse.Namespace,
) -> tuple[str, int, float, list | None]:
    generated: list[int] = []
    printed = ""
    started = time.perf_counter()
    stops = stop_ids(tokenizer)

    for _ in range(args.max_new):
        logits = out.logits[0, -1]
        logits = apply_sampling_filters(
            logits,
            ids,
            tokenizer,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram=args.no_repeat_ngram,
        )
        if args.greedy:
            next_id = int(torch.argmax(logits).item())
        else:
            probs = F.softmax(logits / max(args.temperature, 1e-4), dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        if next_id in stops:
            break
        ids.append(next_id)
        generated.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=args.device)
        out = model(x, slots_in=slots, return_slots=True, ablation=args.ablation)
        slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None

        decoded = tokenizer.decode(generated)
        if "\ufffd" in decoded:
            stable = decoded.split("\ufffd", 1)[0]
        else:
            stable = decoded
        delta = stable[len(printed) :]
        if delta:
            print(delta, end="", flush=True)
            printed = stable

    final = tokenizer.decode(generated)
    tail = final[len(printed) :]
    if tail:
        print(tail, end="", flush=True)
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(f"\n({len(generated) / elapsed:.1f} tok/s)")
    return final, len(generated), elapsed, slots


def load_model(args: argparse.Namespace) -> tuple[LUMALM, LUMATokenizer]:
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"LUMA checkpoint not found: {ckpt}")
    dtype = pick_dtype(args.dtype, args.device)
    payload = torch.load(ckpt, map_location=args.device, weights_only=True)
    raw_cfg = payload["config"]
    cfg = LUMAConfig(**raw_cfg)
    tokenizer = build_tokenizer(cfg.tokenizer_backend, cfg.qwen_tokenizer_path, cfg.bytepatch_vocab_path)
    assert_tokenizer_contract(raw_cfg, tokenizer)
    model = LUMALM(cfg).to(device=args.device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, tokenizer


def format_prompt(system: str, history: list[tuple[str, str]], user_text: str) -> str:
    parts = [f"{IM_START}system\n{system.strip()}{IM_END}\n"]
    for user, assistant in history[-6:]:
        parts.append(f"{IM_START}user\n{user.strip()}{IM_END}\n{IM_START}assistant\n{assistant.strip()}{IM_END}\n")
    parts.append(f"{IM_START}user\n{user_text.strip()}{IM_END}\n{IM_START}assistant\n")
    return "".join(parts)


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model(args)

    if args.prompt or args.once:
        prompt = format_prompt(args.system, [], args.prompt or "Hello")
        generate_stream(model, tokenizer, prompt, args)
        return

    print(f"LUMA streaming chat: {args.ckpt}")
    print("Type /exit to quit.")
    prefix_ids = tokenizer.encode(f"{IM_START}system\n{args.system.strip()}{IM_END}\n", add_bos=True, add_eos=False)
    x = torch.tensor([prefix_ids[-args.context :]], dtype=torch.long, device=args.device)
    out = model(x, return_slots=True, ablation=args.ablation)
    slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        turn = f"{IM_START}user\n{user_text.strip()}{IM_END}\n{IM_START}assistant\n"
        turn_ids = tokenizer.encode(turn, add_bos=False, add_eos=False)
        x = torch.tensor([turn_ids], dtype=torch.long, device=args.device)
        out = model(x, slots_in=slots, return_slots=True, ablation=args.ablation)
        slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
        print("LUMA> ", end="", flush=True)
        _, _, _, slots = generate_from_state(model, tokenizer, turn_ids.copy(), out, slots, args)


if __name__ == "__main__":
    main()
