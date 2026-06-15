from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import Qwen36_27B_TextConfig
from .dflash import DFlashDraftAdapter
from .generate import GenerationConfig, build_chat_ids, choose_weight_device, load_tokenizer, sample_next
from .loader import QuantizedStore
from .model import QwenBurstModel
from .state import DecodeState


def eos_token_ids(tokenizer) -> tuple[int, ...]:
    ids = []
    for name in ("eos_token_id", "pad_token_id"):
        val = getattr(tokenizer, name, None)
        if isinstance(val, int):
            ids.append(val)
    return tuple(set(ids))


@torch.no_grad()
def target_only(model, state, prompt_ids, max_new, eos):
    logits = None
    for i, tid in enumerate(prompt_ids):
        logits = model.forward_one(tid, state, return_logits=(i == len(prompt_ids) - 1))
    assert logits is not None
    next_id = sample_next(logits, GenerationConfig(eos_token_ids=eos))
    out = []
    for _ in range(max_new):
        if eos and next_id in eos:
            break
        out.append(next_id)
        logits = model.forward_one(next_id, state, return_logits=True)
        next_id = sample_next(logits, GenerationConfig(eos_token_ids=eos))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure qwenburst target-only vs native DFlash speculative path")
    ap.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    ap.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb3"))
    ap.add_argument("--dflash-draft-dir", type=Path, required=True)
    ap.add_argument("--prompt", default="Say hello.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--recent-window", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gpu-embed-head", action="store_true")
    ap.add_argument("--block-size", type=int, default=None)
    args = ap.parse_args()

    cfg = Qwen36_27B_TextConfig.from_hf_config(args.hf_model)
    tokenizer = load_tokenizer(args.hf_model)
    prompt_ids = build_chat_ids(tokenizer, args.prompt)
    eos = eos_token_ids(tokenizer)

    weight_device = choose_weight_device(args.qb_model, "auto", args.device)
    store = QuantizedStore(args.qb_model, device=weight_device)
    embed_store = None if args.gpu_embed_head else QuantizedStore(args.qb_model, device="cpu")
    head_store = None if args.gpu_embed_head else QuantizedStore(args.qb_model, device="cpu")
    model = QwenBurstModel(store, cfg=cfg, device=args.device, embed_store=embed_store, head_store=head_store)
    draft = DFlashDraftAdapter.from_lowbit_dir(args.dflash_draft_dir, device=args.device)

    state = DecodeState.allocate(cfg, max_seq_len=args.recent_window, device=args.device, kv_window_policy="ring")
    t0 = time.perf_counter()
    base_ids = target_only(model, state, prompt_ids, args.max_new_tokens, eos)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    state = DecodeState.allocate(cfg, max_seq_len=args.recent_window, device=args.device, kv_window_policy="ring")
    dflash_ids = []
    accepted = []
    t2 = time.perf_counter()
    for tid, acc_len in draft.generate(
        model,
        state,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos,
        block_size=args.block_size,
    ):
        dflash_ids.append(tid)
        accepted.append(acc_len)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    base_dt = max(t1 - t0, 1e-9)
    dflash_dt = max(t3 - t2, 1e-9)
    print("target_text:", tokenizer.decode(base_ids, skip_special_tokens=True))
    print("dflash_text:", tokenizer.decode(dflash_ids, skip_special_tokens=True))
    print(f"target_only generated={len(base_ids)} elapsed={base_dt:.3f}s tok/s={len(base_ids)/base_dt:.2f}")
    print(f"dflash generated={len(dflash_ids)} elapsed={dflash_dt:.3f}s tok/s={len(dflash_ids)/dflash_dt:.2f}")
    if accepted:
        print(f"dflash_accept avg={sum(accepted)/len(accepted):.2f} max={max(accepted)} steps={len(accepted)}")
    print(f"speedup={((len(dflash_ids)/dflash_dt)/(len(base_ids)/base_dt) if base_ids and dflash_ids else 0):.3f}x")


if __name__ == "__main__":
    main()
