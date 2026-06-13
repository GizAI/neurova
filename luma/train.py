from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import SyntheticMemoryDataset
from .data import PackedTextDataset
from .data import RecordTextDataset
from .data import IGNORE_INDEX
from .model import LUMAConfig, LUMALM
from .tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small LUMA memory-edit language model.")
    p.add_argument("--out", default="runs/luma-smoke")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--data", nargs="*", default=[])
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--init-from", default="")
    p.add_argument("--dataset-mode", choices=["packed", "records"], default="packed")
    p.add_argument("--answer-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = ByteTokenizer()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    cfg = LUMAConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layer=args.layers,
        n_slots=args.slots,
        topk=args.topk,
        chunk_size=args.chunk,
    )
    model = LUMALM(cfg).to(device=device, dtype=dtype)
    if args.init_from:
        payload = torch.load(args.init_from, map_location=device, weights_only=True)
        saved_cfg = LUMAConfig(**payload["config"])
        if saved_cfg.to_dict() != cfg.to_dict():
            raise ValueError(f"--init-from config mismatch: saved={saved_cfg.to_dict()} current={cfg.to_dict()}")
        model.load_state_dict(payload["model"])
        print(json.dumps({"init_from": args.init_from}), flush=True)
    if args.data:
        if args.dataset_mode == "records":
            data = RecordTextDataset(
                tokenizer=tokenizer,
                paths=[Path(item) for item in args.data],
                seq_len=args.seq_len,
                max_records=args.max_records,
                answer_only=args.answer_only,
            )
        else:
            data = PackedTextDataset(
                tokenizer=tokenizer,
                paths=[Path(item) for item in args.data],
                seq_len=args.seq_len,
                max_records=args.max_records,
            )
        tokens = int(data.rows.numel()) if hasattr(data, "rows") else int(data.ids.numel())
        print(json.dumps({"dataset": args.dataset_mode, "answer_only": args.answer_only, "records": data.records, "tokens": tokens}), flush=True)
    else:
        data = SyntheticMemoryDataset(tokenizer=tokenizer, seq_len=args.seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log_path = out / "train_log.jsonl"
    model.train()
    for step in range(1, args.steps + 1):
        x, y = data.batch(args.batch_size, device)
        out_obj = model(x, return_aux=True)
        lm_loss = F.cross_entropy(out_obj.logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=IGNORE_INDEX)
        entropy = out_obj.aux["slot_entropy"] if out_obj.aux else torch.tensor(0.0, device=device)
        loss = lm_loss - 0.001 * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            row = {"step": step, "loss": round(float(loss.item()), 4), "lm_loss": round(float(lm_loss.item()), 4)}
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
    ckpt = {"config": cfg.to_dict(), "model": model.state_dict()}
    torch.save(ckpt, out / "model.pt")
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
