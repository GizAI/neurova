#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.data import TokenStreamDataset
from saneflow.model import SaneFlowConfig, SaneFlowLM
from saneflow.tokenizer import SaneFlowBPETokenizer
from scripts.saneflow_train import lm_loss, pick_dtype


@torch.no_grad()
def eval_path(
    *,
    model: SaneFlowLM,
    tokenizer: SaneFlowBPETokenizer,
    cfg: SaneFlowConfig,
    path: Path,
    seq_len: int,
    batch_size: int,
    batches: int,
    device: torch.device,
    loss_chunk_tokens: int,
) -> dict:
    data = TokenStreamDataset(
        tokenizer=tokenizer,
        paths=[path],
        seq_len=seq_len,
        max_records=2000,
        dataset_device=torch.device("cpu"),
        loss_mode="causal",
    )
    losses = []
    for _ in range(batches):
        x, y = data.batch(batch_size, device)
        hidden = model.forward_hidden(x, activation_checkpointing=False)
        losses.append(float(lm_loss(model, hidden, y, cfg.vocab_size, loss_chunk_tokens).item()))
    mean = sum(losses) / max(1, len(losses))
    return {"path": str(path), "loss": mean, "batches": len(losses), "tokens": batches * batch_size * seq_len}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-source validation losses for DoReMi-style data mixing.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_practical_pretrain_mix.json"))
    parser.add_argument("--out", type=Path, default=Path("saneflow/data/corpus/mixes/saneflow_practical_pretrain_v1/source_losses.json"))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--loss-chunk-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)
    payload = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg = SaneFlowConfig(**payload["config"])
    tokenizer = SaneFlowBPETokenizer(cfg.tokenizer_path)
    model = SaneFlowLM(cfg).to(device=device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    results = {}
    for source in recipe["sources"]:
        path = Path(source["valid"])
        if not path.exists() or path.stat().st_size == 0:
            results[source["name"]] = {"path": str(path), "loss": None, "error": "missing_or_empty"}
            continue
        try:
            results[source["name"]] = eval_path(
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                path=path,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                batches=args.batches,
                device=device,
                loss_chunk_tokens=args.loss_chunk_tokens,
            )
        except Exception as exc:
            results[source["name"]] = {"path": str(path), "loss": None, "error": str(exc)}

    out = {
        "ckpt": args.ckpt,
        "recipe": str(args.recipe),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "source_losses": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
