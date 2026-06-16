from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import SyntheticMemoryDataset
from .data import SyntheticSlotProofDataset
from .data import PackedTextDataset
from .data import RecordTextDataset
from .data import WeightedMixedDataset
from .data import IGNORE_INDEX
from .model import LUMAConfig, LUMALM
from .tokenizer import build_tokenizer, tokenizer_fingerprint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small LUMA memory-edit language model.")
    p.add_argument("--recipe", choices=["custom", "memory_proof", "mixed_chat"], default="custom")
    p.add_argument("--out", default="luma/runs/luma-smoke")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--local-heads", type=int, default=4)
    p.add_argument("--disable-slots", action="store_true")
    p.add_argument("--disable-local-attention", action="store_true")
    p.add_argument("--copy-window", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--data", nargs="*", default=[])
    p.add_argument("--raw-data", nargs="*", default=[])
    p.add_argument("--qa-data", nargs="*", default=[])
    p.add_argument("--chat-data", nargs="*", default=[])
    p.add_argument("--memory-data", nargs="*", default=[])
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--max-raw-records", type=int, default=0)
    p.add_argument("--max-qa-records", type=int, default=0)
    p.add_argument("--max-chat-records", type=int, default=0)
    p.add_argument("--max-memory-records", type=int, default=0)
    p.add_argument("--init-from", default="")
    p.add_argument("--dataset-mode", choices=["packed", "records"], default="packed")
    p.add_argument("--raw-dataset-mode", choices=["packed", "records"], default="packed")
    p.add_argument("--raw-answer-only", action="store_true")
    p.add_argument("--answer-only", action="store_true")
    p.add_argument("--raw-weight", type=float, default=0.45)
    p.add_argument("--qa-weight", type=float, default=0.0)
    p.add_argument("--chat-weight", type=float, default=0.25)
    p.add_argument("--memory-weight", type=float, default=0.30)
    p.add_argument("--slot-proof-weight", type=float, default=0.0)
    p.add_argument("--slot-proof-gap-lines", type=int, default=8)
    p.add_argument("--tokenizer-backend", choices=["byte", "bytepatch", "qwen", "qwen35"], default="bytepatch")
    p.add_argument("--qwen-tokenizer-path", default="luma/tokenizers/qwen35")
    p.add_argument("--bytepatch-vocab-path", default="luma/tokenizers/luma_bytepatch/bytepatch_vocab.json")
    p.add_argument("--slot-entropy-weight", type=float, default=0.0)
    p.add_argument("--slot-usage-weight", type=float, default=0.0)
    p.add_argument("--overwrite-penalty-weight", type=float, default=0.0)
    p.add_argument("--ablation-margin-weight", type=float, default=0.0)
    p.add_argument("--ablation-margin", type=float, default=0.25)
    p.add_argument("--memory-logit-weight", type=float, default=0.0)
    p.add_argument("--memory-read-bias", type=float, default=-6.0)
    p.add_argument("--token-read-topk", type=int, default=8)
    p.add_argument("--ablation-probe-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=0, help="Save latest.pt every N steps; 0 disables interim saves.")
    return p.parse_args()


def apply_recipe(args: argparse.Namespace) -> None:
    if args.recipe == "custom":
        return
    if args.recipe == "memory_proof":
        args.raw_weight = 0.0
        args.chat_weight = 0.0
        args.memory_weight = 0.0
        args.slot_proof_weight = 1.0
        if args.ablation_margin_weight == 0.0:
            args.ablation_margin_weight = 1.0
        if args.memory_logit_weight == 0.0:
            args.memory_logit_weight = 1.0
        if args.memory_read_bias == -6.0:
            args.memory_read_bias = -2.0
        return
    if args.recipe == "mixed_chat":
        args.raw_weight = 0.35
        args.chat_weight = 0.20
        args.memory_weight = 0.25
        args.slot_proof_weight = 0.20
        return
    raise ValueError(f"unknown recipe={args.recipe!r}")


def main() -> None:
    args = parse_args()
    apply_recipe(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer = build_tokenizer(args.tokenizer_backend, args.qwen_tokenizer_path, args.bytepatch_vocab_path)
    tok_sha = tokenizer_fingerprint(
        args.tokenizer_backend,
        qwen_path=args.qwen_tokenizer_path,
        bytepatch_vocab_path=args.bytepatch_vocab_path,
    )
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    cfg = LUMAConfig(
        vocab_size=tokenizer.vocab_size,
        tokenizer_backend=args.tokenizer_backend,
        qwen_tokenizer_path=args.qwen_tokenizer_path,
        bytepatch_vocab_path=args.bytepatch_vocab_path,
        tokenizer_sha256=tok_sha,
        d_model=args.d_model,
        n_layer=args.layers,
        n_slots=args.slots,
        topk=args.topk,
        chunk_size=args.chunk,
        local_heads=args.local_heads,
        copy_window=args.copy_window,
        memory_read_bias=args.memory_read_bias,
        token_read_topk=args.token_read_topk,
        use_slots=not args.disable_slots,
        use_local_attention=not args.disable_local_attention and args.local_heads > 0,
    )
    model = LUMALM(cfg).to(device=device, dtype=dtype)
    if args.init_from:
        payload = torch.load(args.init_from, map_location=device, weights_only=True)
        saved_cfg = LUMAConfig(**payload["config"])
        if saved_cfg.to_dict() != cfg.to_dict():
            raise ValueError(f"--init-from config mismatch: saved={saved_cfg.to_dict()} current={cfg.to_dict()}")
        model.load_state_dict(payload["model"])
        print(json.dumps({"init_from": args.init_from}), flush=True)
    dataset_summary = None
    if args.raw_data or args.qa_data or args.chat_data or args.memory_data or args.slot_proof_weight > 0:
        components = []
        if args.raw_data and args.raw_weight > 0:
            if args.raw_dataset_mode == "records":
                raw = RecordTextDataset(
                    tokenizer=tokenizer,
                    paths=[Path(item) for item in args.raw_data],
                    seq_len=args.seq_len,
                    max_records=args.max_raw_records or args.max_records,
                    answer_only=args.raw_answer_only,
                )
            else:
                raw = PackedTextDataset(
                    tokenizer=tokenizer,
                    paths=[Path(item) for item in args.raw_data],
                    seq_len=args.seq_len,
                    max_records=args.max_raw_records or args.max_records,
                )
            components.append(("raw", raw, args.raw_weight))
        if args.qa_data and args.qa_weight > 0:
            qa = RecordTextDataset(
                tokenizer=tokenizer,
                paths=[Path(item) for item in args.qa_data],
                seq_len=args.seq_len,
                max_records=args.max_qa_records or args.max_records,
                answer_only=True,
            )
            components.append(("qa", qa, args.qa_weight))
        if args.chat_data and args.chat_weight > 0:
            chat = RecordTextDataset(
                tokenizer=tokenizer,
                paths=[Path(item) for item in args.chat_data],
                seq_len=args.seq_len,
                max_records=args.max_chat_records or args.max_records,
                answer_only=True,
            )
            components.append(("chat", chat, args.chat_weight))
        if args.memory_data and args.memory_weight > 0:
            memory = RecordTextDataset(
                tokenizer=tokenizer,
                paths=[Path(item) for item in args.memory_data],
                seq_len=args.seq_len,
                max_records=args.max_memory_records or args.max_records,
                answer_only=True,
            )
            components.append(("memory", memory, args.memory_weight))
        if args.slot_proof_weight > 0:
            slot_proof = SyntheticSlotProofDataset(
                tokenizer=tokenizer,
                seq_len=args.seq_len,
                gap_lines=args.slot_proof_gap_lines,
            )
            components.append(("slot_proof", slot_proof, args.slot_proof_weight))
        data = WeightedMixedDataset(components)
        dataset_summary = data.summary()
        print(json.dumps({
            "dataset": "mixed",
            "recipe": args.recipe,
            "components": dataset_summary,
            "records": data.records,
            "tokenizer_backend": args.tokenizer_backend,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256": tok_sha,
        }), flush=True)
    elif args.data:
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
        dataset_summary = {
            "dataset": args.dataset_mode,
            "answer_only": args.answer_only,
            "records": data.records,
            "tokens": tokens,
        }
        print(json.dumps({
            "dataset": args.dataset_mode,
            "recipe": args.recipe,
            "answer_only": args.answer_only,
            "records": data.records,
            "tokens": tokens,
            "tokenizer_backend": args.tokenizer_backend,
            "vocab_size": tokenizer.vocab_size,
            "tokenizer_sha256": tok_sha,
        }), flush=True)
    else:
        data = SyntheticMemoryDataset(tokenizer=tokenizer, seq_len=args.seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log_path = out / "train_log.jsonl"

    train_metadata = {
        "args": vars(args),
        "dataset_summary": dataset_summary,
        "tokenizer_backend": args.tokenizer_backend,
        "tokenizer_sha256": tok_sha,
    }

    def save_checkpoint(path: Path, step: int) -> None:
        ckpt = {
            "config": cfg.to_dict(),
            "model": model.state_dict(),
            "step": step,
            "train_metadata": train_metadata,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = data.batch(args.batch_size, device)
        out_obj = model(x, return_aux=True)
        lm_loss = F.cross_entropy(out_obj.logits.reshape(-1, cfg.vocab_size), y.reshape(-1), ignore_index=IGNORE_INDEX)
        memory_loss = torch.tensor(0.0, device=device)
        if args.memory_logit_weight > 0:
            memory_loss = F.cross_entropy(
                out_obj.memory_logits.reshape(-1, cfg.vocab_size),
                y.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        no_slots_loss = None
        random_keys_loss = None
        margin_loss = torch.tensor(0.0, device=device)
        if args.ablation_margin_weight > 0:
            with torch.no_grad():
                no_slots = model(x, ablation="no_slots")
                random_keys = model(x, ablation="random_slot_keys")
                no_slots_loss = F.cross_entropy(
                    no_slots.logits.reshape(-1, cfg.vocab_size),
                    y.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                random_keys_loss = F.cross_entropy(
                    random_keys.logits.reshape(-1, cfg.vocab_size),
                    y.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
            target = torch.minimum(no_slots_loss, random_keys_loss).detach()
            margin_loss = F.relu(lm_loss - target + args.ablation_margin)
        entropy = out_obj.aux["slot_entropy"] if out_obj.aux else torch.tensor(0.0, device=device)
        usage_entropy = out_obj.aux["slot_usage_entropy"] if out_obj.aux else torch.tensor(0.0, device=device)
        overwrite = out_obj.aux["overwrite_rate"] if out_obj.aux else torch.tensor(0.0, device=device)
        loss = (
            lm_loss
            + args.memory_logit_weight * memory_loss
            + args.ablation_margin_weight * margin_loss
            - args.slot_entropy_weight * entropy
            - args.slot_usage_weight * usage_entropy
            + args.overwrite_penalty_weight * overwrite
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 10 == 0 or step == args.steps:
            if (
                no_slots_loss is None
                and args.ablation_probe_every > 0
                and (step == 1 or step % args.ablation_probe_every == 0 or step == args.steps)
            ):
                with torch.no_grad():
                    no_slots = model(x, ablation="no_slots")
                    random_keys = model(x, ablation="random_slot_keys")
                    no_slots_loss = F.cross_entropy(
                        no_slots.logits.reshape(-1, cfg.vocab_size),
                        y.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
                    random_keys_loss = F.cross_entropy(
                        random_keys.logits.reshape(-1, cfg.vocab_size),
                        y.reshape(-1),
                        ignore_index=IGNORE_INDEX,
                    )
            slot_top = []
            if out_obj.aux and "slot_usage" in out_obj.aux:
                usage = out_obj.aux["slot_usage"].detach().cpu()
                top_counts, top_idx = usage.topk(k=min(8, usage.size(-1)), dim=-1)
                slot_top = [
                    [[int(i), int(c)] for i, c in zip(top_idx[layer].tolist(), top_counts[layer].tolist()) if c > 0]
                    for layer in range(usage.size(0))
                ]
            row = {
                "step": step,
                "loss": round(float(loss.item()), 4),
                "lm_loss": round(float(lm_loss.item()), 4),
                "memory_lm_loss": round(float(memory_loss.item()), 4),
                "ablation_margin_loss": round(float(margin_loss.item()), 4),
                "no_slots_lm_loss": round(float(no_slots_loss.item()), 4) if no_slots_loss is not None else None,
                "random_slot_keys_lm_loss": round(float(random_keys_loss.item()), 4) if random_keys_loss is not None else None,
                "slot_entropy": round(float(entropy.item()), 4),
                "slot_usage_entropy": round(float(usage_entropy.item()), 4),
                "slot_update_frequency": round(float(out_obj.aux["slot_update_frequency"].item()), 4) if out_obj.aux else 0.0,
                "slot_overwrite_rate": round(float(overwrite.item()), 4),
                "slot_delta": round(float(out_obj.aux["slot_delta"].item()), 4) if out_obj.aux else 0.0,
                "slot_confidence_mean": round(float(out_obj.aux["confidence_mean"].item()), 4) if out_obj.aux else 0.0,
                "slot_utility_mean": round(float(out_obj.aux["utility_mean"].item()), 4) if out_obj.aux else 0.0,
                "slot_read_gate_mean": round(float(out_obj.aux["read_gate_mean"].item()), 4) if out_obj.aux else 0.0,
                "fact_pool_entropy": round(float(out_obj.aux["pool_entropy"].item()), 4) if out_obj.aux else 0.0,
                "slot_topk_histogram": slot_top,
            }
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            save_checkpoint(out / "latest.pt", step)
    save_checkpoint(out / "model.pt", args.steps)
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
