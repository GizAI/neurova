#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.data import TokenStreamDataset
from saneflow.model import SaneFlowConfig, SaneFlowLM
from saneflow.tokenizer import SaneFlowBPETokenizer
from scripts.saneflow_train import CombinedOptim, Muon, lm_loss, pick_dtype


def normalize(weights: torch.Tensor) -> torch.Tensor:
    return weights.clamp_min(1e-8) / weights.clamp_min(1e-8).sum()


def source_paths(recipe: dict, split: str) -> dict[str, Path]:
    return {source["name"]: Path(source[split]) for source in recipe["sources"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small DoReMi proxy with prior or excess-loss DRO domain weights.")
    parser.add_argument("--recipe", type=Path, default=Path("configs/saneflow_practical_pretrain_mix.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/doremi_proxy_practical_v1"))
    parser.add_argument("--reference", default="", help="Optional reference checkpoint trained on prior/default ratios.")
    parser.add_argument("--mode", choices=["prior", "dro"], default="dro")
    parser.add_argument("--tokenizer-path", default="tokenizers/saneflow_fineweb_edu_16k")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=0.012)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dro-eta", type=float, default=0.08)
    parser.add_argument("--loss-chunk-tokens", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--activation-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.tf32 and torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    args.out.mkdir(parents=True, exist_ok=True)
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    paths = source_paths(recipe, "train")
    names = [source["name"] for source in recipe["sources"] if paths[source["name"]].exists() and paths[source["name"]].stat().st_size > 0]
    if not names:
        raise SystemExit("no non-empty sources available for DoReMi proxy training")
    priors = torch.tensor([float(next(s["ratio"] for s in recipe["sources"] if s["name"] == name)) for name in names])
    priors = normalize(priors)

    tokenizer = SaneFlowBPETokenizer(args.tokenizer_path)
    cfg = SaneFlowConfig(
        vocab_size=len(tokenizer),
        model_type="dense_transformer",
        d_model=args.d_model,
        n_layer=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        d_ff=args.d_ff,
        rope_theta=100000.0,
        qk_norm=True,
        tokenizer_path=args.tokenizer_path,
    )
    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)
    datasets = {
        name: TokenStreamDataset(
            tokenizer=tokenizer,
            paths=[paths[name]],
            seq_len=args.seq_len,
            max_records=5000,
            dataset_device=torch.device("cpu"),
            loss_mode="causal",
        )
        for name in names
    }
    model = SaneFlowLM(cfg).to(device=device, dtype=dtype)
    reference = None
    if args.mode == "dro" and not args.reference:
        raise SystemExit("--mode dro requires --reference so weights use excess loss against a prior-ratio model")
    if args.reference:
        payload = torch.load(args.reference, map_location=device, weights_only=True)
        ref_cfg = SaneFlowConfig(**payload["config"])
        reference = SaneFlowLM(ref_cfg).to(device=device, dtype=dtype)
        reference.load_state_dict(payload["model"])
        reference.eval()

    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
            muon_params.append(param)
        else:
            adam_params.append(param)
    opt = CombinedOptim(
        [
            Muon(muon_params, lr=args.muon_lr, weight_decay=args.weight_decay) if muon_params else None,
            torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=torch.cuda.is_available())
            if adam_params
            else None,
        ]
    )

    q = priors.to(device=device, dtype=torch.float32)
    q_sum = torch.zeros_like(q)
    log_path = args.out / "train_log.jsonl"
    start_time = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        domain_losses = []
        reference_losses = []
        for name in names:
            x, y = datasets[name].batch(args.batch_size, device)
            hidden = model.forward_hidden(x, activation_checkpointing=args.activation_checkpointing)
            loss = lm_loss(model, hidden, y, cfg.vocab_size, args.loss_chunk_tokens)
            domain_losses.append(loss)
            if reference is not None:
                with torch.no_grad():
                    ref_hidden = reference.forward_hidden(x, activation_checkpointing=False)
                    ref_loss = lm_loss(reference, ref_hidden, y, reference.cfg.vocab_size, args.loss_chunk_tokens)
                reference_losses.append(ref_loss.detach())
            else:
                reference_losses.append(torch.zeros_like(loss))
        losses = torch.stack(domain_losses)
        ref_losses = torch.stack(reference_losses)
        if args.mode == "prior":
            excess = losses.detach()
            q = priors.to(device=device, dtype=torch.float32)
            objective = (q.detach() * losses).sum()
        else:
            excess = losses.detach() - ref_losses.detach()
            q = normalize(q * torch.exp(args.dro_eta * excess.float()))
            objective = (q.detach() * (losses - ref_losses)).sum()
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        q_sum += q.detach()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            avg_q = normalize(q_sum / step)
            row = {
                "step": step,
                "mode": args.mode,
                "objective": round(float(objective.item()), 4),
                "elapsed_s": round(time.perf_counter() - start_time, 1),
                "weights": {name: round(float(avg_q[i].item()), 6) for i, name in enumerate(names)},
                "losses": {name: round(float(losses[i].item()), 4) for i, name in enumerate(names)},
                "excess": {name: round(float(excess[i].item()), 4) for i, name in enumerate(names)},
            }
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step}, args.out / "latest.pt")

    avg_q = normalize(q_sum / max(1, args.steps))
    weights = {name: float(avg_q[i].item()) for i, name in enumerate(names)}
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": args.steps}, args.out / "model.pt")
    (args.out / "doremi_ratios.json").write_text(
        json.dumps(
            {
                "method": "prior_ratio_proxy" if args.mode == "prior" else "doremi_excess_loss_group_dro_proxy",
                "recipe": str(args.recipe),
                "reference": args.reference,
                "proxy": str(args.out / "model.pt"),
                "mode": args.mode,
                "ratios": weights,
                "priors": {name: float(priors[i].item()) for i, name in enumerate(names)},
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
