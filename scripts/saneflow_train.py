#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.data import SampleAlignedDataset, TokenStreamDataset
from saneflow.model import SaneFlowConfig, SaneFlowLM
from saneflow.tokenizer import SaneFlowBPETokenizer


class AdEMAMix(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float) -> None:
        super().__init__(
            params,
            dict(lr=lr, weight_decay=weight_decay, beta1=0.9, beta2=0.999, beta3=0.9999, alpha=5.0, eps=1e-8),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                m_fast = state["m_fast"]
                m_slow = state["m_slow"]
                v = state["v"]
                m_fast.lerp_(grad, 1.0 - beta1)
                m_slow.lerp_(grad, 1.0 - beta3)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                update = m_fast / (1.0 - beta1**step) + alpha * m_slow / (1.0 - beta3**step)
                denom = (v / (1.0 - beta2**step)).sqrt().add_(eps)
                p.addcdiv_(update, denom, value=-lr)
        return loss


def orthogonalize_update(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    shape = g.shape
    x = g.float()
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
    if transposed:
        x = x.T
    return x.reshape(shape).to(dtype=g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.0, ns_steps: int = 5) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.mul_(1.0 - lr * wd)
                state = self.state[p]
                if not state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(momentum).add_(p.grad)
                update = orthogonalize_update(buf, ns_steps)
                fan_out, fan_in = update.shape[0], max(1, update.numel() // update.shape[0])
                scale = max(1.0, fan_out / fan_in) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss


class CombinedOptim:
    def __init__(self, optimizers) -> None:
        self.optimizers = [o for o in optimizers if o is not None]
        self.param_groups = [g for o in self.optimizers for g in o.param_groups]

    def zero_grad(self, *args, **kwargs) -> None:
        for opt in self.optimizers:
            opt.zero_grad(*args, **kwargs)

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SaneFlowLM from scratch.")
    p.add_argument("--out", default="runs/saneflow_smoke")
    p.add_argument("--train-data", nargs="+", required=True)
    p.add_argument("--valid-data", nargs="*", default=[])
    p.add_argument("--tokenizer-path", default="tokenizers/saneflow_tinystories_16k")
    p.add_argument("--init-from", default="", help="Optional checkpoint to continue/fine-tune from.")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--d-embed", type=int, default=0)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--conv-kernel", type=int, default=5)
    p.add_argument("--syntax-mix-version", choices=["v1", "v2"], default="v1")
    p.add_argument("--syntax-kernels", default="3,7,15")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", choices=["adamw", "ademamix", "muon"], default="adamw")
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--loss-chunk-tokens", type=int, default=4096)
    p.add_argument("--loss-mode", choices=["causal", "chatml_assistant"], default="causal")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--state-mixer-version", choices=["v1", "v2", "v2_fixed", "off", "delta_matrix"], default="v2")
    p.add_argument("--state-clip", type=float, default=0.0)
    p.add_argument("--state-zoneout", type=float, default=0.0)
    p.add_argument("--attention-interval", type=int, default=0)
    p.add_argument("--attention-window", type=int, default=64)
    p.add_argument("--thought-slots", type=int, default=0)
    p.add_argument("--thought-chunk", type=int, default=1)
    p.add_argument("--thought-start-layer", type=int, default=0)
    p.add_argument("--landmark-interval", type=int, default=0)
    p.add_argument("--landmark-chunk", type=int, default=64)
    p.add_argument("--landmark-max", type=int, default=64)
    p.add_argument("--dataset-device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dataset-layout", choices=["stream", "sample"], default="stream")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    p.add_argument("--activation-checkpointing", action="store_true")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--fused-adamw", action="store_true")
    return p.parse_args()


def pick_dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


@torch.no_grad()
def eval_loss(
    model: SaneFlowLM,
    data: TokenStreamDataset,
    cfg: SaneFlowConfig,
    batch_size: int,
    device: torch.device,
    loss_chunk_tokens: int,
    loss_mode: str,
) -> float:
    model.eval()
    losses = []
    for _ in range(8):
        if loss_mode == "causal":
            x, y = data.batch(batch_size, device)
            mask = None
        else:
            x, y, mask = data.batch_with_mask(batch_size, device)
        hidden = model.forward_hidden(x, activation_checkpointing=False)
        losses.append(lm_loss(model, hidden, y, cfg.vocab_size, loss_chunk_tokens, mask).item())
    model.train()
    return sum(losses) / len(losses)


def lm_loss(
    model: SaneFlowLM,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    chunk_tokens: int,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_mask is not None:
        flat_mask = target_mask.reshape(-1)
        selected = torch.nonzero(flat_mask, as_tuple=False).flatten()
        if selected.numel() == 0:
            return hidden.sum() * 0.0
        flat_hidden = hidden.reshape(-1, hidden.shape[-1]).index_select(0, selected)
        flat_targets = targets.reshape(-1).index_select(0, selected)
        token_count = flat_targets.numel()
        if chunk_tokens <= 0 or token_count <= chunk_tokens:
            logits = model.logits_from_hidden(flat_hidden)
            return F.cross_entropy(logits.reshape(-1, vocab_size), flat_targets)
        total = hidden.new_zeros(())
        for start in range(0, token_count, chunk_tokens):
            end = min(start + chunk_tokens, token_count)
            logits = model.logits_from_hidden(flat_hidden[start:end])
            total = total + F.cross_entropy(logits, flat_targets[start:end], reduction="sum")
        return total / hidden.new_tensor(float(token_count))

    if chunk_tokens <= 0:
        logits = model.logits_from_hidden(hidden)
        return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    total = hidden.new_zeros(())
    token_count = flat_targets.numel()
    for start in range(0, token_count, chunk_tokens):
        end = min(start + chunk_tokens, flat_targets.numel())
        logits = model.logits_from_hidden(flat_hidden[start:end])
        total = total + F.cross_entropy(logits, flat_targets[start:end], reduction="sum")
    return total / hidden.new_tensor(float(token_count))


def load_init_state_allow_vocab_resize(model: SaneFlowLM, payload: dict, old_vocab_size: int, new_vocab_size: int) -> None:
    state = payload["model"]
    model_state = model.state_dict()
    adapted = {}
    resized: list[str] = []
    for name, tensor in state.items():
        target = model_state.get(name)
        if target is None:
            continue
        if tensor.shape == target.shape:
            adapted[name] = tensor
            continue
        if name in {"embed.weight", "lm_head.weight"} and tensor.ndim == 2 and target.ndim == 2 and tensor.shape[1] == target.shape[1]:
            if tensor.shape[0] != old_vocab_size or target.shape[0] != new_vocab_size:
                raise ValueError(f"unexpected vocab resize shape for {name}: {tuple(tensor.shape)} -> {tuple(target.shape)}")
            merged = target.detach().clone()
            take = min(tensor.shape[0], target.shape[0])
            merged[:take].copy_(tensor[:take].to(dtype=target.dtype, device=target.device))
            if target.shape[0] > take:
                # Initialize newly added control-token rows near the old embedding
                # distribution instead of leaving arbitrary random extremes.
                mean = tensor[:take].mean(dim=0, keepdim=True).to(dtype=target.dtype, device=target.device)
                std = tensor[:take].std(dim=0, keepdim=True).mean().to(dtype=target.dtype, device=target.device).clamp_min(1e-4)
                merged[take:].copy_(mean + 0.02 * std * torch.randn_like(merged[take:]))
            adapted[name] = merged
            resized.append(name)
            continue
        raise ValueError(f"--init-from tensor shape mismatch for {name}: {tuple(tensor.shape)} -> {tuple(target.shape)}")
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    if unexpected:
        raise ValueError(f"unexpected init state keys: {unexpected}")
    non_vocab_missing = [x for x in missing if x not in {"embed.weight", "lm_head.weight"}]
    if non_vocab_missing:
        raise ValueError(f"missing init state keys after vocab resize: {non_vocab_missing[:20]}")
    print(json.dumps({"init_from_vocab_resize": {"old_vocab_size": old_vocab_size, "new_vocab_size": new_vocab_size, "resized": resized}}), flush=True)


def main() -> None:
    args = parse_args()
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    tokenizer = SaneFlowBPETokenizer(args.tokenizer_path)
    syntax_kernels = tuple(int(x.strip()) for x in args.syntax_kernels.split(",") if x.strip())
    cfg = SaneFlowConfig(
        vocab_size=len(tokenizer),
        d_embed=args.d_embed,
        d_model=args.d_model,
        n_layer=args.layers,
        n_heads=args.heads,
        d_ff=args.d_ff,
        conv_kernel=args.conv_kernel,
        syntax_mix_version=args.syntax_mix_version,
        syntax_kernels=syntax_kernels,
        state_mixer_version=args.state_mixer_version,
        state_clip=args.state_clip,
        state_zoneout=args.state_zoneout,
        attention_interval=args.attention_interval,
        attention_window=args.attention_window,
        thought_slots=args.thought_slots,
        thought_chunk=args.thought_chunk,
        thought_start_layer=args.thought_start_layer,
        landmark_interval=args.landmark_interval,
        landmark_chunk=args.landmark_chunk,
        landmark_max=args.landmark_max,
        tokenizer_path=args.tokenizer_path,
    )
    dataset_cls = SampleAlignedDataset if args.dataset_layout == "sample" else TokenStreamDataset
    print(json.dumps({
        "event": "loading_train_dataset",
        "paths": args.train_data,
        "dataset_layout": args.dataset_layout,
        "loss_mode": args.loss_mode,
        "seq_len": args.seq_len,
        "dataset_device": args.dataset_device,
    }), flush=True)
    train = dataset_cls(
        tokenizer=tokenizer,
        paths=[Path(p) for p in args.train_data],
        seq_len=args.seq_len,
        max_records=args.max_records,
        dataset_device=torch.device(args.device if args.dataset_device == "cuda" else "cpu"),
        loss_mode=args.loss_mode,
    )
    valid = None
    if args.valid_data:
        print(json.dumps({
            "event": "loading_valid_dataset",
            "paths": args.valid_data,
            "dataset_layout": args.dataset_layout,
            "loss_mode": args.loss_mode,
        }), flush=True)
        valid = dataset_cls(
            tokenizer=tokenizer,
            paths=[Path(p) for p in args.valid_data],
            seq_len=args.seq_len,
            max_records=min(args.max_records, 2000) if args.max_records else 2000,
            dataset_device=torch.device(args.device if args.dataset_device == "cuda" else "cpu"),
            loss_mode=args.loss_mode,
        )
    print(json.dumps({"event": "datasets_ready", "train_tokens": int(train.ids.numel())}), flush=True)
    model = SaneFlowLM(cfg).to(device=device, dtype=dtype)
    if args.init_from:
        payload = torch.load(Path(args.init_from), map_location=device, weights_only=True)
        old_cfg = SaneFlowConfig(**payload["config"])
        expected = cfg.to_dict()
        loaded = old_cfg.to_dict()
        mismatches = {
            k: (loaded.get(k), expected.get(k))
            for k in (
                "vocab_size",
                "d_embed",
                "d_model",
                "n_layer",
                "n_heads",
                "d_ff",
                "syntax_mix_version",
                "syntax_kernels",
                "state_mixer_version",
                "state_clip",
                "state_zoneout",
                "attention_interval",
                "attention_window",
                "thought_slots",
                "thought_chunk",
                "thought_start_layer",
                "landmark_interval",
                "landmark_chunk",
                "landmark_max",
            )
            if loaded.get(k) != expected.get(k)
        }
        if mismatches:
            only_vocab = set(mismatches) == {"vocab_size"}
            if not only_vocab:
                raise ValueError(f"--init-from config mismatch: {mismatches}")
            load_init_state_allow_vocab_resize(model, payload, int(old_cfg.vocab_size), int(cfg.vocab_size))
        else:
            model.load_state_dict(payload["model"])
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    if args.optimizer == "ademamix":
        opt = AdEMAMix(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "muon":
        muon_params = []
        adam_params = []
        for name, param in model.named_parameters():
            if param.ndim == 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(param)
            else:
                adam_params.append(param)
        opt = CombinedOptim(
            [
                Muon(muon_params, lr=args.muon_lr, momentum=args.muon_momentum, weight_decay=args.weight_decay) if muon_params else None,
                torch.optim.AdamW(adam_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=args.fused_adamw) if adam_params else None,
            ]
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            fused=args.fused_adamw,
        )
    log_path = out / "train_log.jsonl"

    def save(path: Path, step: int) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step}, tmp)
        tmp.replace(path)

    print(json.dumps({
        "run": str(out),
        "tokens": int(train.ids.numel()),
        "vocab_size": cfg.vocab_size,
        "config": cfg.to_dict(),
        "optimizer": args.optimizer,
        "muon_lr": args.muon_lr if args.optimizer == "muon" else None,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "loss_chunk_tokens": args.loss_chunk_tokens,
        "loss_mode": args.loss_mode,
        "activation_checkpointing": args.activation_checkpointing,
        "dataset_device": args.dataset_device,
        "dataset_layout": args.dataset_layout,
        "compile": args.compile,
        "tf32": args.tf32,
    }), flush=True)
    model.train()
    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = 0
    for step in range(1, args.steps + 1):
        if args.warmup_steps > 0 and step <= args.warmup_steps:
            lr_scale = step / args.warmup_steps
        else:
            progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
            lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        for group in opt.param_groups:
            if "base_lr" not in group:
                group["base_lr"] = group["lr"]
            group["lr"] = group["base_lr"] * lr_scale
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(args.grad_accum_steps):
            if args.loss_mode == "causal":
                x, y = train.batch(args.batch_size, device)
                mask = None
            else:
                x, y, mask = train.batch_with_mask(args.batch_size, device)
            hidden = model.forward_hidden(x, activation_checkpointing=args.activation_checkpointing)
            loss = lm_loss(model, hidden, y, cfg.vocab_size, args.loss_chunk_tokens, mask)
            loss_total += float(loss.item())
            (loss / args.grad_accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        loss_value = loss_total / args.grad_accum_steps

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            now = time.perf_counter()
            elapsed = max(now - last_log_time, 1e-9)
            step_delta = max(step - last_log_step, 1)
            trained_tokens = step_delta * args.grad_accum_steps * args.batch_size * args.seq_len
            row = {
                "step": step,
                "loss": round(loss_value, 4),
                "lr": round(float(opt.param_groups[0]["lr"]), 8),
                "tok_s": round(trained_tokens / elapsed, 1),
                "elapsed_s": round(now - train_start_time, 1),
            }
            last_log_time = now
            last_log_step = step
            if valid is not None and (step == 1 or step % (args.log_every * 10) == 0 or step == args.steps):
                row["valid_loss"] = round(eval_loss(model, valid, cfg, args.batch_size, device, args.loss_chunk_tokens, args.loss_mode), 4)
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            save(out / "latest.pt", step)
    save(out / "model.pt", args.steps)
    print(f"saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
