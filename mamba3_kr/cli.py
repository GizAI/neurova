from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MAMBA = REPO_ROOT / "mamba"
if LOCAL_MAMBA.exists() and str(LOCAL_MAMBA) not in sys.path:
    sys.path.insert(0, str(LOCAL_MAMBA))

from .data import iter_packed_token_batches, iter_texts
from .presets import MODEL_PRESETS
from .state import load_payload, load_state, save_state, state_summary
from .tokenizer import Tokenizer, build_tokenizer


class DeltaTextStreamer:
    def __init__(self, tokenizer: Tokenizer, skip_first: bool = True):
        self.tokenizer = tokenizer
        self.skip_first = skip_first
        self.pending = ""
        self.pending_ids: list[int] = []
        self.pending_tokens = 0
        self.generated_tokens = 0
        self.first_token_time: float | None = None
        self.last_token_time: float | None = None
        self.last_flush_time = time.time()

    def put(self, value: torch.Tensor) -> None:
        token_ids = value.detach().cpu().reshape(-1).tolist()
        if self.skip_first:
            self.skip_first = False
            return
        now = time.time()
        if self.first_token_time is None:
            self.first_token_time = now
        self.last_token_time = now
        self.generated_tokens += len(token_ids)
        self.pending_ids.extend(int(i) for i in token_ids)
        self.pending_tokens += len(token_ids)
        should_flush = (
            self.pending_tokens >= 16
            or now - self.last_flush_time >= 0.12
        )
        if should_flush:
            self.flush()

    def end(self) -> None:
        self.flush()

    def flush(self) -> None:
        if self.pending_ids:
            self.pending += self.tokenizer.decode(self.pending_ids)
            self.pending_ids = []
        if not self.pending:
            return
        print(self.pending, end="", flush=True)
        self.pending = ""
        self.pending_tokens = 0
        self.last_flush_time = time.time()

    @property
    def output_tokens_per_sec(self) -> float:
        if self.generated_tokens <= 1 or self.first_token_time is None or self.last_token_time is None:
            return 0.0
        elapsed = max(self.last_token_time - self.first_token_time, 1e-9)
        return max(0, self.generated_tokens - 1) / elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mamba3-KR experimental runtime")
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("model-info", "check-contract", "smoke", "probe-kernel", "train-tiny", "train-packed", "train-answer", "train-multitask", "eval-loss", "eval-answer-loss", "generate", "fast-generate", "eval-english", "quality-gate", "diagnose-decode", "decode-parity", "bench-decode", "serve", "state-prefill", "state-restore", "state-roundtrip", "verify-rlvr"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=list(MODEL_PRESETS), default="mimo-r4-tiny")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
        p.add_argument("--tokenizer", default="llama31", help="llama31, llama31-instruct, gpt-neox, byte, or hf:<name>")
        p.add_argument("--seq-len", type=int, default=256)
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--checkpoint", type=Path, default=Path("runs/mamba3_kr_tiny/model.pt"))
        p.add_argument("--state-out", type=Path, default=Path("runs/mamba3_kr_tiny/state.pt"))

    sub.choices["train-tiny"].add_argument("--data", nargs="+", type=Path, default=[Path("data/train_dialogues.txt")])
    sub.choices["train-tiny"].add_argument("--steps", type=int, default=20)
    sub.choices["train-tiny"].add_argument("--lr", type=float, default=2e-4)
    sub.choices["train-packed"].add_argument("--data", nargs="+", type=Path, default=[Path("data/english_bootstrap.txt")])
    sub.choices["train-packed"].add_argument("--steps", type=int, default=100)
    sub.choices["train-packed"].add_argument("--lr", type=float, default=2e-4)
    sub.choices["train-packed"].add_argument("--save-every", type=int, default=100)
    sub.choices["train-packed"].add_argument("--no-resume", action="store_true")
    sub.choices["train-packed"].add_argument("--no-save-optimizer", action="store_true")
    sub.choices["train-packed"].add_argument("--grad-accum-steps", type=int, default=1)
    sub.choices["train-packed"].add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    sub.choices["train-packed"].add_argument("--activation-checkpointing", action="store_true")
    sub.choices["train-packed"].add_argument("--deepspeed-config", type=Path, default=None)
    sub.choices["train-packed"].add_argument("--shuffle-texts", action="store_true")
    sub.choices["train-packed"].add_argument("--data-seed", type=int, default=0)
    sub.choices["train-packed"].add_argument("--max-text-chars", type=int, default=65536)
    sub.choices["train-packed"].add_argument("--max-text-tokens", type=int, default=120000)
    sub.choices["train-answer"].add_argument("--data", nargs="+", type=Path, default=[Path("data/mamba3_programmatic_curriculum.jsonl")])
    sub.choices["train-answer"].add_argument("--steps", type=int, default=100)
    sub.choices["train-answer"].add_argument("--lr", type=float, default=1e-4)
    sub.choices["train-answer"].add_argument("--save-every", type=int, default=100)
    sub.choices["train-answer"].add_argument("--no-resume", action="store_true")
    sub.choices["train-answer"].add_argument("--grad-accum-steps", type=int, default=1)
    sub.choices["train-answer"].add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    sub.choices["train-multitask"].add_argument("--base-data", nargs="+", type=Path, default=[Path("data/english_bootstrap.txt")])
    sub.choices["train-multitask"].add_argument("--answer-data", nargs="+", type=Path, default=[Path("data/mamba3_programmatic_curriculum.jsonl")])
    sub.choices["train-multitask"].add_argument("--steps", type=int, default=100)
    sub.choices["train-multitask"].add_argument("--lr", type=float, default=1e-4)
    sub.choices["train-multitask"].add_argument("--save-every", type=int, default=100)
    sub.choices["train-multitask"].add_argument("--no-resume", action="store_true")
    sub.choices["train-multitask"].add_argument("--base-accum-steps", type=int, default=3)
    sub.choices["train-multitask"].add_argument("--answer-accum-steps", type=int, default=1)
    sub.choices["train-multitask"].add_argument("--answer-loss-weight", type=float, default=1.0)
    sub.choices["train-multitask"].add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    sub.choices["probe-kernel"].add_argument("--data", nargs="+", type=Path, default=[Path("data/english_bootstrap.txt")])
    sub.choices["eval-loss"].add_argument("--data", nargs="+", type=Path, default=[Path("data/english_bootstrap.txt")])
    sub.choices["eval-loss"].add_argument("--batches", type=int, default=8)
    sub.choices["eval-answer-loss"].add_argument("--data", nargs="+", type=Path, default=[Path("data/mamba3_programmatic_curriculum.jsonl")])
    sub.choices["eval-answer-loss"].add_argument("--batches", type=int, default=8)

    sub.choices["generate"].add_argument("--prompt", default="안녕하세요. Mamba-3 한국어 모델 테스트입니다.")
    sub.choices["generate"].add_argument("--max-new", type=int, default=64)
    sub.choices["fast-generate"].add_argument("--prompt", default="The main idea is")
    sub.choices["fast-generate"].add_argument("--max-new", type=int, default=64)
    sub.choices["eval-english"].add_argument("--max-new", type=int, default=48)
    sub.choices["quality-gate"].add_argument("--max-new", type=int, default=48)
    sub.choices["quality-gate"].add_argument("--min-avg-new-tokens", type=float, default=8.0)
    sub.choices["quality-gate"].add_argument("--max-repeat-ratio", type=float, default=0.42)
    sub.choices["diagnose-decode"].add_argument("--prompt", default="The main idea is")
    sub.choices["decode-parity"].add_argument("--prompt", default="The main idea is")
    sub.choices["decode-parity"].add_argument("--max-new", type=int, default=24)
    sub.choices["bench-decode"].add_argument("--prompt", default="The main idea is")
    sub.choices["bench-decode"].add_argument("--max-new", type=int, default=64)
    sub.choices["bench-decode"].add_argument("--repeats", type=int, default=3)
    sub.choices["serve"].add_argument("--max-new", type=int, default=128)
    for name in ("fast-generate", "eval-english", "quality-gate", "decode-parity", "bench-decode", "serve"):
        sub.choices[name].add_argument("--top-k", type=int, default=40)
        sub.choices[name].add_argument("--top-p", type=float, default=0.9)
        sub.choices[name].add_argument("--temperature", type=float, default=0.8)
        sub.choices[name].add_argument("--repetition-penalty", type=float, default=1.15)
        sub.choices[name].add_argument("--safe-decode", action="store_true")
        sub.choices[name].add_argument("--cuda-graph", action="store_true")
        sub.choices[name].add_argument("--nan-check", action="store_true")
        sub.choices[name].add_argument("--exact-cache", action="store_true")

    sub.choices["state-prefill"].add_argument("--text", default="Project state checkpoint test for Mamba-3 recurrent memory.")
    sub.choices["state-restore"].add_argument("--state-in", type=Path, default=Path("runs/mamba3_kr_tiny/state.pt"))
    sub.choices["state-roundtrip"].add_argument("--text", default="Project state checkpoint test for Mamba-3 recurrent memory.")
    sub.choices["state-roundtrip"].add_argument("--state-in", type=Path, default=Path("runs/mamba3_kr_tiny/state_roundtrip.pt"))
    sub.choices["verify-rlvr"].add_argument("--rlvr-data", type=Path, default=Path("data/rlvr_verifier_bootstrap.jsonl"))
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def setup_perf(device: str) -> None:
    logging.getLogger("tilelang").setLevel(logging.ERROR)
    logging.getLogger("tilelang.jit").setLevel(logging.ERROR)
    logging.getLogger("tilelang.jit.kernel").setLevel(logging.ERROR)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def normalize_runtime_flags(args: argparse.Namespace) -> None:
    if getattr(args, "mode", "") == "transformer-tiny":
        args.safe_decode = True
        args.cuda_graph = False
    if getattr(args, "cuda_graph", False) and "moe" in getattr(args, "mode", ""):
        args.cuda_graph = False
    if getattr(args, "cuda_graph", False) and "recall" in getattr(args, "mode", ""):
        args.cuda_graph = False
    if getattr(args, "cuda_graph", False) and int(getattr(args, "seq_len", 0)) > 128:
        args.cuda_graph = False
    if getattr(args, "exact_cache", False):
        args.safe_decode = False
        args.cuda_graph = False


def load_or_new(args: argparse.Namespace):
    from .model import build_model, preset_config

    dtype = torch_dtype(args.dtype)
    tokenizer = build_tokenizer(args.tokenizer)
    cfg = preset_config(args.mode, vocab_size=tokenizer.vocab_size)
    if getattr(args, "activation_checkpointing", False):
        raise SystemExit(
            "activation checkpointing is disabled for Mamba-3 TileLang kernels. "
            "Use the official kernel-level backward recomputation path instead; "
            "block-level PyTorch checkpointing conflicts with Mamba3 autograd saved tensors."
        )
    model = build_model(cfg, device=args.device, dtype=dtype)
    if args.checkpoint.exists():
        payload = torch.load(args.checkpoint, map_location=args.device)
        if payload.get("mode") == args.mode and payload.get("tokenizer", "byte") == args.tokenizer:
            model.load_state_dict(payload["model"])
        else:
            print(
                f"skip checkpoint {args.checkpoint}: "
                f"mode={payload.get('mode')} tokenizer={payload.get('tokenizer', 'byte')} "
                f"!= {args.mode}/{args.tokenizer}",
                flush=True,
            )
    model.train(args.cmd in {"train-tiny", "train-packed", "train-answer", "train-multitask"})
    return model, tokenizer, dtype


def _model_config(model):
    base = model.module if hasattr(model, "module") else model
    return getattr(base, "config", None)


def language_model_loss(model, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    cfg = _model_config(model)
    n_meta_tokens = int(getattr(cfg, "n_meta_tokens", 0) or 0)
    if n_meta_tokens > 0 and logits.shape[1] == labels.shape[1] + n_meta_tokens:
        logits = logits[:, n_meta_tokens:]
    elif logits.shape[1] != labels.shape[1]:
        logits = logits[:, -labels.shape[1]:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def answer_loss(model, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    cfg = _model_config(model)
    n_meta_tokens = int(getattr(cfg, "n_meta_tokens", 0) or 0)
    if n_meta_tokens > 0 and logits.shape[1] == labels.shape[1] + n_meta_tokens:
        logits = logits[:, n_meta_tokens:]
    elif logits.shape[1] != labels.shape[1]:
        logits = logits[:, -labels.shape[1]:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)


def model_info(args: argparse.Namespace) -> None:
    from .model import effective_mlp_hidden_dim, estimate_parameters, preset_config

    tokenizer = build_tokenizer(args.tokenizer)
    cfg = preset_config(args.mode, vocab_size=tokenizer.vocab_size)
    print(json.dumps({
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "config": cfg.__dict__,
        "effective_mlp_hidden_dim": effective_mlp_hidden_dim(cfg),
        "estimated_parameters": estimate_parameters(cfg),
        "mamba3_required_features": None if cfg.architecture != "mamba3" else {
            "layer": "Mamba3",
            "mimo_rank": cfg.mimo_rank,
            "is_mimo": cfg.is_mimo,
            "all_layers_mamba3": True,
            "pre_norm": True,
            "swiglu_gated_mlp": cfg.d_intermediate > 0,
            "short_conv": "removed by official Mamba-3 layer",
            "bc_qk_norm_and_bias": "kept in official Mamba-3 implementation",
            "complex_state_tracking": "kept in official Mamba-3 implementation",
            "learnable_meta_tokens": cfg.n_meta_tokens,
        },
    }, ensure_ascii=False, indent=2))


def check_contract(args: argparse.Namespace) -> None:
    from .contract import check_architecture_contract
    from .model import preset_config

    tokenizer = build_tokenizer(args.tokenizer)
    cfg = preset_config(args.mode, vocab_size=tokenizer.vocab_size)
    errors = check_architecture_contract(cfg, args.tokenizer)
    print(json.dumps({
        "ok": not errors,
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "errors": errors,
    }, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


def batch_from_texts(
    tokenizer: Tokenizer,
    texts: list[str],
    seq_len: int,
    device: str,
    batch_size: int = 1,
) -> torch.Tensor:
    ids: list[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
    if len(ids) < seq_len + 1:
        repeats = (seq_len + 1) // max(1, len(ids)) + 1
        ids = ids * repeats
    needed = batch_size * (seq_len + 1)
    if len(ids) < needed:
        ids = ids * (needed // max(1, len(ids)) + 1)
    rows = []
    stride = max(1, seq_len // 2)
    for b in range(batch_size):
        start = b * stride
        rows.append(ids[start : start + seq_len + 1])
    return torch.tensor(rows, device=device, dtype=torch.long)


def _answer_example(tokenizer: Tokenizer, text: str, seq_len: int) -> tuple[list[int], list[int]] | None:
    marker = "\nAnswer:"
    if marker not in text:
        marker = " Answer:"
    if marker not in text:
        return None
    before, after = text.split(marker, 1)
    prompt = before.rstrip() + marker
    answer = after.strip()
    if not answer:
        return None
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
    ids = prompt_ids + answer_ids
    if len(ids) > seq_len + 1:
        keep_answer = min(len(answer_ids), seq_len)
        prompt_budget = max(1, seq_len + 1 - keep_answer)
        prompt_ids = prompt_ids[-prompt_budget:]
        ids = prompt_ids + answer_ids[: seq_len + 1 - len(prompt_ids)]
    if len(ids) < 2:
        return None
    input_ids = ids[:-1]
    labels = ids[1:]
    answer_start = max(0, len(prompt_ids) - 1)
    labels = [tok if idx >= answer_start else -100 for idx, tok in enumerate(labels)]
    if all(label == -100 for label in labels):
        return None
    return input_ids, labels


def iter_answer_texts(paths: list[Path]):
    for path in paths:
        if path.suffix == ".jsonl":
            yield from iter_texts([path])
            continue
        buffer: list[str] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                if line.startswith("Instruction:") and buffer:
                    yield "\n".join(buffer)
                    buffer = [line]
                else:
                    buffer.append(line)
        if buffer:
            yield "\n".join(buffer)


def iter_answer_batches(
    tokenizer: Tokenizer,
    paths: list[Path],
    seq_len: int,
    batch_size: int,
    device: str,
):
    from itertools import cycle
    import torch

    examples = []
    for text in iter_answer_texts(paths):
        item = _answer_example(tokenizer, text, seq_len)
        if item is not None:
            examples.append(item)
    if not examples:
        raise ValueError("no answer-supervised examples found")
    stream = cycle(examples)
    while True:
        rows = []
        label_rows = []
        for _ in range(batch_size):
            input_ids, labels = next(stream)
            pad = seq_len - len(input_ids)
            if pad > 0:
                input_ids = input_ids + [tokenizer.eos_id] * pad
                labels = labels + [-100] * pad
            rows.append(input_ids[:seq_len])
            label_rows.append(labels[:seq_len])
        yield (
            torch.tensor(rows, device=device, dtype=torch.long),
            torch.tensor(label_rows, device=device, dtype=torch.long),
        )


def smoke(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    x = batch_from_texts(
        tokenizer,
        ["안녕하세요. Mamba-3 SISO smoke test."],
        args.seq_len,
        args.device,
        args.batch_size,
    )
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.inference_mode():
        y = model(x, num_last_tokens=1).logits
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    elapsed = time.time() - t0
    tokens = int(x.numel())
    print(json.dumps({
        "ok": True,
        "mode": args.mode,
        "input_shape": list(x.shape),
        "logits_shape": list(y.shape),
        "elapsed_sec": round(elapsed, 4),
        "tokens_per_sec": round(tokens / elapsed, 2) if elapsed > 0 else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
        "device": args.device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, ensure_ascii=False, indent=2))


def _error_payload(exc: BaseException) -> dict:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc).splitlines()[-1] if str(exc) else repr(exc),
    }


def probe_kernel(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    from mamba_ssm.utils.generation import InferenceParams

    report: dict = {
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "device": args.device,
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and args.device.startswith("cuda") else None,
    }
    try:
        model, tokenizer, _ = load_or_new(args)
        batch = next(iter_packed_token_batches(tokenizer, args.data, args.seq_len, args.batch_size, args.device))
    except Exception as exc:
        report["load"] = _error_payload(exc)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    report["load"] = {"ok": True}

    try:
        model.eval()
        with torch.inference_mode():
            logits = model(batch[:, :-1], num_last_tokens=1).logits
        report["forward"] = {
            "ok": bool(torch.isfinite(logits).all().item()),
            "shape": list(logits.shape),
        }
    except Exception as exc:
        report["forward"] = _error_payload(exc)

    try:
        model.eval()
        ids = tokenizer.encode("The main idea is", add_eos=False)
        x = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
        with torch.inference_mode():
            inference_params = InferenceParams(max_seqlen=args.seq_len + 4, max_batch_size=1)
            first_logits = model(x, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
            first_id = int(torch.argmax(first_logits).item())
            inference_params.seqlen_offset += x.shape[1]
            step_logits = model(
                torch.tensor([[first_id]], device=args.device, dtype=torch.long),
                inference_params=inference_params,
                num_last_tokens=1,
            ).logits[0, -1, : tokenizer.vocab_size]
            full_logits = model(torch.tensor([ids[-args.seq_len:] + [first_id]], device=args.device), num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
        report["decode_step"] = {
            "ok": bool(torch.isfinite(step_logits).all().item()),
            "argmax_matches_full_forward": int(torch.argmax(step_logits).item()) == int(torch.argmax(full_logits).item()),
            "cache_argmax": tokenizer.decode([int(torch.argmax(step_logits).item())]),
            "full_argmax": tokenizer.decode([int(torch.argmax(full_logits).item())]),
        }
    except Exception as exc:
        report["decode_step"] = _error_payload(exc)

    try:
        model.train()
        logits = model(batch[:, :-1]).logits
        loss = language_model_loss(model, logits, batch[:, 1:])
        loss.backward()
        report["backward"] = {
            "ok": True,
            "loss": float(loss.detach().cpu()),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
        }
    except Exception as exc:
        report["backward"] = _error_payload(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train_tiny(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    texts = list(iter_texts(args.data))
    if not texts:
        raise SystemExit("no training text found")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    for step in range(1, args.steps + 1):
        text = texts[(step - 1) % len(texts)]
        batch = batch_from_texts(tokenizer, [text], args.seq_len, args.device, args.batch_size)
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        if args.device.startswith("cuda") and step == 1:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        logits = model(batch[:, :-1]).logits
        loss = language_model_loss(model, logits, batch[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        elapsed = time.time() - t0
        tokens = int(batch[:, :-1].numel())
        losses.append(float(loss.detach().cpu()))
        peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device.startswith("cuda") else 0.0
        print(
            f"step={step} loss={losses[-1]:.4f} "
            f"tok_s={tokens / elapsed:.1f} peak_vram_gb={peak:.3f}",
            flush=True,
        )
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "mode": args.mode, "tokenizer": args.tokenizer, "losses": losses},
        args.checkpoint,
    )
    print(json.dumps({"checkpoint": str(args.checkpoint), "final_loss": losses[-1]}, ensure_ascii=False, indent=2))


def train_packed(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    engine = None
    if args.deepspeed_config is not None:
        try:
            import deepspeed
        except Exception as exc:
            raise SystemExit(f"DeepSpeed requested but unavailable: {exc}") from exc
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        engine, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=str(args.deepspeed_config),
        )
        model = engine
        opt = None
    elif args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise SystemExit(f"adamw8bit requested but bitsandbytes is unavailable: {exc}") from exc
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    losses = []
    if args.checkpoint.exists() and not args.no_resume:
        payload = torch.load(args.checkpoint, map_location=args.device)
        if payload.get("mode") == args.mode and payload.get("tokenizer") == args.tokenizer:
            if opt is not None and payload.get("optimizer") is not None:
                opt.load_state_dict(payload["optimizer"])
            start_step = int(payload.get("step", 0))
            losses = list(payload.get("losses", []))
            print(f"resume checkpoint={args.checkpoint} step={start_step}", flush=True)
    batches = iter_packed_token_batches(
        tokenizer,
        args.data,
        args.seq_len,
        args.batch_size,
        args.device,
        shuffle_texts=getattr(args, "shuffle_texts", False),
        seed=getattr(args, "data_seed", 0),
        max_text_chars=getattr(args, "max_text_chars", 65536),
        max_text_tokens=getattr(args, "max_text_tokens", 120000),
    )
    accum = max(1, int(args.grad_accum_steps))
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        if args.device.startswith("cuda") and local_step == 1:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if engine is None:
            opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        tokens = 0
        for _ in range(accum):
            batch = next(batches)
            logits = model(batch[:, :-1]).logits
            loss = language_model_loss(model, logits, batch[:, 1:])
            if engine is not None:
                model.backward(loss / accum)
            else:
                (loss / accum).backward()
            loss_value += float(loss.detach().cpu())
            tokens += int(batch[:, :-1].numel())
        if engine is not None:
            model.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        elapsed = time.time() - t0
        losses.append(loss_value / accum)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device.startswith("cuda") else 0.0
        print(
            f"step={step} loss={losses[-1]:.4f} "
            f"tok_s={tokens / elapsed:.1f} peak_vram_gb={peak:.3f} "
            f"grad_accum={accum} optimizer={args.optimizer} "
            f"deepspeed={args.deepspeed_config is not None}",
            flush=True,
        )
        if args.save_every > 0 and (local_step % args.save_every == 0 or local_step == args.steps):
            _save_training_checkpoint(args, model, opt, step, losses)
    _save_training_checkpoint(args, model, opt, start_step + args.steps, losses)
    print(json.dumps({"checkpoint": str(args.checkpoint), "step": start_step + args.steps, "final_loss": losses[-1]}, ensure_ascii=False, indent=2))


def _make_optimizer(args: argparse.Namespace, model):
    if args.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise SystemExit(f"adamw8bit requested but bitsandbytes is unavailable: {exc}") from exc
        return bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)
    return torch.optim.AdamW(model.parameters(), lr=args.lr)


def train_answer(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    opt = _make_optimizer(args, model)
    start_step = 0
    losses = []
    if args.checkpoint.exists() and not args.no_resume:
        payload = torch.load(args.checkpoint, map_location=args.device)
        if payload.get("mode") == args.mode and payload.get("tokenizer") == args.tokenizer:
            if payload.get("optimizer") is not None:
                opt.load_state_dict(payload["optimizer"])
            start_step = int(payload.get("step", 0))
            losses = list(payload.get("losses", []))
            print(f"resume checkpoint={args.checkpoint} step={start_step}", flush=True)
    batches = iter_answer_batches(tokenizer, args.data, args.seq_len, args.batch_size, args.device)
    accum = max(1, int(args.grad_accum_steps))
    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        if args.device.startswith("cuda") and local_step == 1:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        loss_value = 0.0
        tokens = 0
        answer_tokens = 0
        for _ in range(accum):
            batch, labels = next(batches)
            logits = model(batch).logits
            loss = answer_loss(model, logits, labels)
            (loss / accum).backward()
            loss_value += float(loss.detach().cpu())
            tokens += int(batch.numel())
            answer_tokens += int((labels != -100).sum().item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        elapsed = time.time() - t0
        losses.append(loss_value / accum)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device.startswith("cuda") else 0.0
        print(
            f"step={step} answer_loss={losses[-1]:.4f} "
            f"tok_s={tokens / elapsed:.1f} answer_tok={answer_tokens} "
            f"peak_vram_gb={peak:.3f} grad_accum={accum} optimizer={args.optimizer}",
            flush=True,
        )
        if args.save_every > 0 and (local_step % args.save_every == 0 or local_step == args.steps):
            _save_training_checkpoint(args, model, opt, step, losses)
    _save_training_checkpoint(args, model, opt, start_step + args.steps, losses)
    print(json.dumps({"checkpoint": str(args.checkpoint), "step": start_step + args.steps, "final_loss": losses[-1]}, ensure_ascii=False, indent=2))


def train_multitask(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    opt = _make_optimizer(args, model)
    start_step = 0
    losses = []
    if args.checkpoint.exists() and not args.no_resume:
        payload = torch.load(args.checkpoint, map_location=args.device)
        if payload.get("mode") == args.mode and payload.get("tokenizer") == args.tokenizer:
            if payload.get("optimizer") is not None:
                opt.load_state_dict(payload["optimizer"])
            start_step = int(payload.get("step", 0))
            losses = list(payload.get("losses", []))
            print(f"resume checkpoint={args.checkpoint} step={start_step}", flush=True)

    base_batches = iter_packed_token_batches(tokenizer, args.base_data, args.seq_len, args.batch_size, args.device)
    answer_batches = iter_answer_batches(tokenizer, args.answer_data, args.seq_len, args.batch_size, args.device)
    base_accum = max(0, int(args.base_accum_steps))
    answer_accum = max(0, int(args.answer_accum_steps))
    total_accum = base_accum + answer_accum
    if total_accum <= 0:
        raise SystemExit("train-multitask requires at least one base or answer accumulation step")

    for local_step in range(1, args.steps + 1):
        step = start_step + local_step
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        if args.device.startswith("cuda") and local_step == 1:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        base_loss_value = 0.0
        answer_loss_value = 0.0
        tokens = 0
        answer_tokens = 0

        for _ in range(base_accum):
            batch = next(base_batches)
            logits = model(batch[:, :-1]).logits
            loss = language_model_loss(model, logits, batch[:, 1:])
            (loss / total_accum).backward()
            base_loss_value += float(loss.detach().cpu())
            tokens += int(batch[:, :-1].numel())

        for _ in range(answer_accum):
            batch, labels = next(answer_batches)
            logits = model(batch).logits
            loss = answer_loss(model, logits, labels) * float(args.answer_loss_weight)
            (loss / total_accum).backward()
            answer_loss_value += float(loss.detach().cpu())
            tokens += int(batch.numel())
            answer_tokens += int((labels != -100).sum().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        torch.cuda.synchronize() if args.device.startswith("cuda") else None
        elapsed = time.time() - t0
        avg_base_loss = base_loss_value / max(1, base_accum)
        avg_answer_loss = answer_loss_value / max(1, answer_accum)
        total_loss = (base_loss_value + answer_loss_value) / total_accum
        losses.append(total_loss)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if args.device.startswith("cuda") else 0.0
        print(
            f"step={step} loss={total_loss:.4f} base_loss={avg_base_loss:.4f} "
            f"answer_loss={avg_answer_loss:.4f} tok_s={tokens / elapsed:.1f} "
            f"answer_tok={answer_tokens} peak_vram_gb={peak:.3f} "
            f"base_accum={base_accum} answer_accum={answer_accum} "
            f"answer_weight={args.answer_loss_weight} optimizer={args.optimizer}",
            flush=True,
        )
        if args.save_every > 0 and (local_step % args.save_every == 0 or local_step == args.steps):
            _save_training_checkpoint(args, model, opt, step, losses)
    _save_training_checkpoint(args, model, opt, start_step + args.steps, losses)
    print(json.dumps({"checkpoint": str(args.checkpoint), "step": start_step + args.steps, "final_loss": losses[-1]}, ensure_ascii=False, indent=2))


def _save_training_checkpoint(args: argparse.Namespace, model, opt, step: int, losses: list[float]) -> None:
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model_for_state = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": model_for_state.state_dict(),
            "optimizer": None if getattr(args, "no_save_optimizer", False) else (opt.state_dict() if opt is not None else None),
            "mode": args.mode,
            "tokenizer": args.tokenizer,
            "step": step,
            "losses": losses[-2000:],
        },
        args.checkpoint,
    )


@torch.inference_mode()
def eval_loss(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    model.eval()
    batches = iter_packed_token_batches(tokenizer, args.data, args.seq_len, args.batch_size, args.device)
    losses = []
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.batches):
        batch = next(batches)
        logits = model(batch[:, :-1]).logits
        loss = language_model_loss(model, logits, batch[:, 1:])
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    elapsed = time.time() - t0
    print(json.dumps({
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "batches": args.batches,
        "loss": sum(losses) / len(losses),
        "elapsed_sec": round(elapsed, 4),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
    }, ensure_ascii=False, indent=2))


@torch.inference_mode()
def eval_answer_loss(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    model.eval()
    batches = iter_answer_batches(tokenizer, args.data, args.seq_len, args.batch_size, args.device)
    losses = []
    answer_tokens = 0
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.batches):
        batch, labels = next(batches)
        logits = model(batch).logits
        loss = answer_loss(model, logits, labels)
        losses.append(float(loss.detach().cpu()))
        answer_tokens += int((labels != -100).sum().item())
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    elapsed = time.time() - t0
    print(json.dumps({
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "batches": args.batches,
        "answer_tokens": answer_tokens,
        "loss": sum(losses) / len(losses),
        "elapsed_sec": round(elapsed, 4),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
    }, ensure_ascii=False, indent=2))


@torch.inference_mode()
def generate(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    model, tokenizer, _ = load_or_new(args)
    ids = tokenizer.encode(args.prompt, add_eos=False)
    for _ in range(args.max_new):
        seq = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
        logits = model(seq, num_last_tokens=1).logits[:, -1, :]
        next_id = int(torch.argmax(logits, dim=-1).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    print(tokenizer.decode(ids))


@torch.inference_mode()
def _fast_decode_once(model, tokenizer: Tokenizer, args: argparse.Namespace, prompt: str) -> dict:
    from mamba_ssm.utils.generation import decode

    ids = tokenizer.encode(prompt, add_eos=False)
    input_ids = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
    if not getattr(args, "safe_decode", False) and not getattr(args, "exact_cache", False):
        _require_official_recurrent_shape(model)
    if getattr(args, "cache_verify", False):
        return _verified_cache_decode_once(model, tokenizer, args, input_ids)
    if getattr(args, "exact_cache", False):
        return _exact_cache_decode_once(model, tokenizer, args, input_ids)
    if args.safe_decode:
        return _full_forward_decode_once(model, tokenizer, args, input_ids, reason="requested_safe_decode")
    model._decode_suppress_token_ids = []
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = decode(
        input_ids,
        model,
        max_length=input_ids.shape[1] + args.max_new,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_id,
        vocab_size=tokenizer.vocab_size,
        output_scores=args.nan_check,
        cg=args.cuda_graph,
    )
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    elapsed = time.time() - t0
    if args.nan_check and any(torch.isnan(score).any().item() for score in out.scores):
        return _full_forward_decode_once(
            model,
            tokenizer,
            args,
            input_ids,
            reason="nan_cache_step_logits",
            failed_decode_elapsed=elapsed,
        )
    seq = out.sequences[0].detach().cpu().tolist()
    new_tokens = max(0, len(seq) - input_ids.shape[1])
    return {
        "text": tokenizer.decode(seq),
        "decode": "mamba3_step_fn",
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "cuda_graph": args.cuda_graph,
        "new_tokens": new_tokens,
        "elapsed_sec": round(elapsed, 4),
        "new_tokens_per_sec": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
    }


def _require_official_recurrent_shape(model) -> None:
    backbone = getattr(model, "backbone", None)
    layers = getattr(backbone, "layers", [])
    for idx, layer in enumerate(layers):
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "Mamba3":
            continue
        if bool(getattr(mixer, "is_mimo", False)):
            headdim = int(getattr(mixer, "headdim", 0))
            d_state = int(getattr(mixer, "d_state", 0))
            chunk_size = int(getattr(mixer, "chunk_size", 0))
            mimo_rank = int(getattr(mixer, "mimo_rank", 0))
            if (headdim, d_state, chunk_size, mimo_rank) != (64, 128, 16, 4):
                raise RuntimeError(
                    "official Mamba-3 MIMO recurrent decode requires "
                    f"headdim=64,d_state=128,chunk_size=16,mimo_rank=4; "
                    f"layer {idx} has headdim={headdim},d_state={d_state},"
                    f"chunk_size={chunk_size},mimo_rank={mimo_rank}. "
                    "Use --safe-decode for this checkpoint or train an official-shape preset."
                )


@torch.inference_mode()
def _verified_cache_decode_once(model, tokenizer: Tokenizer, args: argparse.Namespace, input_ids: torch.Tensor) -> dict:
    from mamba_ssm.utils.generation import InferenceParams

    ids = input_ids[0].detach().cpu().tolist()
    start_len = len(ids)
    inference_params = InferenceParams(max_seqlen=args.seq_len + args.max_new + 4, max_batch_size=1)
    corrections = 0
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    cache_logits = model(input_ids, inference_params=inference_params, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]
    full_logits = model(input_ids, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]
    inference_params.seqlen_offset += input_ids.shape[1]

    for _ in range(args.max_new):
        cache_id = int(torch.argmax(cache_logits, dim=-1).item())
        full_id = int(torch.argmax(full_logits, dim=-1).item())
        next_id = full_id
        if cache_id != full_id:
            corrections += 1
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        if getattr(args, "stop_after_sentence", False):
            generated = tokenizer.decode(ids[start_len:])
            min_chars = int(getattr(args, "min_sentence_chars", 18))
            if len(generated.strip()) >= min_chars and generated.rstrip().endswith((".", "!", "?")):
                break
        step_ids = torch.tensor([[next_id]], device=args.device, dtype=torch.long)
        cache_logits = model(step_ids, inference_params=inference_params, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]
        inference_params.seqlen_offset += 1
        full_ids = torch.tensor([ids[-args.seq_len:]], device=args.device, dtype=torch.long)
        full_logits = model(full_ids, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    new_tokens = max(0, len(ids) - start_len)
    return {
        "text": tokenizer.decode(ids),
        "decode": "official_cache_verified_by_full_forward",
        "cache_corrections": corrections,
        "top_k": 1,
        "top_p": 0.0,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "cuda_graph": False,
        "new_tokens": new_tokens,
        "elapsed_sec": round(elapsed, 4),
        "new_tokens_per_sec": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
    }


@torch.inference_mode()
def _exact_cache_decode_once(model, tokenizer: Tokenizer, args: argparse.Namespace, input_ids: torch.Tensor) -> dict:
    ids = input_ids[0].detach().cpu().tolist()
    start_len = len(ids)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for _ in range(args.max_new):
        full_ids = torch.tensor([ids[-args.seq_len:]], device=args.device, dtype=torch.long)
        logits = model(full_ids, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size]
        next_id = int(torch.argmax(logits, dim=-1).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
        if getattr(args, "stop_after_sentence", False):
            generated = tokenizer.decode(ids[start_len:])
            min_chars = int(getattr(args, "min_sentence_chars", 18))
            if len(generated.strip()) >= min_chars and generated.rstrip().endswith((".", "!", "?")):
                break

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = max(time.time() - t0, 1e-9)
    new_tokens = max(0, len(ids) - start_len)
    return {
        "text": tokenizer.decode(ids),
        "decode": "exact_full_forward_logits",
        "top_k": 1,
        "top_p": 0.0,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "cuda_graph": False,
        "new_tokens": new_tokens,
        "elapsed_sec": round(elapsed, 4),
        "new_tokens_per_sec": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
    }


def _chat_prompt(user_text: str) -> str:
    return f"Instruction: {user_text.strip()} Answer:"


ENGLISH_GATE_PROMPTS = [
    {
        "prompt": "The main idea is",
        "required": ["goal", "test", "improve"],
    },
    {
        "prompt": "A good teacher should",
        "required": ["clear", "important"],
    },
    {
        "prompt": "In simple words, science is",
        "required": ["learn", "reality"],
    },
    {
        "prompt": "Write one clear sentence about courage:",
        "required": ["courage", "difficult"],
    },
]

ENGLISH_GATE_BANNED_PATTERNS = [
    "@xmath",
    "@xcite",
    "temperature_c",
    "humidity_percent",
    "\"city\"",
    "```json",
    "```yaml",
    "the most common, and the most common",
    "the following lemma",
]


def _token_repeat_ratio(text: str) -> float:
    words = [part.strip(".,;:!?()[]{}\"'").lower() for part in text.split()]
    words = [word for word in words if word]
    if len(words) <= 1:
        return 0.0
    repeats = sum(1 for left, right in zip(words, words[1:]) if left == right)
    return repeats / max(1, len(words) - 1)


def _repeated_ngram_count(text: str, n: int = 3) -> int:
    words = [part.strip(".,;:!?()[]{}\"'").lower() for part in text.split()]
    words = [word for word in words if word]
    if len(words) < n * 2:
        return 0
    seen: set[tuple[str, ...]] = set()
    repeated = 0
    for idx in range(0, len(words) - n + 1):
        key = tuple(words[idx : idx + n])
        if key in seen:
            repeated += 1
        else:
            seen.add(key)
    return repeated


def _quality_prompt_report(model, tokenizer: Tokenizer, args: argparse.Namespace, item: dict) -> dict:
    result = _fast_decode_once(model, tokenizer, args, item["prompt"])
    text = result.pop("text")
    lowered = text.lower()
    missing = [word for word in item["required"] if word.lower() not in lowered]
    repeat_ratio = _token_repeat_ratio(text)
    banned = [pattern for pattern in ENGLISH_GATE_BANNED_PATTERNS if pattern in lowered]
    repeated_ngrams = _repeated_ngram_count(text, 3)
    ok = (
        not missing
        and not banned
        and result["new_tokens"] >= 4
        and repeat_ratio <= args.max_repeat_ratio
        and repeated_ngrams == 0
    )
    return {
        "prompt": item["prompt"],
        "output": text,
        "missing_required_terms": missing,
        "banned_patterns": banned,
        "repeat_ratio": round(repeat_ratio, 4),
        "repeated_3grams": repeated_ngrams,
        "new_tokens": result["new_tokens"],
        "new_tokens_per_sec": result["new_tokens_per_sec"],
        "ok": ok,
        "metrics": result,
    }


@torch.inference_mode()
def _stream_chat_once(model, tokenizer: Tokenizer, args: argparse.Namespace, user_text: str) -> None:
    from mamba_ssm.utils.generation import decode

    prompt = _chat_prompt(user_text)
    ids = tokenizer.encode(prompt, add_eos=False)
    input_ids = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
    streamer = DeltaTextStreamer(tokenizer)
    _ = decode(
        input_ids,
        model,
        max_length=input_ids.shape[1] + args.max_new,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_id,
        vocab_size=tokenizer.vocab_size,
        output_scores=False,
        streamer=streamer,
        cg=args.cuda_graph,
    )
    print(f" ({streamer.output_tokens_per_sec:.1f} tok/s)", flush=True)


@torch.inference_mode()
def _full_forward_decode_once(
    model,
    tokenizer: Tokenizer,
    args: argparse.Namespace,
    input_ids: torch.Tensor,
    reason: str,
    failed_decode_elapsed: float | None = None,
) -> dict:
    ids = input_ids[0].detach().cpu().tolist()
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(args.max_new):
        seq = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
        logits = model(seq, num_last_tokens=1).logits[:, -1, : tokenizer.vocab_size].float()
        if torch.isnan(logits).any().item():
            break
        next_id = int(torch.argmax(logits, dim=-1).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    torch.cuda.synchronize() if args.device.startswith("cuda") else None
    elapsed = time.time() - t0
    new_tokens = max(0, len(ids) - input_ids.shape[1])
    return {
        "text": tokenizer.decode(ids),
        "decode": "full_forward_fallback",
        "fallback_reason": reason,
        "failed_decode_elapsed_sec": round(failed_decode_elapsed, 4) if failed_decode_elapsed is not None else None,
        "top_k": 1,
        "top_p": 0.0,
        "temperature": 1.0,
        "repetition_penalty": 1.0,
        "new_tokens": new_tokens,
        "elapsed_sec": round(elapsed, 4),
        "new_tokens_per_sec": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None,
    }


def _top_tokens(tokenizer: Tokenizer, logits: torch.Tensor, k: int = 10) -> list[dict]:
    vals, idx = torch.topk(logits.float(), k)
    return [
        {"logit": round(float(v), 4), "id": int(i), "text": tokenizer.decode([int(i)])}
        for v, i in zip(vals.detach().cpu(), idx.detach().cpu())
    ]


@torch.inference_mode()
def diagnose_decode(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    from mamba_ssm.utils.generation import InferenceParams

    model, tokenizer, _ = load_or_new(args)
    ids = tokenizer.encode(args.prompt, add_eos=False)
    input_ids = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
    inference_params = InferenceParams(max_seqlen=args.seq_len + 4, max_batch_size=1)
    first_logits = model(input_ids, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    first_id = int(torch.argmax(first_logits).item())
    inference_params.seqlen_offset += input_ids.shape[1]
    step_logits = model(
        torch.tensor([[first_id]], device=args.device, dtype=torch.long),
        inference_params=inference_params,
        num_last_tokens=1,
    ).logits[0, -1, : tokenizer.vocab_size]
    full_ids = torch.tensor([ids[-args.seq_len:] + [first_id]], device=args.device, dtype=torch.long)
    full_logits = model(full_ids, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    report = {
        "prompt": args.prompt,
        "first_token": {"id": first_id, "text": tokenizer.decode([first_id])},
        "cache_step_finite": bool(torch.isfinite(step_logits).all().item()),
        "full_forward_finite": bool(torch.isfinite(full_logits).all().item()),
        "cache_step_argmax": {
            "id": int(torch.argmax(step_logits).item()),
            "text": tokenizer.decode([int(torch.argmax(step_logits).item())]),
        },
        "full_forward_argmax": {
            "id": int(torch.argmax(full_logits).item()),
            "text": tokenizer.decode([int(torch.argmax(full_logits).item())]),
        },
        "cache_step_top": _top_tokens(tokenizer, step_logits),
        "full_forward_top": _top_tokens(tokenizer, full_logits),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


@torch.inference_mode()
def decode_parity(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(args)
    eager_args = argparse.Namespace(**vars(args))
    graph_args = argparse.Namespace(**vars(args))
    eager_args.cuda_graph = False
    graph_args.cuda_graph = False if "moe" in args.mode else True
    eager = _fast_decode_once(model, tokenizer, eager_args, args.prompt)
    graph = _fast_decode_once(model, tokenizer, graph_args, args.prompt)
    payload = {
        "ok": eager["text"] == graph["text"],
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "eager_cuda_graph": eager["cuda_graph"],
        "graph_cuda_graph": graph["cuda_graph"],
        "eager_text": eager["text"],
        "graph_text": graph["text"],
        "eager_metrics": {k: v for k, v in eager.items() if k != "text"},
        "graph_metrics": {k: v for k, v in graph.items() if k != "text"},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


@torch.inference_mode()
def fast_generate(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(args)
    result = _fast_decode_once(model, tokenizer, args, args.prompt)
    print(result.pop("text"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


@torch.inference_mode()
def eval_english(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(args)
    prompts = [
        "The main idea is",
        "A good teacher should",
        "In simple words, science is",
        "Write one clear sentence about courage:",
    ]
    _ = _fast_decode_once(model, tokenizer, args, "Warm up.")
    for prompt in prompts:
        result = _fast_decode_once(model, tokenizer, args, prompt)
        text = result.pop("text")
        print("=== INPUT ===")
        print(prompt)
        print("=== OUTPUT ===")
        print(text)
        print("=== METRICS ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))


@torch.inference_mode()
def quality_gate(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(args)
    _ = _fast_decode_once(model, tokenizer, args, "Warm up.")
    reports = [_quality_prompt_report(model, tokenizer, args, item) for item in ENGLISH_GATE_PROMPTS]
    avg_new_tokens = sum(item["new_tokens"] for item in reports) / max(1, len(reports))
    avg_tok_s_values = [
        item["new_tokens_per_sec"]
        for item in reports
        if item["new_tokens_per_sec"] is not None
    ]
    avg_tok_s = sum(avg_tok_s_values) / max(1, len(avg_tok_s_values))
    ok = all(item["ok"] for item in reports) and avg_new_tokens >= args.min_avg_new_tokens
    payload = {
        "ok": ok,
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "avg_new_tokens": round(avg_new_tokens, 2),
        "avg_new_tokens_per_sec": round(avg_tok_s, 2),
        "min_avg_new_tokens": args.min_avg_new_tokens,
        "max_repeat_ratio": args.max_repeat_ratio,
        "reports": reports,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not ok:
        raise SystemExit(1)


@torch.inference_mode()
def bench_decode(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    normalize_runtime_flags(args)

    model, tokenizer, _ = load_or_new(args)
    ids = tokenizer.encode(args.prompt, add_eos=False)
    input_ids = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
    runs = []
    last_text = tokenizer.decode(ids)
    if not args.safe_decode:
        from mamba_ssm.utils.generation import decode
    for idx in range(args.repeats + 1):
        if args.safe_decode:
            result = _full_forward_decode_once(
                model,
                tokenizer,
                args,
                input_ids,
                reason="bench_safe_decode",
            )
            last_text = result["text"]
            new_tokens = result["new_tokens"]
            elapsed = float(result["elapsed_sec"])
            tok_s = result["new_tokens_per_sec"]
            peak_vram_gb = result["peak_vram_gb"]
        else:
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            out = decode(
                input_ids,
                model,
                max_length=input_ids.shape[1] + args.max_new,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_id,
                vocab_size=tokenizer.vocab_size,
                output_scores=False,
                cg=args.cuda_graph,
            )
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            seq = out.sequences[0].detach().cpu().tolist()
            last_text = tokenizer.decode(seq)
            new_tokens = max(0, len(seq) - input_ids.shape[1])
            tok_s = round(new_tokens / elapsed, 2) if elapsed > 0 else None
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if args.device.startswith("cuda") else None
        runs.append({
            "kind": "warmup" if idx == 0 else "measure",
            "new_tokens": new_tokens,
            "elapsed_sec": round(elapsed, 4),
            "new_tokens_per_sec": tok_s,
            "peak_vram_gb": peak_vram_gb,
        })
    measured = runs[1:]
    avg_tok_s = sum(run["new_tokens_per_sec"] for run in measured if run["new_tokens_per_sec"] is not None) / max(1, len(measured))
    print(last_text)
    print(json.dumps({
        "decode": "full_forward_fallback" if args.safe_decode else "mamba3_step_fn",
        "safe_decode": args.safe_decode,
        "cuda_graph": args.cuda_graph,
        "mode": args.mode,
        "dtype": args.dtype,
        "runs": runs,
        "avg_measured_new_tokens_per_sec": round(avg_tok_s, 2),
    }, ensure_ascii=False, indent=2))


@torch.inference_mode()
def serve(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    normalize_runtime_flags(args)
    model, tokenizer, _ = load_or_new(args)
    args.cuda_graph = False if "moe" in args.mode or args.seq_len > 128 else True
    _ = _fast_decode_once(model, tokenizer, args, _chat_prompt("Explain why practice matters."))
    print(f"Neurova Mamba-3 ready ({args.mode}, {args.dtype}). Type /q to quit.", flush=True)
    while True:
        try:
            prompt = input("you> ")
        except EOFError:
            break
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in {"/q", "/quit", "quit", "exit"}:
            break
        print("neurova> ", end="", flush=True)
        _stream_chat_once(model, tokenizer, args, prompt)


@torch.inference_mode()
def state_prefill(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    from mamba_ssm.utils.generation import InferenceParams

    model, tokenizer, dtype = load_or_new(args)
    ids = tokenizer.encode(args.text, add_eos=False)
    seq = torch.tensor(ids[-args.seq_len:], device=args.device, dtype=torch.long).unsqueeze(0)
    inference_params = InferenceParams(max_seqlen=args.seq_len, max_batch_size=1)
    _ = model(seq, inference_params=inference_params, num_last_tokens=1)
    metadata = {
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "state_offset": int(seq.shape[1]),
        "text_chars": len(args.text),
    }
    save_state(inference_params.key_value_memory_dict, args.state_out, metadata=metadata)
    print(json.dumps({
        "state": str(args.state_out),
        "summary": state_summary(inference_params.key_value_memory_dict),
        "metadata": metadata,
        "dtype": str(dtype),
    }, ensure_ascii=False, indent=2))


def state_restore(args: argparse.Namespace) -> None:
    dtype = torch_dtype(args.dtype)
    payload = load_payload(args.state_in)
    cache = load_state(args.state_in, args.device, dtype)
    print(json.dumps({
        "state": str(args.state_in),
        "loaded": True,
        "summary": state_summary(cache),
        "metadata": payload.get("metadata", {}),
        "dtype": str(dtype),
        "device": args.device,
    }, ensure_ascii=False, indent=2))


@torch.inference_mode()
def state_roundtrip(args: argparse.Namespace) -> None:
    setup_perf(args.device)
    from mamba_ssm.utils.generation import InferenceParams

    model, tokenizer, dtype = load_or_new(args)
    ids = tokenizer.encode(args.text, add_eos=False)
    ids = ids[-args.seq_len:]
    seq = torch.tensor(ids, device=args.device, dtype=torch.long).unsqueeze(0)
    max_seqlen = args.seq_len + 4
    inference_params = InferenceParams(max_seqlen=max_seqlen, max_batch_size=1)
    prefix_logits = model(seq, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    first_id = int(torch.argmax(prefix_logits).item())
    metadata = {
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "state_offset": int(seq.shape[1]),
        "text_chars": len(args.text),
        "dtype": args.dtype,
    }
    t0 = time.time()
    save_state(inference_params.key_value_memory_dict, args.state_in, metadata=metadata)
    save_elapsed = time.time() - t0
    t1 = time.time()
    payload = load_payload(args.state_in)
    restored_cache = load_state(args.state_in, args.device, dtype)
    load_elapsed = time.time() - t1
    restored_params = InferenceParams(
        max_seqlen=max_seqlen,
        max_batch_size=1,
        seqlen_offset=int(payload.get("metadata", {}).get("state_offset", seq.shape[1])),
        key_value_memory_dict=restored_cache,
    )
    step_token = torch.tensor([[first_id]], device=args.device, dtype=torch.long)
    restored_logits = model(step_token, inference_params=restored_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    full_ids = torch.tensor([ids + [first_id]], device=args.device, dtype=torch.long)
    full_logits = model(full_ids, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    restored_argmax = int(torch.argmax(restored_logits).item())
    full_argmax = int(torch.argmax(full_logits).item())
    summary = state_summary(restored_cache)
    payload_out = {
        "ok": (
            bool(torch.isfinite(restored_logits).all().item())
            and bool(torch.isfinite(full_logits).all().item())
            and restored_argmax == full_argmax
            and payload.get("metadata", {}).get("mode") == args.mode
            and payload.get("metadata", {}).get("tokenizer") == args.tokenizer
        ),
        "mode": args.mode,
        "tokenizer": args.tokenizer,
        "checkpoint": str(args.checkpoint),
        "state": str(args.state_in),
        "state_bytes": summary["bytes"],
        "save_elapsed_sec": round(save_elapsed, 6),
        "load_elapsed_sec": round(load_elapsed, 6),
        "prefix_tokens": len(ids),
        "first_token": {"id": first_id, "text": tokenizer.decode([first_id])},
        "restored_argmax": {"id": restored_argmax, "text": tokenizer.decode([restored_argmax])},
        "full_argmax": {"id": full_argmax, "text": tokenizer.decode([full_argmax])},
        "metadata": payload.get("metadata", {}),
    }
    print(json.dumps(payload_out, ensure_ascii=False, indent=2))
    if not payload_out["ok"]:
        raise SystemExit(1)


def verify_rlvr(args: argparse.Namespace) -> None:
    from .rlvr import verify_file

    print(json.dumps(verify_file(args.rlvr_data), ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.cmd == "model-info":
        model_info(args)
    elif args.cmd == "check-contract":
        check_contract(args)
    elif args.cmd == "smoke":
        smoke(args)
    elif args.cmd == "probe-kernel":
        probe_kernel(args)
    elif args.cmd == "train-tiny":
        train_tiny(args)
    elif args.cmd == "train-packed":
        train_packed(args)
    elif args.cmd == "train-answer":
        train_answer(args)
    elif args.cmd == "train-multitask":
        train_multitask(args)
    elif args.cmd == "eval-loss":
        eval_loss(args)
    elif args.cmd == "eval-answer-loss":
        eval_answer_loss(args)
    elif args.cmd == "generate":
        generate(args)
    elif args.cmd == "fast-generate":
        fast_generate(args)
    elif args.cmd == "eval-english":
        eval_english(args)
    elif args.cmd == "quality-gate":
        quality_gate(args)
    elif args.cmd == "diagnose-decode":
        diagnose_decode(args)
    elif args.cmd == "decode-parity":
        decode_parity(args)
    elif args.cmd == "bench-decode":
        bench_decode(args)
    elif args.cmd == "serve":
        serve(args)
    elif args.cmd == "state-prefill":
        state_prefill(args)
    elif args.cmd == "state-restore":
        state_restore(args)
    elif args.cmd == "state-roundtrip":
        state_roundtrip(args)
    elif args.cmd == "verify-rlvr":
        verify_rlvr(args)
    else:
        raise SystemExit(args.cmd)


if __name__ == "__main__":
    main()
