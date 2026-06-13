#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_MAMBA = ROOT / "mamba"
if LOCAL_MAMBA.exists() and str(LOCAL_MAMBA) not in sys.path:
    sys.path.insert(0, str(LOCAL_MAMBA))

from mamba_ssm.utils.generation import InferenceParams
from mamba3_kr.cli import _fast_decode_once, load_or_new, setup_perf
from scripts.mamba3_safe_chat import (
    clean_answer,
    clean_generated_answer,
    generate,
    iter_generate,
    make_prompt,
    truncate_after_sentence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent Neurova Mamba-3 chat server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", default="mimo-r4-tiny")
    parser.add_argument("--tokenizer", default="llama31")
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/mamba3_current/model.pt"))
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-new", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup-new", type=int, default=2)
    parser.add_argument("--stop-after-sentence", action="store_true", default=True)
    parser.add_argument("--min-sentence-chars", type=int, default=18)
    parser.add_argument("--decode-mode", choices=["safe", "cache-verify", "exact-cache", "cache"], default="safe")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--nan-check", action="store_true")
    parser.add_argument("--cache-parity-guard", action="store_true", default=True)
    parser.add_argument("--no-cache-parity-guard", dest="cache_parity_guard", action="store_false")
    parser.add_argument("--cache-parity-steps", type=int, default=8)
    return parser.parse_args()


class State:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.lock = threading.Lock()
        setup_perf(args.device)
        load_args = SimpleNamespace(
            cmd="fast-generate",
            mode=args.mode,
            tokenizer=args.tokenizer,
            checkpoint=args.checkpoint,
            device=args.device,
            dtype=args.dtype,
            activation_checkpointing=False,
        )
        self.model, self.tokenizer, _ = load_or_new(load_args)
        self.model.eval()
        if args.warmup_new > 0:
            warm_args = copy.copy(args)
            warm_args.max_new = min(args.max_new, args.warmup_new)
            warm_args.stop_after_sentence = False
            if args.decode_mode == "cache":
                _fast_chat(self.model, self.tokenizer, warm_args, "Can you help me?")
            else:
                generate(self.model, self.tokenizer, warm_args, "Can you help me?")


def _fast_chat(model, tokenizer, args: argparse.Namespace, prompt: str) -> tuple[str, int, float]:
    decode_args = copy.copy(args)
    use_safe = args.decode_mode == "safe"
    decode_args.exact_cache = args.decode_mode == "exact-cache"
    decode_args.cache_verify = args.decode_mode == "cache-verify"
    if args.decode_mode == "cache" and args.cache_parity_guard:
        use_safe = not _cache_argmax_parity_ok(model, tokenizer, args, make_prompt(prompt))
    decode_args.safe_decode = use_safe
    result = _fast_decode_once(model, tokenizer, decode_args, make_prompt(prompt))
    text = str(result.get("text", ""))
    answer = clean_answer(text, make_prompt(prompt))
    if not answer:
        answer = clean_generated_answer(text)
    if args.stop_after_sentence:
        answer = truncate_after_sentence(answer, int(args.min_sentence_chars))
    return answer, int(result.get("new_tokens", 0)), float(result.get("elapsed_sec", 0.0) or 0.0)


@torch.inference_mode()
def _cache_argmax_parity_ok(model, tokenizer, args: argparse.Namespace, prompt: str) -> bool:
    ids = tokenizer.encode(prompt, add_eos=False)
    ids = ids[-args.seq_len:]
    if not ids:
        return False
    inference_params = InferenceParams(max_seqlen=args.seq_len + args.max_new + 4, max_batch_size=1)
    input_ids = torch.tensor([ids], device=args.device, dtype=torch.long)
    cache_logits = model(input_ids, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    full_logits = model(input_ids, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
    if int(torch.argmax(cache_logits).item()) != int(torch.argmax(full_logits).item()):
        return False
    inference_params.seqlen_offset += input_ids.shape[1]
    generated: list[int] = []
    for _ in range(max(1, int(args.cache_parity_steps))):
        next_id = int(torch.argmax(full_logits).item())
        generated.append(next_id)
        if next_id == tokenizer.eos_id:
            return True
        step_ids = torch.tensor([[next_id]], device=args.device, dtype=torch.long)
        cache_logits = model(step_ids, inference_params=inference_params, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
        inference_params.seqlen_offset += 1
        full_ids = torch.tensor([ids + generated], device=args.device, dtype=torch.long)
        full_logits = model(full_ids, num_last_tokens=1).logits[0, -1, : tokenizer.vocab_size]
        if int(torch.argmax(cache_logits).item()) != int(torch.argmax(full_logits).item()):
            return False
    return True


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_payload(self) -> dict:
            length = int(self.headers.get("content-length", "0") or "0")
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(
                    200,
                    {
                        "ok": True,
                        "mode": state.args.mode,
                        "checkpoint": str(state.args.checkpoint),
                        "device": state.args.device,
                        "decode_mode": state.args.decode_mode,
                        "cache_parity_guard": bool(state.args.cache_parity_guard),
                        "cache_parity_steps": int(state.args.cache_parity_steps),
                    },
                )
                return
            self._json(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path not in {"/generate", "/stream"}:
                self._json(404, {"ok": False, "error": "not found"})
                return
            try:
                payload = self._read_payload()
                req_args = copy.copy(state.args)
                req_args.max_new = int(payload.get("max_new") or req_args.max_new)
                prompt = str(payload.get("prompt", "")).strip()
                if not prompt:
                    raise ValueError("missing prompt")
                t0 = time.time()
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    with state.lock:
                        if req_args.decode_mode == "safe":
                            tokens = 0
                            elapsed = 0.0
                            pending = ""
                            for delta, tokens, elapsed in iter_generate(state.model, state.tokenizer, req_args, prompt):
                                if delta:
                                    pending += delta
                                    if len(pending) >= 16 or pending.endswith((".", "!", "?", "\n")):
                                        self.wfile.write(pending.encode("utf-8"))
                                        self.wfile.flush()
                                        pending = ""
                            if pending:
                                self.wfile.write(pending.encode("utf-8"))
                                self.wfile.flush()
                        else:
                            answer, tokens, elapsed = _fast_chat(state.model, state.tokenizer, req_args, prompt)
                            self.wfile.write(answer.encode("utf-8"))
                            self.wfile.flush()
                    suffix = f"\n({tokens / max(elapsed, 1e-9):.1f} tok/s)\n"
                    self.wfile.write(suffix.encode("utf-8"))
                    self.wfile.flush()
                with state.lock:
                    if self.path != "/stream":
                        if req_args.decode_mode in {"cache-verify", "exact-cache", "cache"}:
                            answer, tokens, elapsed = _fast_chat(state.model, state.tokenizer, req_args, prompt)
                        else:
                            answer, tokens, elapsed = _fast_chat(state.model, state.tokenizer, req_args, prompt)
                        self._json(
                            200,
                            {
                                "ok": True,
                                "answer": answer,
                                "tokens": tokens,
                                "elapsed": elapsed,
                                "tok_s": tokens / max(elapsed, 1e-9),
                            },
                        )
            except Exception as exc:
                print(f"request failed path={self.path}: {exc!r}", flush=True)
                self._json(500, {"ok": False, "error": str(exc)})

    return Handler


def main() -> None:
    args = parse_args()
    state = State(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(
        f"Neurova Mamba-3 chat server ready on {args.host}:{args.port} "
        f"({args.mode}, {args.checkpoint})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
