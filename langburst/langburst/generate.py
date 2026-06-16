from __future__ import annotations

import argparse
import time

import torch

from .cli_features import add_adapter_arg, add_model_path_args, add_runtime_feature_args, runtime_features_from_args
from .core.adapter import adapter_registry
from .core.runtime import GenerationConfig, RuntimeEngine
from .core.text_stream import StreamingTextDecoder


def main() -> None:
    ap = argparse.ArgumentParser(description="Adapter runtime chat/generation runner")
    add_model_path_args(ap)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=16384)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print prefill/decode timing to stderr")
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto", help="where low-bit layer weights live between matvecs")
    ap.add_argument("--gpu-embed-head", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cpu-embed", action="store_true", help="offload only token embeddings to CPU if 16GB VRAM is too tight")
    add_adapter_arg(ap)
    add_runtime_feature_args(ap)
    args = ap.parse_args()
    features = runtime_features_from_args(args)

    engine = RuntimeEngine(
        adapter=adapter_registry.get(args.adapter),
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        cpu_embed=args.cpu_embed,
        features=features,
    )
    prompt_ids = engine.encode_prompt(args.prompt, args.system)
    eos = []
    for name in ("eos_token_id", "pad_token_id"):
        val = getattr(engine.tokenizer, name, None)
        if isinstance(val, int):
            eos.append(val)
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_ids=tuple(set(eos)),
    )

    t0 = time.perf_counter()
    if args.stream:
        out_ids: list[int] = []
        decoder = StreamingTextDecoder(engine.tokenizer, skip_special_tokens=False)
        for tid in engine.generate_ids(prompt_ids, gen_cfg):
            out_ids.append(tid)
            text = decoder.push(tid)
            if text:
                print(text, end="", flush=True)
        text = decoder.flush()
        if text:
            print(text, end="", flush=True)
    else:
        out_ids = engine.generate_ids_greedy_gpu(prompt_ids, gen_cfg)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    if args.stream:
        print()
    else:
        print(engine.tokenizer.decode(out_ids, skip_special_tokens=True))
    if args.stats:
        import sys
        dt = max(t1 - t0, 1e-9)
        print(f"[runtime] generated={len(out_ids)} elapsed={dt:.3f}s tok/s={len(out_ids)/dt:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
