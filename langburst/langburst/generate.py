from __future__ import annotations

import argparse
import time

from .cli_features import add_adapter_arg, add_model_path_args, add_runtime_feature_args, engine_model_spec_from_args
from .core.defaults import serving_recent_window_default
from .engines import ensure_engines_loaded, engine_registry
from .engines.base import EngineChatRequest, EngineSamplingParams


def main() -> None:
    ensure_engines_loaded()
    ap = argparse.ArgumentParser(description="LangBurst engine-backed chat/generation runner")
    ap.add_argument("--engine", choices=engine_registry.ids(), default=engine_registry.default_engine_id(), help="serving engine provider; native is the default")
    ap.add_argument("--model", default=None, help="engine model path/name; defaults to --hf-model")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--qwen36-lowbit", action="store_true", help="request LangBurst Qwen3.6 low-bit/custom-model bridge")
    ap.add_argument("--recurrent-state", action="store_true", help="request recurrent-state support from the selected engine")
    ap.add_argument("--vllm-custom-model", default=None, help="optional vLLM custom model bridge id/class for LangBurst Qwen3.6")
    ap.add_argument("--enable-mtp", action="store_true", help="enable vLLM MTP speculative decoding when supported")
    ap.add_argument("--mtp-speculative-tokens", type=int, default=2, help="number of vLLM MTP speculative tokens")
    ap.add_argument("--no-trust-remote-code", action="store_true")
    add_model_path_args(ap, required=False)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=serving_recent_window_default())
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--cpu-embed", action="store_true")
    add_adapter_arg(ap)
    add_runtime_feature_args(ap)
    args = ap.parse_args()

    try:
        spec = engine_model_spec_from_args(args)
        backend = engine_registry.create(spec, engine_id=args.engine)
        req = EngineChatRequest(
            request_id="cli",
            model=spec.public_name,
            messages=[{"role": "system", "content": args.system}, {"role": "user", "content": args.prompt}]
            if args.system
            else [{"role": "user", "content": args.prompt}],
            sampling=EngineSamplingParams(
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else -1,
            ),
            stream=args.stream,
        )
        t0 = time.perf_counter()
        if args.stream:
            out_text = ""
            for chunk in backend.stream_chat(req):
                out_text += chunk.text
                print(chunk.text, end="", flush=True)
            print()
        else:
            result = backend.generate_chat(req)
            out_text = result.text
            print(out_text)
        t1 = time.perf_counter()
    except RuntimeError as exc:
        ap.exit(2, f"langburst-chat: error: {exc}\n")
    except ValueError as exc:
        ap.error(str(exc))

    if args.stats:
        import sys

        approx_tokens = max(1, len(out_text.split()))
        dt = max(t1 - t0, 1e-9)
        print(f"[engine:{args.engine}] elapsed={dt:.3f}s approx_word/s={approx_tokens/dt:.2f}", file=sys.stderr)


if __name__ == "__main__":
    main()
