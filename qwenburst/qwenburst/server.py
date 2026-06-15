from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .adapters import Qwen36Adapter  # registers qwen36
from .cli_features import add_runtime_feature_args, runtime_features_from_args
from .core.adapter import adapter_registry
from .core.features import RuntimeFeatureOverride, RuntimeFeatures
from .core.runtime import GenerationConfig, RuntimeEngine


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[dict[str, Any]] | None = ""


class ChatCompletionRequest(BaseModel):
    model: str = "qwenburst-qwen3.6-27b-q4-marlin"
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_k: int = 0
    stream: bool = False
    runtime_profile: Literal["original", "stateful", "research"] | None = None
    kv_window_policy: Literal["error", "shift", "ring"] | None = None
    stateful_chat: bool | None = None
    infinite_streaming: bool | None = None
    snapshots: bool | None = None
    boundary_decay: float | None = None
    episodic_memory: bool | None = None
    ttt_sidecar: bool | None = None
    speculative_mtp: bool | None = None
    cuda_graph: bool | None = None
    block_prefill: bool | None = None
    prefill_chunk_size: int | None = None


def _request_generation_config(engine: RuntimeEngine, req: ChatCompletionRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=req.max_new_tokens or req.max_completion_tokens or req.max_tokens or 256,
        temperature=req.temperature,
        top_k=req.top_k,
        eos_token_ids=engine.eos_token_ids(),
    )


def _request_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in req.messages]


def _request_features(engine: RuntimeEngine, req: ChatCompletionRequest) -> RuntimeFeatures:
    features = RuntimeFeatures.from_profile(req.runtime_profile) if req.runtime_profile else engine.features
    return features.with_overrides(RuntimeFeatureOverride.from_obj(req))


def create_app(engine: RuntimeEngine):
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qwenburst-server requires fastapi and uvicorn") from exc

    app = FastAPI(title="QwenBurst OpenAI-Compatible Server")

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "qwenburst",
                }
            ],
        }

    @app.get("/v1/qwenburst/features")
    def features():
        return engine.features.summary()

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        if req.stream:
            def events():
                request_features = _request_features(engine, req)
                for _, text in engine.completion_tokens(
                    _request_messages(req),
                    _request_generation_config(engine, req),
                    features=request_features,
                ):
                    payload = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": engine.model_name,
                        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                    }
                    yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                done = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield "data: " + json.dumps(done, ensure_ascii=False) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        ids = engine.completion_ids_greedy_gpu(
            _request_messages(req),
            _request_generation_config(engine, req),
            features=_request_features(engine, req),
        )
        content = engine.tokenizer.decode(ids, skip_special_tokens=True)
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": engine.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenAI-compatible QwenBurst thin server")
    ap.add_argument("--hf-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B"))
    ap.add_argument("--qb-model", type=Path, default=Path("/home/user/models/Qwen3.6-27B-qb4-marlin-fused"))
    ap.add_argument("--model-name", default="qwenburst-qwen3.6-27b-q4-marlin")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=8192)
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--gpu-embed-head", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cpu-embed", action="store_true", help="offload only token embeddings to CPU if 16GB VRAM is too tight")
    ap.add_argument("--adapter", default="qwen36", choices=("qwen36",), help="model adapter")
    add_runtime_feature_args(ap)
    args = ap.parse_args()
    features = runtime_features_from_args(args)

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("qwenburst-server requires uvicorn") from exc

    engine = RuntimeEngine(
        adapter=adapter_registry.get(args.adapter),
        hf_model=args.hf_model,
        qb_model=args.qb_model,
        device=args.device,
        recent_window=args.recent_window,
        weight_device=args.weight_device,
        cpu_embed=args.cpu_embed,
        model_name=args.model_name,
        features=features,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
