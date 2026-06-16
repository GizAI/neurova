from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel

from .cli_features import add_adapter_arg, add_runtime_feature_args, runtime_features_from_args
from .core.adapter import adapter_registry
from .core.defaults import (
    DEFAULT_KV_BLOCK_SIZE,
    DEFAULT_KV_BLOCKS,
    DEFAULT_MAX_GENERATION_TOKENS,
    DEFAULT_MAX_BATCHED_TOKENS,
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MAX_STATE_POOL_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_RESERVE_FREE_VRAM_MIB,
    DEFAULT_SERVING_RECENT_WINDOW,
)
from .core.features import RuntimeFeatureOverride, RuntimeFeatures
from .core.manager import EngineManager, EngineResourcePolicy, ModelResourceSpec, load_model_specs
from .core.platform import PLATFORM_NAME
from .core.runtime import GenerationConfig, RuntimeEngine
from .core.text_stream import StreamingTextDecoder


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[dict[str, Any]] | None = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
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
    state_pool: bool | None = None
    snapshots: bool | None = None
    boundary_decay: float | None = None
    gpu_sampling: bool | None = None
    speculative_decoding: bool | None = None
    cuda_graph: bool | None = None
    block_prefill: bool | None = None
    infinite_streaming: bool | None = None
    episodic_memory: bool | None = None
    ttt_sidecar: bool | None = None
    prefill_chunk_size: int | None = None


@dataclass(frozen=True)
class CompletionContext:
    engine: RuntimeEngine
    prompt_ids: list[int]
    generation_tokens: int
    features: RuntimeFeatures
    gen_cfg: GenerationConfig
    completion_id: str
    created: int


def _requested_generation_tokens(req: ChatCompletionRequest) -> int:
    return req.max_new_tokens or req.max_completion_tokens or req.max_tokens or 256


def _request_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in req.messages]


def _request_features(engine: RuntimeEngine, req: ChatCompletionRequest) -> RuntimeFeatures:
    features = RuntimeFeatures.from_profile(req.runtime_profile) if req.runtime_profile else engine.features
    return features.with_overrides(RuntimeFeatureOverride.from_obj(req))


def _sse_payload(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _record_oom(manager: EngineManager, engine: RuntimeEngine, exc: BaseException) -> dict[str, Any]:
    manager.mark_runtime_error(engine.model_name, exc)
    cleared = manager.clear_runtime_pools(engine.model_name)
    return {
        "message": "CUDA out of memory during generation; runtime pools were cleared",
        "type": "cuda_oom",
        "model": engine.model_name,
        "cleared_runtime_pools": cleared,
    }


def _completion_context(manager: EngineManager, req: ChatCompletionRequest) -> CompletionContext:
    engine = manager.get(req.model)
    prompt_ids = engine.encode_messages(_request_messages(req))
    generation_tokens = _requested_generation_tokens(req)
    manager.validate_generation_request(
        prompt_tokens=len(prompt_ids),
        generation_tokens=generation_tokens,
    )
    features = _request_features(engine, req)
    return CompletionContext(
        engine=engine,
        prompt_ids=prompt_ids,
        generation_tokens=generation_tokens,
        features=features,
        gen_cfg=GenerationConfig(
            max_new_tokens=generation_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            eos_token_ids=engine.eos_token_ids(),
        ),
        completion_id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
    )


def _completion_ids(manager: EngineManager, req: ChatCompletionRequest, ctx: CompletionContext) -> list[int]:
    if req.temperature <= 0 and req.top_k <= 0:
        worker = manager.create_batch_worker(req.model, ctx.features)
        handle = worker.submit(
            ctx.prompt_ids,
            max_new_tokens=ctx.generation_tokens,
            eos_token_ids=ctx.engine.eos_token_ids(),
            request_id=ctx.completion_id,
        )
        return handle.wait_ids()
    with manager.acquire_request():
        return ctx.engine.completion_ids_greedy_gpu_from_ids(
            ctx.prompt_ids,
            ctx.gen_cfg,
            features=ctx.features,
        )


def create_app(engine_or_manager: RuntimeEngine | EngineManager):
    try:
        from fastapi import FastAPI
        from fastapi import HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langburst-server requires fastapi and uvicorn") from exc

    app = FastAPI(title=f"{PLATFORM_NAME} OpenAI-Compatible Server")
    manager = engine_or_manager if isinstance(engine_or_manager, EngineManager) else EngineManager.from_engine(engine_or_manager)

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": row["id"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "langburst",
                }
                for row in manager.list_models()
            ],
        }

    @app.get("/v1/langburst/models")
    def langburst_models():
        return manager.summary()

    @app.get("/v1/langburst/health")
    def langburst_health():
        return manager.health()

    @app.get("/v1/langburst/features")
    def features(model: str | None = None):
        try:
            return manager.resolve_plan(model).summary()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/v1/langburst/models/{model_name}")
    def unload_model(model_name: str):
        return {"model": model_name, "unloaded": manager.unload(model_name)}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        try:
            ctx = _completion_context(manager, req)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        if req.stream:
            def events():
                handle = None
                try:
                    if req.temperature <= 0 and req.top_k <= 0:
                        worker = manager.create_batch_worker(req.model, ctx.features)
                        handle = worker.submit(
                            ctx.prompt_ids,
                            max_new_tokens=ctx.generation_tokens,
                            eos_token_ids=ctx.engine.eos_token_ids(),
                            request_id=ctx.completion_id,
                        )
                        decoder = StreamingTextDecoder(ctx.engine.tokenizer, skip_special_tokens=False)
                        for token_id in handle.iter_token_ids():
                            text = decoder.push(token_id)
                            if not text:
                                continue
                            payload = {
                                "id": ctx.completion_id,
                                "object": "chat.completion.chunk",
                                "created": ctx.created,
                                "model": ctx.engine.model_name,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                            }
                            yield _sse_payload(payload)
                        text = decoder.flush()
                        if text:
                            payload = {
                                "id": ctx.completion_id,
                                "object": "chat.completion.chunk",
                                "created": ctx.created,
                                "model": ctx.engine.model_name,
                                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                            }
                            yield _sse_payload(payload)
                    else:
                        with manager.acquire_request():
                            for _, text in ctx.engine.completion_tokens_from_ids(
                                ctx.prompt_ids,
                                ctx.gen_cfg,
                                features=ctx.features,
                            ):
                                payload = {
                                    "id": ctx.completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": ctx.created,
                                    "model": ctx.engine.model_name,
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                                yield _sse_payload(payload)
                except TimeoutError as exc:
                    yield _sse_payload({"error": {"message": str(exc), "type": "admission_error"}})
                except torch.cuda.OutOfMemoryError as exc:
                    yield _sse_payload({"error": _record_oom(manager, ctx.engine, exc)})
                finally:
                    if handle is not None and not handle.done.is_set():
                        handle.cancel()
                done = {
                    "id": ctx.completion_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": ctx.engine.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield _sse_payload(done)
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream")

        try:
            ids = _completion_ids(manager, req, ctx)
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(status_code=503, detail=_record_oom(manager, ctx.engine, exc)) from exc
        content = ctx.engine.tokenizer.decode(ids, skip_special_tokens=True)
        return JSONResponse(
            {
                "id": ctx.completion_id,
                "object": "chat.completion",
                "created": ctx.created,
                "model": ctx.engine.model_name,
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
    ap = argparse.ArgumentParser(description="OpenAI-compatible adapter runtime server")
    ap.add_argument("--hf-model", type=Path, default=None, help="HF model dir; required unless --models-json is used")
    ap.add_argument("--qb-model", type=Path, default=None, help="converted runtime model dir; required unless --models-json is used")
    ap.add_argument("--model-name", default=None, help="served model id; defaults to the adapter descriptor model name")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=DEFAULT_SERVING_RECENT_WINDOW)
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--gpu-embed-head", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cpu-embed", action="store_true", help="offload only token embeddings to CPU if 16GB VRAM is too tight")
    add_adapter_arg(ap)
    ap.add_argument("--max-loaded-models", type=int, default=1)
    ap.add_argument("--max-active-requests", type=int, default=1)
    ap.add_argument("--max-queued-requests", type=int, default=0)
    ap.add_argument("--admission-timeout-s", type=float, default=None)
    ap.add_argument("--reserve-free-vram-mib", type=int, default=DEFAULT_RESERVE_FREE_VRAM_MIB)
    ap.add_argument("--max-state-pool-size", type=int, default=DEFAULT_MAX_STATE_POOL_SIZE)
    ap.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS, help="reject longer prompts before allocating runtime state")
    ap.add_argument("--max-generation-tokens", type=int, default=DEFAULT_MAX_GENERATION_TOKENS, help="reject longer generations before allocating runtime state")
    ap.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_BATCHED_TOKENS, help="continuous-batching token budget")
    ap.add_argument("--batch-prefill-chunk-size", type=int, default=DEFAULT_PREFILL_CHUNK_SIZE, help="continuous-batching prefill chunk size")
    ap.add_argument("--kv-block-size", type=int, default=DEFAULT_KV_BLOCK_SIZE, help="paged-KV logical block size")
    ap.add_argument("--kv-blocks", type=int, default=DEFAULT_KV_BLOCKS, help="paged-KV block table capacity")
    ap.add_argument("--models-json", type=Path, default=None, help="multi-model resource spec JSON")
    add_runtime_feature_args(ap)
    args = ap.parse_args()
    features = runtime_features_from_args(args)

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langburst-server requires uvicorn") from exc

    if args.models_json:
        specs = load_model_specs(args.models_json, features)
    else:
        if args.hf_model is None or args.qb_model is None:
            ap.error("--hf-model and --qb-model are required unless --models-json is used")
        adapter = adapter_registry.get(args.adapter)
        specs = [
            ModelResourceSpec(
                model_name=args.model_name or adapter.descriptor.default_model_name,
                adapter_id=args.adapter,
                hf_model=args.hf_model,
                qb_model=args.qb_model,
                device=args.device,
                recent_window=args.recent_window,
                weight_device=args.weight_device,
                cpu_embed=args.cpu_embed,
                runtime_features=features,
            )
        ]
    manager = EngineManager(
        specs,
        policy=EngineResourcePolicy(
            max_loaded_models=args.max_loaded_models,
            max_active_requests=args.max_active_requests,
            max_queued_requests=args.max_queued_requests,
            admission_timeout_s=args.admission_timeout_s,
            reserve_free_vram_mib=args.reserve_free_vram_mib,
            max_state_pool_size=args.max_state_pool_size,
            max_prompt_tokens=args.max_prompt_tokens,
            max_generation_tokens=args.max_generation_tokens,
            max_num_batched_tokens=args.max_num_batched_tokens,
            prefill_chunk_size=args.batch_prefill_chunk_size,
            kv_block_size=args.kv_block_size,
            kv_blocks=args.kv_blocks,
        ),
    )
    app = create_app(manager)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
