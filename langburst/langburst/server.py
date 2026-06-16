from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel
from starlette.requests import Request

from .cli_features import add_adapter_arg, add_model_path_args, add_runtime_feature_args, runtime_features_from_args
from .core.defaults import (
    DEFAULT_MAX_BATCHED_TOKENS,
    DEFAULT_MAX_GENERATION_TOKENS,
    DEFAULT_MAX_STATE_POOL_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_RESERVE_FREE_VRAM_MIB,
    kv_block_size_default,
    kv_blocks_default,
    max_prompt_tokens_default,
    serving_recent_window_default,
)
from .core.features import RuntimeFeatures
from .core.platform import PLATFORM_NAME
from .core.text_stream import StreamingTextDecoder
from .core.usage import RequestUsage
from .engines import ensure_engines_loaded, engine_registry
from .engines.base import EngineBackend, EngineChatRequest, EngineFeatureRequest, EngineModelSpec, EngineSamplingParams
from .engines.native_impl.manager import EngineManager, EngineResourcePolicy, ModelResourceSpec, load_model_specs
from .engines.native_impl.runtime import GenerationConfig


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | str
    content: str | list[dict[str, Any]] | None = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_new_tokens: int | None = None
    min_tokens: int | None = None
    min_new_tokens: int | None = None
    temperature: float = 0.0
    top_k: int = 0
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram_size: int = 0
    logit_bias: dict[str, float] | dict[int, float] | None = None
    bad_words_ids: list[int] | list[list[int]] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    stop: str | list[str] | None = None
    stop_sequences: list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    include_stop_str_in_output: bool = False
    seed: int | None = None
    n: int = 1
    logprobs: bool | None = None
    top_logprobs: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    session_id: str | None = None
    stateful_session: bool = False
    reset_session: bool = False
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
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
    prefix_cache: bool | None = None
    infinite_streaming: bool | None = None
    episodic_memory: bool | None = None
    ttt_sidecar: bool | None = None
    prefill_chunk_size: int | None = None


class SessionCreateRequest(BaseModel):
    model: str | None = None


def _sse_payload(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _requested_generation_tokens(req: ChatCompletionRequest) -> int:
    fallback = int(os.environ.get("LANGBURST_DEFAULT_MAX_TOKENS", "1024"))
    return req.max_new_tokens or req.max_completion_tokens or req.max_tokens or fallback


def _requested_min_generation_tokens(req: ChatCompletionRequest) -> int:
    return int(req.min_new_tokens if req.min_new_tokens is not None else req.min_tokens or 0)


def _request_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in req.messages]


def _stop_strings(req: ChatCompletionRequest) -> list[str]:
    out: list[str] = []
    if isinstance(req.stop, str):
        out.append(req.stop)
    elif isinstance(req.stop, list):
        out.extend(str(v) for v in req.stop)
    if req.stop_sequences:
        out.extend(str(v) for v in req.stop_sequences)
    return [s for s in out if s]


def _usage_payload(usage: RequestUsage) -> dict[str, Any]:
    return {
        **usage.openai_usage(),
        "performance": usage.performance(),
    }


def _native_request_timeout_s() -> float | None:
    raw = os.environ.get("LANGBURST_REQUEST_TIMEOUT_S", "300").strip().lower()
    if raw in {"", "0", "none", "off", "false"}:
        return None
    return max(0.0, float(raw))


def _log_native_request_metrics(*, usage: RequestUsage, stream: bool, model: str, request_id: str) -> None:
    perf = usage.performance()
    parts: list[str] = [
        f"request_id={request_id}",
        f"model={model}",
        f"stream={stream}",
        f"prompt_tokens={usage.prompt_tokens}",
        f"cached_tokens={usage.cached_input_tokens}",
        f"completion_tokens={usage.completion_tokens}",
    ]
    if usage.requested_completion_tokens is not None:
        parts.append(f"requested_completion_tokens={usage.requested_completion_tokens}")
    if usage.finish_reason is not None:
        parts.append(f"finish_reason={usage.finish_reason}")
    for key in ("queue_wait_s", "ttft_s", "e2e_s", "decode_s", "e2e_tok_s", "decode_tok_s", "mean_itl_s"):
        value = perf.get(key)
        if value is not None:
            parts.append(f"{key}={float(value):.4f}")
    print("[langburst][native-manager][profile] " + " ".join(parts), flush=True)


def _logit_bias(req: ChatCompletionRequest) -> dict[int, float] | None:
    if not req.logit_bias:
        return None
    return {int(token): float(bias) for token, bias in req.logit_bias.items()}


def _flat_token_ids(value: list[int] | list[list[int]] | None) -> tuple[int, ...]:
    if not value:
        return ()
    out: list[int] = []
    for item in value:
        if isinstance(item, list):
            out.extend(int(v) for v in item)
        else:
            out.append(int(item))
    return tuple(out)


def _request_feature_overrides(req: ChatCompletionRequest, base: RuntimeFeatures) -> RuntimeFeatures:
    profile = req.runtime_profile or base.profile
    features = base if profile == base.profile else RuntimeFeatures.from_profile(profile).with_overrides(
        kv_cache_dtype=base.kv_cache_dtype,
        prefill_chunk_size=base.prefill_chunk_size,
    )
    overrides = {
        "kv_window_policy": req.kv_window_policy,
        "stateful_chat": req.stateful_chat,
        "state_pool": req.state_pool,
        "snapshots": req.snapshots,
        "boundary_decay": req.boundary_decay,
        "gpu_sampling": req.gpu_sampling,
        "speculative_decoding": req.speculative_decoding,
        "cuda_graph": req.cuda_graph,
        "block_prefill": req.block_prefill,
        "prefix_cache": req.prefix_cache,
        "infinite_streaming": req.infinite_streaming,
        "episodic_memory": req.episodic_memory,
        "ttt_sidecar": req.ttt_sidecar,
        "prefill_chunk_size": req.prefill_chunk_size,
    }
    return features.with_overrides_from_mapping({k: v for k, v in overrides.items() if v is not None})


def _native_generation_config(engine, req: ChatCompletionRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=_requested_generation_tokens(req),
        min_new_tokens=_requested_min_generation_tokens(req),
        temperature=float(req.temperature),
        top_k=max(0, int(req.top_k)),
        top_p=1.0 if req.top_p is None else float(req.top_p),
        min_p=0.0 if req.min_p is None else float(req.min_p),
        repetition_penalty=1.0 if req.repetition_penalty is None else float(req.repetition_penalty),
        presence_penalty=float(req.presence_penalty),
        frequency_penalty=float(req.frequency_penalty),
        no_repeat_ngram_size=int(req.no_repeat_ngram_size),
        logit_bias=_logit_bias(req),
        bad_token_ids=_flat_token_ids(req.bad_words_ids),
        suppress_tokens=tuple(int(t) for t in (req.suppress_tokens or []) + (req.begin_suppress_tokens or [])),
        seed=req.seed,
        eos_token_ids=engine.eos_token_ids(),
        stop_token_ids=tuple(int(t) for t in (req.stop_token_ids or [])),
        ignore_eos=bool(req.ignore_eos),
    )


def _native_stop_token_sequences(engine, req: ChatCompletionRequest) -> tuple[tuple[int, ...], ...]:
    out: list[tuple[int, ...]] = []
    for stop in _stop_strings(req):
        ids = tuple(int(t) for t in engine.tokenizer.encode(stop))
        if ids:
            out.append(ids)
    return tuple(out)


def _uses_native_session(req: ChatCompletionRequest) -> bool:
    return bool(req.session_id or req.stateful_session or req.previous_response_id)


def create_app(manager: EngineManager):
    """Native LangBurst server surface backed by the in-process EngineManager.

    This is intentionally kept beside the provider router. vLLM is the default
    engine provider, but native remains independently runnable without vLLM.
    """

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langburst-server requires fastapi and uvicorn") from exc

    app = FastAPI(title=f"{PLATFORM_NAME} Native Engine")

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": manager.list_models()}

    @app.get("/v1/langburst/models")
    def langburst_models():
        return manager.summary()

    @app.get("/v1/langburst/health")
    def langburst_health():
        return manager.health()

    @app.get("/v1/langburst/sessions")
    def list_sessions():
        return manager.sessions.summary()

    @app.post("/v1/langburst/sessions")
    def create_session(req: SessionCreateRequest | None = None):
        return {"id": manager.create_session(), "model": req.model if req else None}

    @app.delete("/v1/langburst/sessions/{session_id}")
    def delete_session(session_id: str):
        return {"deleted": manager.delete_session(session_id)}

    @app.delete("/v1/langburst/models/{model_name}")
    def unload_model(model_name: str):
        return {"model": model_name, "unloaded": manager.unload(model_name)}

    @app.post("/v1/chat/completions")
    def chat_completions(request: Request, req: ChatCompletionRequest):
        try:
            engine = manager.get(req.model)
            features = _request_feature_overrides(req, engine.features)
            prompt_ids = engine.encode_messages(_request_messages(req))
            generation_tokens = _requested_generation_tokens(req)
            manager.validate_generation_request(prompt_tokens=len(prompt_ids), generation_tokens=generation_tokens)
            gen_cfg = _native_generation_config(engine, req)
            session_id = req.session_id or (manager.create_session() if req.stateful_session else None)
            session_record = (
                manager.get_session(session_id=session_id, model_name=engine.model_name, features=features)
                if session_id
                else None
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        worker = manager.create_batch_worker(engine.model_name, features)
        stop_sequences = _native_stop_token_sequences(engine, req)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        if req.stream:
            async def events():
                try:
                    usage = RequestUsage(prompt_tokens=len(prompt_ids))
                    handle = worker.submit(
                        prompt_ids,
                        max_new_tokens=generation_tokens,
                        eos_token_ids=gen_cfg.eos_token_ids,
                        generation_config=gen_cfg,
                        prompt_cache_key=req.prompt_cache_key,
                        session_record=session_record,
                        stop_sequences=stop_sequences,
                        include_stop_str_in_output=req.include_stop_str_in_output,
                        request_id=request_id,
                    )
                    decoder = StreamingTextDecoder(engine.tokenizer, skip_special_tokens=False)
                    while True:
                        if await request.is_disconnected():
                            handle.cancel()
                            break
                        token_id, done = await asyncio.to_thread(handle.poll_output, 0.25)
                        if done:
                            break
                        if token_id is None:
                            continue
                        text = decoder.push(token_id)
                        if text:
                            yield _sse_payload(
                                {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": engine.model_name,
                                    **({"session_id": session_id} if session_id else {}),
                                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                                }
                            )
                        usage.apply_metrics(handle.metrics())
                        usage.requested_completion_tokens = generation_tokens
                        usage.finish_reason = handle.finish_reason
                    tail = decoder.flush()
                    usage.apply_metrics(handle.metrics())
                    usage.requested_completion_tokens = generation_tokens
                    usage.finish_reason = handle.finish_reason
                    _log_native_request_metrics(
                        usage=usage,
                        stream=True,
                        model=engine.model_name,
                        request_id=request_id,
                    )
                    if tail:
                        yield _sse_payload(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": engine.model_name,
                                **({"session_id": session_id} if session_id else {}),
                                "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": None}],
                            }
                        )
                    done = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": engine.model_name,
                        **({"session_id": session_id} if session_id else {}),
                        **({"usage": _usage_payload(usage)} if req.stream_options and req.stream_options.include_usage else {}),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": handle.finish_reason}],
                    }
                    yield _sse_payload(done)
                    yield "data: [DONE]\n\n"
                except BaseException as exc:
                    yield _sse_payload({"error": {"message": str(exc), "type": type(exc).__name__}})
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )

        try:
            with manager.acquire_request():
                handle = worker.submit(
                    prompt_ids,
                    max_new_tokens=generation_tokens,
                    eos_token_ids=gen_cfg.eos_token_ids,
                    generation_config=gen_cfg,
                    prompt_cache_key=req.prompt_cache_key,
                    session_record=session_record,
                    stop_sequences=stop_sequences,
                    include_stop_str_in_output=req.include_stop_str_in_output,
                    request_id=request_id,
                )
                ids = handle.wait_ids(timeout=_native_request_timeout_s())
                usage = RequestUsage(prompt_tokens=len(prompt_ids))
                usage.apply_metrics(handle.metrics())
                usage.requested_completion_tokens = generation_tokens
                usage.finish_reason = handle.finish_reason
                _log_native_request_metrics(
                    usage=usage,
                    stream=False,
                    model=engine.model_name,
                    request_id=request_id,
                )
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": engine.model_name,
                **({"session_id": session_id} if session_id else {}),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": engine.tokenizer.decode(ids, skip_special_tokens=True)},
                        "finish_reason": handle.finish_reason,
                    }
                ],
                "usage": _usage_payload(usage),
            }
        )

    return app


def _validate_request_shape(req: ChatCompletionRequest) -> None:
    if req.n != 1:
        raise ValueError("LangBurst currently supports n=1")
    if req.logprobs or req.top_logprobs:
        raise ValueError("logprobs are delegated only when the selected engine provider supports them")
    if req.tools or req.tool_choice or req.parallel_tool_calls:
        raise ValueError("tool calling is delegated only when the selected engine provider supports it")


def _engine_sampling(req: ChatCompletionRequest) -> EngineSamplingParams:
    return EngineSamplingParams(
        max_tokens=_requested_generation_tokens(req),
        min_tokens=_requested_min_generation_tokens(req),
        temperature=float(req.temperature),
        top_p=1.0 if req.top_p is None else float(req.top_p),
        top_k=-1 if int(req.top_k) <= 0 else int(req.top_k),
        stop=tuple(_stop_strings(req)),
        stop_token_ids=tuple(int(t) for t in (req.stop_token_ids or [])),
        seed=req.seed,
        ignore_eos=bool(req.ignore_eos),
    )


def _engine_feature_request_from_args(args: argparse.Namespace) -> EngineFeatureRequest:
    features = runtime_features_from_args(args)
    return EngineFeatureRequest.from_mapping(
        {
            **features.summary(),
            "qwen36_lowbit": bool(args.qwen36_lowbit or args.qb_model),
            "ring_kv": features.kv_window_policy == "ring",
            "recurrent_state": bool(args.recurrent_state or args.qwen36_lowbit or args.qb_model),
            "infinite_context": bool(features.infinite_streaming),
        }
    )


def _engine_chat_request(req: ChatCompletionRequest) -> EngineChatRequest:
    _validate_request_shape(req)
    return EngineChatRequest(
        request_id=f"chatcmpl-{uuid.uuid4().hex}",
        model=req.model,
        messages=_request_messages(req),
        sampling=_engine_sampling(req),
        stream=bool(req.stream),
        user=req.user,
        metadata=req.metadata,
        raw_request=req,
    )


def create_engine_app(backend: EngineBackend):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("langburst-server requires fastapi and uvicorn") from exc

    app = FastAPI(title=f"{PLATFORM_NAME} Engine Router")
    backend.start()

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": backend.list_models()}

    @app.get("/v1/langburst/health")
    def langburst_health():
        return backend.health()

    @app.get("/v1/langburst/engines")
    def langburst_engines():
        ensure_engines_loaded()
        return {
            "default": engine_registry.default_engine_id(),
            "data": [descriptor.summary() for descriptor in engine_registry.list()],
        }

    @app.get("/v1/langburst/features")
    def features():
        return {
            **backend.descriptor.summary(),
            "feature_plan": backend.feature_plan.summary(),
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: Request, req: ChatCompletionRequest):
        try:
            engine_req = _engine_chat_request(req)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if req.stream:
            async def events():
                try:
                    for chunk in backend.stream_chat(engine_req):
                        if await request.is_disconnected():
                            break
                        if chunk.text:
                            yield _sse_payload(
                                {
                                    "id": engine_req.request_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": chunk.model,
                                    "choices": [{"index": 0, "delta": {"content": chunk.text}, "finish_reason": None}],
                                }
                            )
                        if chunk.finish_reason:
                            done = {
                                "id": engine_req.request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": chunk.model,
                                **(
                                    {"usage": chunk.usage.openai_usage()}
                                    if chunk.usage is not None and req.stream_options and req.stream_options.include_usage
                                    else {}
                                ),
                                "choices": [{"index": 0, "delta": {}, "finish_reason": chunk.finish_reason}],
                            }
                            yield _sse_payload(done)
                    yield "data: [DONE]\n\n"
                except BaseException as exc:
                    yield _sse_payload({"error": {"message": str(exc), "type": type(exc).__name__}})
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
            )
        try:
            result = backend.generate_chat(engine_req)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return JSONResponse(
            {
                "id": engine_req.request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": result.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result.text},
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": result.usage.openai_usage(),
            }
        )

    return app


def _model_spec_from_args(args: argparse.Namespace) -> EngineModelSpec:
    model = args.model or (str(args.hf_model) if args.hf_model is not None else None)
    if model is None:
        raise ValueError("--model or --hf-model is required")
    extra = {
        "adapter": args.adapter,
        "qb_model": str(args.qb_model) if args.qb_model is not None else None,
        "device": args.device,
        "recent_window": args.recent_window,
        "weight_device": args.weight_device,
        "cpu_embed": bool(args.cpu_embed),
        "runtime_profile": args.runtime_profile,
        "vllm_custom_model": args.vllm_custom_model,
        "enable_mtp": bool(args.enable_mtp),
        "mtp_speculative_tokens": args.mtp_speculative_tokens,
    }
    return EngineModelSpec(
        model=model,
        served_model_name=args.served_model_name or args.model_name,
        tokenizer=args.tokenizer,
        dtype=str(os.environ.get("LANGBURST_DTYPE", "auto")),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len or args.recent_window,
        quantization=args.quantization,
        trust_remote_code=not args.no_trust_remote_code,
        features=_engine_feature_request_from_args(args),
        extra={k: v for k, v in extra.items() if v is not None},
    )


def main() -> None:
    ensure_engines_loaded()
    ap = argparse.ArgumentParser(description="OpenAI-compatible LangBurst engine router")
    ap.add_argument("--engine", choices=engine_registry.ids(), default=engine_registry.default_engine_id(), help="serving engine provider; vllm is the default")
    ap.add_argument("--model", default=None, help="engine model path/name; defaults to --hf-model")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--qwen36-lowbit", action="store_true")
    ap.add_argument("--recurrent-state", action="store_true")
    ap.add_argument("--vllm-custom-model", default=None)
    ap.add_argument("--enable-mtp", action="store_true")
    ap.add_argument("--mtp-speculative-tokens", type=int, default=2)
    ap.add_argument("--no-trust-remote-code", action="store_true")
    add_model_path_args(ap, required=False)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recent-window", type=int, default=serving_recent_window_default())
    ap.add_argument("--weight-device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--cpu-embed", action="store_true")
    add_adapter_arg(ap)
    ap.add_argument("--max-loaded-models", type=int, default=1)
    ap.add_argument("--max-active-requests", type=int, default=1)
    ap.add_argument("--max-queued-requests", type=int, default=0)
    ap.add_argument("--admission-timeout-s", type=float, default=None)
    ap.add_argument("--reserve-free-vram-mib", type=int, default=DEFAULT_RESERVE_FREE_VRAM_MIB)
    ap.add_argument("--max-state-pool-size", type=int, default=DEFAULT_MAX_STATE_POOL_SIZE)
    ap.add_argument("--max-prompt-tokens", type=int, default=max_prompt_tokens_default())
    ap.add_argument("--max-generation-tokens", type=int, default=DEFAULT_MAX_GENERATION_TOKENS)
    ap.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_BATCHED_TOKENS)
    ap.add_argument("--batch-prefill-chunk-size", type=int, default=DEFAULT_PREFILL_CHUNK_SIZE)
    ap.add_argument("--kv-block-size", type=int, default=kv_block_size_default())
    ap.add_argument("--kv-blocks", type=int, default=kv_blocks_default())
    ap.add_argument("--max-sessions", type=int, default=int(os.environ.get("LANGBURST_MAX_SESSIONS", "16")))
    ap.add_argument("--session-ttl-s", type=float, default=float(os.environ.get("LANGBURST_SESSION_TTL_S", "3600")))
    ap.add_argument("--models-json", type=Path, default=None)
    add_runtime_feature_args(ap)
    args = ap.parse_args()

    try:
        import uvicorn

        if args.engine == "native":
            features = runtime_features_from_args(args)
            if args.models_json:
                specs = load_model_specs(args.models_json, features)
            else:
                if args.hf_model is None or args.qb_model is None:
                    ap.error("--hf-model and --qb-model are required for --engine native unless --models-json is used")
                specs = [ModelResourceSpec.from_args(args, features=features)]
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
                    max_sessions=args.max_sessions,
                    session_ttl_s=None if args.session_ttl_s <= 0 else args.session_ttl_s,
                ),
            )
            app = create_app(manager)
        else:
            backend = engine_registry.create(_model_spec_from_args(args), engine_id=args.engine)
            app = create_engine_app(backend)
    except ValueError as exc:
        ap.error(str(exc))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
