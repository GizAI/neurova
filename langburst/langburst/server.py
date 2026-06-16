from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel
from starlette.requests import Request

from .cli_features import add_adapter_arg, add_runtime_feature_args, runtime_features_from_args
from .core.defaults import (
    DEFAULT_MAX_GENERATION_TOKENS,
    DEFAULT_MAX_BATCHED_TOKENS,
    DEFAULT_MAX_STATE_POOL_SIZE,
    DEFAULT_PREFILL_CHUNK_SIZE,
    DEFAULT_RESERVE_FREE_VRAM_MIB,
    kv_block_size_default,
    kv_blocks_default,
    max_prompt_tokens_default,
    serving_recent_window_default,
)
from .core.features import RuntimeFeatureOverride, RuntimeFeatures
from .core.manager import EngineManager, EngineResourcePolicy, ModelResourceSpec, load_model_specs
from .core.platform import PLATFORM_NAME
from .core.runtime import GenerationConfig, RuntimeEngine
from .core.text_stream import StreamingTextDecoder
from .core.usage import RequestUsage


class ReasoningOptions(BaseModel):
    effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None


class TextOptions(BaseModel):
    verbosity: Literal["low", "medium", "high"] | None = None


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
    session_id: str | None = None
    stateful_session: bool = False
    reset_session: bool = False
    previous_response_id: str | None = None
    prompt_cache_key: str | None = None
    prompt_cache_retention: Literal["in_memory", "ephemeral"] | None = None
    reasoning: ReasoningOptions | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    text: TextOptions | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
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


@dataclass(frozen=True)
class CompletionContext:
    engine: RuntimeEngine
    prompt_ids: list[int]
    generation_tokens: int
    features: RuntimeFeatures
    gen_cfg: GenerationConfig
    completion_id: str
    created: int
    session_id: str | None = None


@dataclass(frozen=True)
class CompletionResult:
    ids: list[int]
    usage: RequestUsage
    finish_reason: str = "stop"


_STREAM_STOP = object()


def _requested_generation_tokens(req: ChatCompletionRequest) -> int:
    return req.max_new_tokens or req.max_completion_tokens or req.max_tokens or 256


def _requested_min_generation_tokens(req: ChatCompletionRequest) -> int:
    return int(req.min_new_tokens if req.min_new_tokens is not None else req.min_tokens or 0)


def _request_messages(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    return [{"role": m.role, "content": m.content} for m in req.messages]


def _request_features(engine: RuntimeEngine, req: ChatCompletionRequest) -> RuntimeFeatures:
    features = RuntimeFeatures.from_profile(req.runtime_profile) if req.runtime_profile else engine.features
    return features.with_overrides(RuntimeFeatureOverride.from_obj(req))


def _normalize_logit_bias(raw: dict[str, float] | dict[int, float] | None) -> dict[int, float]:
    if not raw:
        return {}
    out: dict[int, float] = {}
    for key, value in raw.items():
        out[int(key)] = float(value)
    return out


def _flatten_token_ids(raw: list[int] | list[list[int]] | None) -> tuple[int, ...]:
    if not raw:
        return ()
    out: list[int] = []
    for item in raw:
        if isinstance(item, list):
            out.extend(int(t) for t in item)
        else:
            out.append(int(item))
    return tuple(out)


def _stop_strings(req: ChatCompletionRequest) -> list[str]:
    out: list[str] = []
    if isinstance(req.stop, str):
        out.append(req.stop)
    elif isinstance(req.stop, list):
        out.extend(str(v) for v in req.stop)
    if req.stop_sequences:
        out.extend(str(v) for v in req.stop_sequences)
    return [s for s in out if s]


def _stop_sequence_ids(engine: RuntimeEngine, req: ChatCompletionRequest) -> tuple[tuple[int, ...], ...]:
    seqs: list[tuple[int, ...]] = []
    for stop in _stop_strings(req):
        token_ids = engine.tokenizer.encode(stop)
        if hasattr(token_ids, "ids"):
            token_ids = token_ids.ids
        if token_ids:
            seqs.append(tuple(int(t) for t in token_ids))
    return tuple(seqs)


def _validate_request_shape(req: ChatCompletionRequest) -> None:
    if req.n != 1:
        raise ValueError("LangBurst currently supports n=1")
    if req.logprobs:
        raise ValueError("logprobs are not implemented for this runtime path")
    if req.top_logprobs:
        raise ValueError("top_logprobs are not implemented for this runtime path")
    if req.response_format:
        kind = req.response_format.get("type")
        if kind not in (None, "text"):
            raise ValueError("structured response_format is not implemented for this runtime path")
    if req.tools or req.tool_choice or req.parallel_tool_calls:
        raise ValueError("tool calling is not implemented for this runtime path")


def _sse_payload(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _record_oom(manager: EngineManager, engine: RuntimeEngine, exc: BaseException) -> dict[str, Any]:
    manager.mark_runtime_error(engine.model_name, exc)
    recovered = manager.recover_runtime_oom(engine.model_name)
    return {
        "message": "GPU memory pressure during generation; runtime was recovered. Retry with a smaller context window or wait for the current request to finish.",
        "type": "gpu_memory_pressure",
        "model": engine.model_name,
        "recovered_runtime": recovered,
    }


def _completion_context(manager: EngineManager, req: ChatCompletionRequest) -> CompletionContext:
    _validate_request_shape(req)
    engine = manager.get(req.model)
    prompt_ids = engine.encode_messages(_request_messages(req))
    generation_tokens = _requested_generation_tokens(req)
    manager.validate_generation_request(
        prompt_tokens=len(prompt_ids),
        generation_tokens=generation_tokens,
    )
    features = _request_features(engine, req)
    session_id = req.session_id
    if req.reset_session and session_id:
        manager.delete_session(session_id, model_name=engine.model_name)
    if req.stateful_session and not session_id:
        session_id = manager.create_session()
    return CompletionContext(
        engine=engine,
        prompt_ids=prompt_ids,
        generation_tokens=generation_tokens,
        features=features,
        gen_cfg=GenerationConfig(
            max_new_tokens=generation_tokens,
            min_new_tokens=_requested_min_generation_tokens(req),
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=1.0 if req.top_p is None else float(req.top_p),
            min_p=0.0 if req.min_p is None else float(req.min_p),
            repetition_penalty=1.0 if req.repetition_penalty is None else float(req.repetition_penalty),
            presence_penalty=float(req.presence_penalty),
            frequency_penalty=float(req.frequency_penalty),
            no_repeat_ngram_size=int(req.no_repeat_ngram_size),
            logit_bias=_normalize_logit_bias(req.logit_bias),
            bad_token_ids=_flatten_token_ids(req.bad_words_ids),
            suppress_tokens=tuple(int(t) for t in (req.suppress_tokens or [])) + tuple(int(t) for t in (req.begin_suppress_tokens or [])),
            seed=req.seed,
            eos_token_ids=engine.eos_token_ids(),
            stop_token_ids=tuple(int(t) for t in (req.stop_token_ids or [])),
            ignore_eos=bool(req.ignore_eos),
        ),
        completion_id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        session_id=session_id,
    )


def _uses_session(req: ChatCompletionRequest, ctx: CompletionContext) -> bool:
    return bool(req.stateful_session or ctx.session_id)


def _submit_completion(manager: EngineManager, req: ChatCompletionRequest, ctx: CompletionContext):
    worker = manager.create_batch_worker(req.model, ctx.features)
    session_record = None
    if _uses_session(req, ctx):
        session_record = manager.get_session(session_id=ctx.session_id or manager.create_session(), model_name=ctx.engine.model_name, features=ctx.features)
    handle = worker.submit(
        ctx.prompt_ids,
        max_new_tokens=ctx.generation_tokens,
        eos_token_ids=ctx.engine.eos_token_ids(),
        generation_config=ctx.gen_cfg,
        prompt_cache_key=req.prompt_cache_key,
        session_record=session_record,
        stop_sequences=_stop_sequence_ids(ctx.engine, req),
        include_stop_str_in_output=req.include_stop_str_in_output,
        request_id=ctx.completion_id,
    )
    return handle


def _serve_with_batch() -> bool:
    return os.environ.get("LANGBURST_SERVE_BATCH", "1").strip().lower() in {"1", "true", "yes", "on"}


def _finish_reason(ids: list[int], ctx: CompletionContext) -> str:
    return "length" if len(ids) >= ctx.generation_tokens else "stop"


def _trim_stop_sequences(ids: list[int], stop_sequences: tuple[tuple[int, ...], ...], *, include_stop: bool) -> list[int]:
    if not ids or not stop_sequences:
        return ids
    for seq in stop_sequences:
        if seq and len(ids) >= len(seq) and tuple(ids[-len(seq) :]) == tuple(seq):
            return ids if include_stop else ids[: -len(seq)]
    return ids


def _plain_completion_ids(manager: EngineManager, req: ChatCompletionRequest, ctx: CompletionContext) -> CompletionResult:
    usage = RequestUsage(prompt_tokens=len(ctx.prompt_ids))
    stop_sequences = _stop_sequence_ids(ctx.engine, req)
    with manager.acquire_request():
        session_record = None
        if _uses_session(req, ctx):
            session_record = manager.get_session(
                session_id=ctx.session_id or manager.create_session(),
                model_name=ctx.engine.model_name,
                features=ctx.features,
            )
            session_record.lock.acquire()
        try:
            with ctx.engine.lock, torch.no_grad():
                if session_record is not None:
                    ids = ctx.engine.generate_ids_greedy_gpu(
                        ctx.prompt_ids,
                        ctx.gen_cfg,
                        state=session_record.state,
                        features=ctx.features,
                    )
                    if ids:
                        ctx.engine.forward_one(int(ids[-1]), session_record.state, return_logits=False)
                        session_record.prompt_tokens += len(ctx.prompt_ids)
                        session_record.generated_tokens += len(ids)
                        session_record.turns += 1
                        session_record.touch()
                else:
                    with ctx.engine.pooled_state(ctx.features) as state:
                        ids = ctx.engine.generate_ids_greedy_gpu(
                            ctx.prompt_ids,
                            ctx.gen_cfg,
                            state=state,
                            features=ctx.features,
                        )
        finally:
            if session_record is not None:
                session_record.lock.release()
    ids = _trim_stop_sequences(ids, stop_sequences, include_stop=req.include_stop_str_in_output)
    usage.finish_now(completion_tokens=len(ids))
    return CompletionResult(ids=ids, usage=usage, finish_reason=_finish_reason(ids, ctx))


class PlainStreamHandle:
    def __init__(self) -> None:
        self.queue: queue.Queue[object] = queue.Queue()
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.error: BaseException | None = None
        self.ids: list[int] = []
        self.usage: RequestUsage | None = None
        self.finish_reason = "stop"

    def poll_output(self, timeout: float = 0.25) -> tuple[int | None, bool]:
        try:
            item = self.queue.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            return None, False
        if item is _STREAM_STOP:
            if self.error is not None:
                raise self.error
            return None, True
        return int(item), False

    def cancel(self) -> None:
        self.cancelled.set()


def _start_plain_stream(manager: EngineManager, req: ChatCompletionRequest, ctx: CompletionContext) -> PlainStreamHandle:
    handle = PlainStreamHandle()

    def run() -> None:
        usage = RequestUsage(prompt_tokens=len(ctx.prompt_ids))
        emitted: list[int] = []
        try:
            with manager.acquire_request():
                session_record = None
                if _uses_session(req, ctx):
                    session_record = manager.get_session(
                        session_id=ctx.session_id or manager.create_session(),
                        model_name=ctx.engine.model_name,
                        features=ctx.features,
                    )
                    session_record.lock.acquire()
                try:
                    with ctx.engine.lock, torch.no_grad():
                        if session_record is not None:
                            state = session_record.state
                            token_iter = ctx.engine.generate_ids(ctx.prompt_ids, ctx.gen_cfg, state=state, features=ctx.features)
                            for token_id in token_iter:
                                if handle.cancelled.is_set():
                                    break
                                emitted.append(int(token_id))
                                handle.queue.put(int(token_id))
                            if emitted:
                                ctx.engine.forward_one(int(emitted[-1]), state, return_logits=False)
                                session_record.prompt_tokens += len(ctx.prompt_ids)
                                session_record.generated_tokens += len(emitted)
                                session_record.turns += 1
                                session_record.touch()
                        else:
                            with ctx.engine.pooled_state(ctx.features) as state:
                                token_iter = ctx.engine.generate_ids(ctx.prompt_ids, ctx.gen_cfg, state=state, features=ctx.features)
                                for token_id in token_iter:
                                    if handle.cancelled.is_set():
                                        break
                                    emitted.append(int(token_id))
                                    handle.queue.put(int(token_id))
                finally:
                    if session_record is not None:
                        session_record.lock.release()
            emitted = _trim_stop_sequences(emitted, _stop_sequence_ids(ctx.engine, req), include_stop=req.include_stop_str_in_output)
            usage.finish_now(completion_tokens=len(emitted))
            handle.ids = emitted
            handle.usage = usage
            handle.finish_reason = _finish_reason(emitted, ctx)
        except BaseException as exc:
            handle.error = exc
        finally:
            handle.done.set()
            handle.queue.put(_STREAM_STOP)

    threading.Thread(target=run, name="langburst-plain-stream", daemon=True).start()
    return handle


def _completion_ids(manager: EngineManager, req: ChatCompletionRequest, ctx: CompletionContext) -> CompletionResult:
    if not _serve_with_batch():
        return _plain_completion_ids(manager, req, ctx)
    handle = _submit_completion(manager, req, ctx)
    ids = handle.wait_ids()
    usage = RequestUsage(prompt_tokens=len(ctx.prompt_ids))
    usage.apply_metrics(handle.metrics())
    finish_reason = "length" if len(ids) >= ctx.generation_tokens else "stop"
    return CompletionResult(ids=ids, usage=usage, finish_reason=finish_reason)


def _usage_payload(usage: RequestUsage, *, include_performance: bool = True) -> dict[str, Any]:
    payload = usage.openai_usage()
    if include_performance:
        payload["performance"] = usage.performance()
    return payload


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

    @app.post("/v1/langburst/sessions")
    def create_session(req: SessionCreateRequest | None = None):
        model_name = req.model if req is not None else None
        if model_name is not None:
            try:
                manager.resolve_plan(model_name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"id": manager.create_session(), "object": "langburst.session", "model": model_name}

    @app.get("/v1/langburst/sessions")
    def list_sessions():
        return manager.sessions.summary()

    @app.delete("/v1/langburst/sessions/{session_id}")
    def delete_session(session_id: str, model: str | None = None):
        return {"id": session_id, "deleted": manager.delete_session(session_id, model_name=model)}

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
    def chat_completions(request: Request, req: ChatCompletionRequest):
        try:
            ctx = _completion_context(manager, req)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            model_name = req.model or manager.default_model_name()
            manager.mark_runtime_error(model_name, exc)
            manager.recover_runtime_oom(model_name)
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "GPU memory pressure while loading the model; runtime was recovered. Retry with a smaller context window.",
                    "type": "gpu_memory_pressure",
                    "model": model_name,
                    "recovered_runtime": True,
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        if req.stream:
            async def events():
                handle = None
                usage = RequestUsage(prompt_tokens=len(ctx.prompt_ids))
                try:
                    decoder = StreamingTextDecoder(ctx.engine.tokenizer, skip_special_tokens=False)
                    handle = _submit_completion(manager, req, ctx) if _serve_with_batch() else _start_plain_stream(manager, req, ctx)
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
                        if not text:
                            continue
                        payload = {
                            "id": ctx.completion_id,
                            "object": "chat.completion.chunk",
                            "created": ctx.created,
                            "model": ctx.engine.model_name,
                            **({"session_id": ctx.session_id} if _uses_session(req, ctx) else {}),
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                        yield _sse_payload(payload)
                    if handle is not None:
                        usage.apply_metrics(handle.metrics())
                    text = decoder.flush()
                    if text:
                        payload = {
                            "id": ctx.completion_id,
                            "object": "chat.completion.chunk",
                            "created": ctx.created,
                            "model": ctx.engine.model_name,
                            **({"session_id": ctx.session_id} if _uses_session(req, ctx) else {}),
                            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                        }
                        yield _sse_payload(payload)
                except TimeoutError as exc:
                    yield _sse_payload({"error": {"message": str(exc), "type": "admission_error"}})
                except torch.cuda.OutOfMemoryError as exc:
                    yield _sse_payload({"error": _record_oom(manager, ctx.engine, exc)})
                except BaseException as exc:
                    print(
                        f"[langburst] streaming request {ctx.completion_id} failed: {type(exc).__name__}: {exc}\n"
                        f"{traceback.format_exc()}",
                        flush=True,
                    )
                    yield _sse_payload({"error": {"message": str(exc), "type": type(exc).__name__}})
                finally:
                    if handle is not None and not handle.done.is_set():
                        handle.cancel()
                include_usage = bool(req.stream_options and req.stream_options.include_usage)
                done = {
                    "id": ctx.completion_id,
                    "object": "chat.completion.chunk",
                    "created": ctx.created,
                    "model": ctx.engine.model_name,
                    **({"session_id": ctx.session_id} if _uses_session(req, ctx) else {}),
                    **({"usage": _usage_payload(usage)} if include_usage else {}),
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield _sse_payload(done)
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = _completion_ids(manager, req, ctx)
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(status_code=503, detail=_record_oom(manager, ctx.engine, exc)) from exc
        content = ctx.engine.tokenizer.decode(result.ids, skip_special_tokens=True)
        return JSONResponse(
            {
                "id": ctx.completion_id,
                "object": "chat.completion",
                "created": ctx.created,
                "model": ctx.engine.model_name,
                **({"session_id": ctx.session_id} if _uses_session(req, ctx) else {}),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": _usage_payload(result.usage),
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
    ap.add_argument("--recent-window", type=int, default=serving_recent_window_default())
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
    ap.add_argument("--max-prompt-tokens", type=int, default=max_prompt_tokens_default(), help="reject longer prompts before allocating runtime state")
    ap.add_argument("--max-generation-tokens", type=int, default=DEFAULT_MAX_GENERATION_TOKENS, help="reject longer generations before allocating runtime state")
    ap.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_BATCHED_TOKENS, help="continuous-batching token budget")
    ap.add_argument("--batch-prefill-chunk-size", type=int, default=DEFAULT_PREFILL_CHUNK_SIZE, help="continuous-batching prefill chunk size")
    ap.add_argument("--kv-block-size", type=int, default=kv_block_size_default(), help="paged-KV logical block size")
    ap.add_argument("--kv-blocks", type=int, default=kv_blocks_default(), help="paged-KV block table capacity")
    ap.add_argument("--max-sessions", type=int, default=int(os.environ.get("LANGBURST_MAX_SESSIONS", "16")), help="maximum explicit stateful sessions kept in memory")
    ap.add_argument("--session-ttl-s", type=float, default=float(os.environ.get("LANGBURST_SESSION_TTL_S", "3600")), help="idle stateful session TTL in seconds; use 0 to disable TTL")
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
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
