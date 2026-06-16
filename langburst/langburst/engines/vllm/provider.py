from __future__ import annotations

from typing import Any, Iterable

from ..base import (
    EngineBackend,
    EngineCapabilities,
    EngineChatChunk,
    EngineChatRequest,
    EngineChatResult,
    EngineDescriptor,
    EngineModelSpec,
    EngineSamplingParams,
    EngineUsage,
    resolve_engine_feature_plan,
)
from .bridge import (
    VLLMConversationStore,
    build_vllm_bridge_config,
    resolve_lowbit_max_num_batched_tokens,
    vllm_engine_extra_kwargs,
)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        text = _content_to_text(msg.get("content", ""))
        lines.append(f"{role}: {text}")
    lines.append("assistant:")
    return "\n".join(lines)


def _sampling_kwargs(params: EngineSamplingParams) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_tokens": int(params.max_tokens),
        "min_tokens": int(params.min_tokens),
        "temperature": float(params.temperature),
        "top_p": float(params.top_p),
        "ignore_eos": bool(params.ignore_eos),
    }
    if params.top_k is not None and int(params.top_k) >= 0:
        kwargs["top_k"] = int(params.top_k)
    if params.stop:
        kwargs["stop"] = list(params.stop)
    if params.stop_token_ids:
        kwargs["stop_token_ids"] = list(params.stop_token_ids)
    if params.seed is not None:
        kwargs["seed"] = int(params.seed)
    kwargs.update(params.extra)
    return kwargs


def _chat_template_kwargs(request: EngineChatRequest, spec: EngineModelSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"enable_thinking": True}
    configured = spec.extra.get("chat_template_kwargs")
    if isinstance(configured, dict):
        out.update(configured)
    raw = request.raw_request
    request_kwargs = getattr(raw, "chat_template_kwargs", None) if raw is not None else None
    if isinstance(request_kwargs, dict):
        out.update(request_kwargs)
    reasoning_effort = getattr(raw, "reasoning_effort", None) if raw is not None else None
    if reasoning_effort == "none":
        out["enable_thinking"] = False
    elif reasoning_effort is not None:
        out["enable_thinking"] = True
    return out


class VLLMBackend:
    descriptor = EngineDescriptor(
        engine_id="vllm",
        display_name="vLLM",
        module="langburst.engines.vllm",
        capabilities=EngineCapabilities(
            continuous_batching=True,
            paged_kv=True,
            prefix_cache=True,
            structured_output=True,
            speculative_decoding=True,
            host_state=True,
            custom_model=True,
            quantization=("awq", "gptq", "fp8", "int8", "bitsandbytes", "gguf", "torchao", "langburst-lowbit"),
        ),
    )

    def __init__(self, spec: EngineModelSpec) -> None:
        self.spec = spec
        self.feature_plan = resolve_engine_feature_plan(self.descriptor.capabilities, spec.features)
        self.bridge = build_vllm_bridge_config(spec, self.feature_plan)
        self.sessions = VLLMConversationStore(
            max_sessions=int(spec.extra.get("max_sessions", 128)),
            max_messages_per_session=int(spec.extra.get("max_messages_per_session", 64)),
        )
        self._llm: Any | None = None

    def start(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except Exception as exc:  # pragma: no cover - depends on deployment env
            raise RuntimeError(
                "vLLM is the default LangBurst engine but is not installed. "
                "Install langburst with the vllm dependency or choose --engine native|sglang|exl3."
            ) from exc
        if self.spec.features.qwen36_lowbit:
            from .plugins import register as register_vllm_plugins

            register_vllm_plugins()
        kwargs: dict[str, Any] = {
            "model": self.spec.model,
            "trust_remote_code": self.spec.trust_remote_code,
            "tensor_parallel_size": self.spec.tensor_parallel_size,
        }
        kwargs.update(self.bridge.engine_kwargs)
        if self.spec.tokenizer:
            kwargs["tokenizer"] = self.spec.tokenizer
        if self.spec.dtype and self.spec.dtype != "auto":
            kwargs["dtype"] = self.spec.dtype
        elif self.spec.dtype and "dtype" not in kwargs:
            kwargs["dtype"] = self.spec.dtype
        if self.spec.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = self.spec.gpu_memory_utilization
        if self.spec.max_model_len is not None:
            kwargs["max_model_len"] = self.spec.max_model_len
        if self.spec.quantization:
            kwargs["quantization"] = self.spec.quantization
        kwargs.update(vllm_engine_extra_kwargs(self.spec.extra))
        if self.spec.features.qwen36_lowbit and self.spec.features.recurrent_state:
            enable_mtp = bool(self.spec.extra.get("enable_mtp", False))
            kwargs["max_num_batched_tokens"] = max(
                resolve_lowbit_max_num_batched_tokens(self.spec, enable_mtp=enable_mtp),
                int(kwargs.get("max_num_batched_tokens", 0) or 0),
            )
        self._llm = LLM(**kwargs)

    def shutdown(self) -> None:
        self._llm = None

    def list_models(self) -> list[dict[str, Any]]:
        return [{"id": self.spec.public_name, "object": "model", "owned_by": "langburst-vllm"}]

    def health(self) -> dict[str, Any]:
        return {
            "ok": self._llm is not None,
            "engine": self.descriptor.summary(),
            "model": self.spec.public_name,
            "feature_plan": self.feature_plan.summary(),
            "bridge": self.bridge.summary(),
            "sessions": self.sessions.summary(),
        }

    def generate_chat(self, request: EngineChatRequest) -> EngineChatResult:
        self.start()
        assert self._llm is not None
        try:
            from vllm import SamplingParams
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("vLLM SamplingParams is unavailable") from exc
        sampling = SamplingParams(**_sampling_kwargs(request.sampling))
        session_id, reset = _request_session(request)
        messages = self.sessions.resolve(session_id=session_id, messages=request.messages, reset=reset)
        if hasattr(self._llm, "chat"):
            outputs = self._llm.chat(
                messages,
                sampling_params=sampling,
                use_tqdm=False,
                chat_template_kwargs=_chat_template_kwargs(request, self.spec),
            )
        else:
            outputs = self._llm.generate([_messages_to_prompt(messages)], sampling_params=sampling, use_tqdm=False)
        first = outputs[0]
        text = first.outputs[0].text if getattr(first, "outputs", None) else ""
        finish_reason = getattr(first.outputs[0], "finish_reason", "stop") if getattr(first, "outputs", None) else "stop"
        usage = EngineUsage(
            prompt_tokens=len(getattr(first, "prompt_token_ids", []) or []),
            completion_tokens=len(getattr(first.outputs[0], "token_ids", []) or []) if getattr(first, "outputs", None) else 0,
        )
        self.sessions.resolve(session_id=session_id, messages=request.messages, assistant_text=text)
        return EngineChatResult(text=text, model=self.spec.public_name, finish_reason=str(finish_reason or "stop"), usage=usage, raw=first)

    def stream_chat(self, request: EngineChatRequest) -> Iterable[EngineChatChunk]:
        # The library-level synchronous LLM API does not expose the same token
        # stream contract as the vLLM HTTP server. Keep LangBurst's backend
        # surface stable by yielding a final chunk; online AsyncLLM support can
        # replace this provider internals without changing server code.
        result = self.generate_chat(request)
        yield EngineChatChunk(
            text=result.text,
            model=result.model,
            finish_reason=result.finish_reason,
            usage=result.usage,
            raw=result.raw,
        )


class VLLMProvider:
    descriptor = VLLMBackend.descriptor

    def create(self, spec: EngineModelSpec) -> EngineBackend:
        return VLLMBackend(spec)


def _request_session(request: EngineChatRequest) -> tuple[str | None, bool]:
    raw = request.raw_request
    if raw is None:
        return None, False
    session_id = getattr(raw, "session_id", None) or getattr(raw, "previous_response_id", None)
    stateful = bool(getattr(raw, "stateful_session", False))
    reset = bool(getattr(raw, "reset_session", False))
    return (str(session_id) if session_id else (request.request_id if stateful else None), reset)
