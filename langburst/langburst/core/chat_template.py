from __future__ import annotations

from typing import Any, Mapping


def resolve_chat_template_kwargs(
    request: Any | None = None,
    *,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single source of truth for chat-template kwargs.

    `reasoning_effort` is an OpenAI-compatible request hint, not a portable
    signal that a visible-thinking HF chat template should emit `<think>`
    spans. Visible thinking is therefore controlled by explicit chat-template
    kwargs only.
    """
    out: dict[str, Any] = {}
    if base:
        out.update(dict(base))
    request_kwargs = getattr(request, "chat_template_kwargs", None)
    if isinstance(request_kwargs, Mapping):
        out.update(dict(request_kwargs))
    out.setdefault("enable_thinking", False)
    reasoning_effort = getattr(request, "reasoning_effort", None)
    if reasoning_effort == "none":
        out["enable_thinking"] = False
    out.setdefault("preserve_thinking", bool(out["enable_thinking"]))
    return out
