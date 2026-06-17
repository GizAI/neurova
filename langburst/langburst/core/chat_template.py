from __future__ import annotations

from typing import Any, Mapping


def resolve_chat_template_kwargs(
    request: Any | None = None,
    *,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Single source of truth for chat-template kwargs."""
    out: dict[str, Any] = {}
    if base:
        out.update(dict(base))
    request_kwargs = getattr(request, "chat_template_kwargs", None)
    if isinstance(request_kwargs, Mapping):
        filtered = dict(request_kwargs)
        filtered.pop("enable_thinking", None)
        out.update(filtered)
    out.setdefault("enable_thinking", False)
    reasoning_effort = getattr(request, "reasoning_effort", None)
    if reasoning_effort == "none":
        out["enable_thinking"] = False
    elif reasoning_effort is not None:
        out["enable_thinking"] = True
    return out
