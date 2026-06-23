from __future__ import annotations

import json
from typing import Any, Mapping, Sequence


def content_to_text(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(p for p in parts if p)


def normalize_for_chat_template(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize OpenAI-compatible roles for HF chat templates.

    OpenAI-compatible APIs may send a `developer` role. Most HF chat templates,
    including Qwen-family templates, accept `system` but not `developer`. They
    also commonly expect at most one leading system message. Keep user and
    assistant turns as conversation data, and merge only instruction roles.
    """

    cleaned: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user") or "user").lower()
        normalized: dict[str, Any] = {"role": role, "content": content_to_text(message.get("content", ""))}
        if role == "assistant":
            tool_calls = _normalize_assistant_tool_calls(message.get("tool_calls"))
            if tool_calls:
                normalized["tool_calls"] = tool_calls
        cleaned.append(normalized)

    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    for message in cleaned:
        role = message["role"]
        content = message["content"]
        if role in {"developer", "system"}:
            if content:
                system_parts.append(content)
            continue
        out.append(dict(message))
    if system_parts:
        return [{"role": "system", "content": "\n\n".join(system_parts)}, *out]
    return out


def _normalize_assistant_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        function = item.get("function") if isinstance(item.get("function"), Mapping) else item
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except Exception:
                parsed = {}
            arguments = parsed if isinstance(parsed, Mapping) else {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        out.append({"name": name, "arguments": {str(key): value for key, value in arguments.items()}})
    return out
