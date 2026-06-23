from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def normalize_tools_for_chat_template(tools: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Normalize OpenAI-compatible function tools for Qwen HF chat templates."""
    if not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = dict(tool["function"])
        else:
            function = dict(tool)
        name = str(function.get("name", "")).strip()
        if not name:
            continue
        normalized: dict[str, Any] = {"name": name}
        if function.get("description") is not None:
            normalized["description"] = str(function.get("description", ""))
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            normalized["parameters"] = parameters
        else:
            normalized["parameters"] = {"type": "object", "properties": {}}
        out.append(normalized)
    return out or None


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FENCED_SHELL_RE = re.compile(r"```(?:bash|sh|shell)\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FENCED_NAMED_TOOL_RE = re.compile(
    r"```([A-Za-z_][A-Za-z0-9_.-]*)\s*\n(.*?)(?:\n```|\Z)",
    re.DOTALL,
)
_INVOKE_RE = re.compile(r"<invoke\s+name=[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</invoke>", re.DOTALL | re.IGNORECASE)
_INVOKE_PARAM_RE = re.compile(r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
_TOOL_USE_RE = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL | re.IGNORECASE)
_TOOL_NAME_RE = re.compile(r"<tool_(?:uses|name)>\s*([^<]+?)\s*</tool_(?:uses|name)>", re.DOTALL | re.IGNORECASE)
_TOOL_INPUT_RE = re.compile(r"<tool_input>\s*(.*?)\s*</tool_input>", re.DOTALL | re.IGNORECASE)


def parse_qwen_tool_calls(
    text: str,
    *,
    tools: Sequence[dict[str, Any]] | None = None,
) -> tuple[str, list[ToolCall]]:
    """Parse Qwen's native XML tool-call format into OpenAI tool calls."""
    calls: list[ToolCall] = []
    content_parts: list[str] = []
    cursor = 0
    for match in _TOOL_CALL_RE.finditer(text):
        before = text[cursor : match.start()].strip()
        if before:
            content_parts.append(before)
        cursor = match.end()
        body = match.group(1)
        parsed = _function_body(body)
        if parsed is None:
            continue
        name, parameters_body = parsed
        args: dict[str, Any] = {}
        for parameter in _PARAMETER_RE.finditer(parameters_body):
            key = parameter.group(1).strip()
            value = parameter.group(2).strip()
            if key:
                args[key] = _parse_parameter_value(value)
        calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                name=name,
                arguments=json.dumps(args, ensure_ascii=False, separators=(",", ":")),
            )
        )
    tail = text[cursor:].strip()
    if tail:
        content_parts.append(tail)
    if not calls:
        json_calls = _parse_json_tool_calls(text, tools=tools)
        if json_calls:
            return "", json_calls
        fenced_tool_calls = _parse_fenced_named_tool_calls(text, tools=tools)
        if fenced_tool_calls:
            return "", fenced_tool_calls
        shell_calls = _parse_shell_tool_calls(text, tools=tools)
        if shell_calls:
            return "", shell_calls
        tagged_calls = _parse_tagged_json_tool_calls(text, tools=tools)
        if tagged_calls:
            return "", tagged_calls
        invoke_calls = _parse_invoke_tool_calls(text, tools=tools)
        if invoke_calls:
            return "", invoke_calls
        tool_use_calls = _parse_tool_use_calls(text, tools=tools)
        if tool_use_calls:
            return "", tool_use_calls
        named_input_calls = _parse_named_tool_input_calls(text, tools=tools)
        if named_input_calls:
            return "", named_input_calls
        return text, []
    return "\n\n".join(content_parts), calls


def openai_tool_calls(calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": call.arguments},
        }
        for call in calls
    ]


def _parse_parameter_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{\"-" or stripped in {"true", "false", "null"} or stripped[0].isdigit():
        try:
            return json.loads(stripped)
        except Exception:
            return value
    return value


def _function_body(body: str) -> tuple[str, str] | None:
    function = _FUNCTION_RE.search(body)
    if function:
        return function.group(1).strip(), function.group(2)

    lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
    if not lines:
        return None
    name = lines[0]
    if name.startswith("<") or name.endswith(">"):
        return None
    parameters_body = "\n".join(lines[1:])
    if not _PARAMETER_RE.search(parameters_body):
        return None
    return name, parameters_body


def _parse_json_tool_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    if not specs:
        return []
    candidates = _json_candidates(text)

    out: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            call = _json_item_to_tool_call(item, specs)
            if call is None:
                continue
            key = (call.name, call.arguments)
            if key in seen:
                continue
            seen.add(key)
            out.append(call)
    return out


def _json_candidates(text: str) -> list[str]:
    candidates = [match.group(1) for match in _FENCED_JSON_RE.finditer(text)]
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        candidates.append(stripped)
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except Exception:
            continue
        candidates.append(text[index : index + end])
    return candidates


def _parse_shell_tool_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    exec_spec = specs.get("exec_command")
    if not exec_spec or "cmd" not in exec_spec["properties"]:
        return []
    blocks = [match.group(1).strip() for match in _FENCED_SHELL_RE.finditer(text)]
    out: list[ToolCall] = []
    seen: set[str] = set()
    for block in blocks:
        if not block or block in seen:
            continue
        seen.add(block)
        out.append(_tool_call_from_args("exec_command", {"cmd": block}))
    return out


def _parse_fenced_named_tool_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    if not specs:
        return []
    out: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for match in _FENCED_NAMED_TOOL_RE.finditer(text):
        name = match.group(1).strip()
        spec = specs.get(name)
        if not spec or "cmd" not in spec["properties"]:
            continue
        command = match.group(2).strip()
        if not command:
            continue
        call = _tool_call_from_args(name, {"cmd": command})
        key = (call.name, call.arguments)
        if key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def _parse_tagged_json_tool_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    if not specs:
        return []
    out: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for name in specs:
        pattern = re.compile(r"<" + re.escape(name) + r">\s*(\{.*?\})(?=\s*(?:<|$))", re.DOTALL)
        for match in pattern.finditer(text):
            try:
                args = json.loads(match.group(1))
            except Exception:
                continue
            if not isinstance(args, dict):
                continue
            call = _tool_call_from_args(name, {str(k): v for k, v in args.items()})
            key = (call.name, call.arguments)
            if key in seen:
                continue
            seen.add(key)
            out.append(call)
    return out


def _parse_invoke_tool_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    if not specs:
        return []
    out: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for match in _INVOKE_RE.finditer(text):
        name = match.group(1).strip()
        if name not in specs:
            continue
        args = {
            parameter.group(1).strip(): parameter.group(2).strip()
            for parameter in _INVOKE_PARAM_RE.finditer(match.group(2))
            if parameter.group(1).strip()
        }
        if not args:
            continue
        call = _tool_call_from_args(name, args)
        key = (call.name, call.arguments)
        if key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def _parse_tool_use_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    if not specs:
        return []
    out: list[ToolCall] = []
    seen: set[tuple[str, str]] = set()
    for match in _TOOL_USE_RE.finditer(text):
        try:
            item = json.loads(match.group(1))
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if name not in specs:
            continue
        raw_args = item.get("arguments", {})
        if not isinstance(raw_args, dict):
            continue
        call = _tool_call_from_args(name, {str(k): v for k, v in raw_args.items()})
        key = (call.name, call.arguments)
        if key in seen:
            continue
        seen.add(key)
        out.append(call)
    return out


def _parse_named_tool_input_calls(text: str, *, tools: Sequence[dict[str, Any]] | None) -> list[ToolCall]:
    specs = _tool_specs(tools)
    if not specs:
        return []
    name_match = _TOOL_NAME_RE.search(text)
    input_match = _TOOL_INPUT_RE.search(text)
    if not name_match or not input_match:
        return []
    name = name_match.group(1).strip()
    spec = specs.get(name)
    if not spec or "cmd" not in spec["properties"]:
        return []
    command = input_match.group(1).strip()
    if not command:
        return []
    return [_tool_call_from_args(name, {"cmd": command})]


def _tool_specs(tools: Sequence[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for tool in normalize_tools_for_chat_template(tools) or []:
        parameters = tool.get("parameters") if isinstance(tool.get("parameters"), dict) else {}
        required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
        properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
        specs[str(tool["name"])] = {
            "required": {str(key) for key in required},
            "properties": {str(key) for key in properties},
        }
    return specs


def _json_item_to_tool_call(item: Any, specs: dict[str, dict[str, Any]]) -> ToolCall | None:
    if not isinstance(item, dict):
        return None
    raw_name = item.get("name") or item.get("tool") or item.get("function")
    args = {str(k): v for k, v in item.items() if k not in {"name", "tool", "function"}}
    if isinstance(item.get("arguments"), dict):
        args = {str(k): v for k, v in item["arguments"].items()}
    if isinstance(raw_name, dict):
        raw_name = raw_name.get("name")
        if isinstance(item.get("function"), dict) and isinstance(item["function"].get("arguments"), dict):
            args = {str(k): v for k, v in item["function"]["arguments"].items()}
    name = str(raw_name).strip() if raw_name else ""
    if name:
        if name not in specs:
            return None
        return _tool_call_from_args(name, args)

    matches = []
    keys = set(args)
    for candidate_name, spec in specs.items():
        required = spec["required"]
        properties = spec["properties"]
        if required and not required.issubset(keys):
            continue
        if keys and (not properties or not keys.issubset(properties)):
            continue
        if not keys and required:
            continue
        matches.append(candidate_name)
    if len(matches) != 1:
        return None
    return _tool_call_from_args(matches[0], args)


def _tool_call_from_args(name: str, args: dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=json.dumps(args, ensure_ascii=False, separators=(",", ":")),
    )
