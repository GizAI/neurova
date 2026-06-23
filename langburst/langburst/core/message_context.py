from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


Message = dict[str, Any]
MessageEncoder = Callable[[Sequence[Message]], Sequence[int]]


@dataclass(frozen=True)
class MessageContextPolicy:
    """Owns the raw chat-history contract before tokenizer rendering.

    OpenAI-compatible clients usually send the complete visible transcript on
    every turn. That is correct for short conversational turns, but a long
    pasted document followed by a direct question is a standalone task: replaying
    unrelated old turns can dominate the next-token distribution and make the
    model continue an earlier topic. Once such a document turn exists, short
    follow-up questions should stay anchored to that document instead of
    re-injecting unrelated or previously corrupted assistant turns.
    """

    mode: str = "auto_long_user"
    standalone_user_min_tokens: int = 1024

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "MessageContextPolicy":
        return cls(
            mode=str(env.get("LANGBURST_MESSAGE_CONTEXT_POLICY", "auto_long_user") or "auto_long_user"),
            standalone_user_min_tokens=max(0, int(env.get("LANGBURST_STANDALONE_USER_MIN_TOKENS", "1024") or "0")),
        )


def metadata_context_policy(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("langburst_message_context") or metadata.get("message_context_policy")
    return str(value) if value is not None else None


def select_messages_for_generation(
    messages: Sequence[Message],
    *,
    encoder: MessageEncoder,
    policy: MessageContextPolicy,
    metadata: Mapping[str, Any] | None = None,
) -> list[Message]:
    """Return the messages that should be rendered for this generation.

    Modes:
      - raw: preserve the client transcript exactly.
      - latest_user: keep leading system messages and the latest user message.
      - auto_long_user: latest_user only when that latest user turn is large
        enough to be a pasted-document task.
    """

    mode = (metadata_context_policy(metadata) or policy.mode).strip().lower()
    normalized = [dict(message) for message in messages]
    if mode in {"", "raw", "none", "off"}:
        return normalized

    latest_user_index = _latest_user_index(normalized)
    if latest_user_index is None:
        return normalized

    agent_context = _agent_context_mode(metadata)
    if agent_context:
        agent_selected = _select_agent_context_messages(
            normalized,
            latest_user_index=latest_user_index,
            context_mode=agent_context,
        )
        if agent_selected is not None:
            return agent_selected

    if mode in {"latest", "latest_user", "standalone"}:
        return _leading_instruction_messages(normalized[:latest_user_index]) + [normalized[latest_user_index]]

    if mode not in {"auto", "auto_long_user", "long_user"}:
        raise ValueError(
            "LANGBURST_MESSAGE_CONTEXT_POLICY must be one of raw, auto_long_user, latest_user"
        )

    if policy.standalone_user_min_tokens <= 0:
        return normalized

    long_user_index = _last_standalone_user_index(
        normalized[: latest_user_index + 1],
        encoder=encoder,
        min_tokens=policy.standalone_user_min_tokens,
    )
    if long_user_index is None:
        return normalized

    leading = _leading_instruction_messages(normalized[:long_user_index])
    latest_user = normalized[latest_user_index]
    long_user = normalized[long_user_index]
    if long_user_index == latest_user_index:
        return leading + [latest_user]
    return leading + [long_user, latest_user]


def _latest_user_index(messages: Sequence[Message]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if str(messages[idx].get("role", "")).lower() == "user":
            return idx
    return None


def _agent_context_mode(metadata: Mapping[str, Any] | None) -> str:
    if not isinstance(metadata, Mapping):
        return ""
    value = metadata.get("langburst_agent_context") or metadata.get("agent_context")
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in {"", "0", "false", "off", "no"}:
            return ""
        return mode
    return "context" if value else ""


def _select_agent_context_messages(
    messages: Sequence[Message],
    *,
    latest_user_index: int,
    context_mode: str,
) -> list[Message] | None:
    """Render agent-client prelude as instruction context, not dialogue.

    OpenAI Responses-style agent clients can send an initial instruction block
    and environment/context block before the actual user task. When such a
    request is bridged to Chat Completions, those pre-task blocks often arrive
    as consecutive user-role messages. Chat templates interpret consecutive
    user turns as dialogue, which makes the model answer the context itself.

    Only apply this contract before any assistant turn exists. Once a visible
    assistant turn exists, the messages are ordinary conversation history.
    """

    prelude = list(messages[:latest_user_index])
    if not prelude:
        return None
    if context_mode in {"latest", "latest_user", "plain", "plain_text"}:
        tool_loop_start = _agent_tool_loop_start(messages, latest_user_index=latest_user_index)
        if tool_loop_start is not None:
            return [dict(message) for message in messages[tool_loop_start:]]
    if any(str(message.get("role", "")).lower() == "assistant" for message in prelude):
        return None
    if context_mode in {"latest", "latest_user", "plain", "plain_text"}:
        return [dict(messages[latest_user_index]), *_agent_tool_tail(messages[latest_user_index + 1 :])]

    leading = _leading_instruction_messages(prelude)
    context_parts: list[str] = []
    for message in prelude[len(leading):]:
        role = str(message.get("role", "")).lower()
        if role != "user":
            return None
        text = _message_text(message)
        if text:
            context_parts.append(text)
    if not context_parts:
        return None

    return [
        *leading,
        {"role": "developer", "content": "\n\n".join(context_parts)},
        dict(messages[latest_user_index]),
        *_agent_tool_tail(messages[latest_user_index + 1 :]),
    ]


def _agent_tool_tail(messages: Sequence[Message]) -> list[Message]:
    out: list[Message] = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        if role == "tool":
            out.append(dict(message))
            continue
        if role == "assistant" and (message.get("tool_calls") or _message_text(message)):
            out.append(dict(message))
            continue
        return []
    return out


def _agent_tool_loop_start(messages: Sequence[Message], *, latest_user_index: int) -> int | None:
    cursor = latest_user_index - 1
    saw_tool_loop = False
    while cursor >= 0:
        role = str(messages[cursor].get("role", "")).lower()
        if role in {"assistant", "tool"}:
            saw_tool_loop = True
            cursor -= 1
            continue
        break
    if saw_tool_loop and cursor >= 0 and str(messages[cursor].get("role", "")).lower() == "user":
        return cursor
    return None


def _leading_instruction_messages(messages: Sequence[Message]) -> list[Message]:
    out: list[Message] = []
    for message in messages:
        if _is_instruction_role(message):
            out.append(dict(message))
            continue
        break
    return out


def _is_instruction_role(message: Message) -> bool:
    return str(message.get("role", "")).lower() in {"system", "developer"}


def _message_text(message: Message) -> str:
    content = message.get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray, str)):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type", "text"))
            if item_type in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _last_standalone_user_index(
    messages: Sequence[Message],
    *,
    encoder: MessageEncoder,
    min_tokens: int,
) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if str(messages[idx].get("role", "")).lower() != "user":
            continue
        if len(encoder([messages[idx]])) >= min_tokens:
            return idx
    return None
