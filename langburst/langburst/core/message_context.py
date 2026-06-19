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

    if mode in {"latest", "latest_user", "standalone"}:
        return _leading_system_messages(normalized[:latest_user_index]) + [normalized[latest_user_index]]

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

    leading = _leading_system_messages(normalized[:long_user_index])
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


def _leading_system_messages(messages: Sequence[Message]) -> list[Message]:
    out: list[Message] = []
    for message in messages:
        if str(message.get("role", "")).lower() == "system":
            out.append(dict(message))
            continue
        break
    return out


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
