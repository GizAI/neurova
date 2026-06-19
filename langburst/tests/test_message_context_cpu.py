from __future__ import annotations

import pytest

from langburst.core.message_context import MessageContextPolicy, select_messages_for_generation


def _encoder(messages):
    return [0] * sum(len(str(message.get("content", "")).split()) for message in messages)


def test_auto_long_user_preserves_short_conversation():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "tell me more"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(standalone_user_min_tokens=10),
    )

    assert selected == messages


def test_auto_long_user_keeps_system_and_latest_large_user_only():
    messages = [
        {"role": "system", "content": "answer latest"},
        {"role": "user", "content": "introduce yourself"},
        {"role": "assistant", "content": "I am Qwen"},
        {"role": "user", "content": " ".join(["document"] * 20) + "\nquestion?"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(standalone_user_min_tokens=10),
    )

    assert selected == [messages[0], messages[-1]]


def test_auto_long_user_anchors_short_followup_to_recent_large_user():
    messages = [
        {"role": "system", "content": "answer latest"},
        {"role": "user", "content": "introduce yourself"},
        {"role": "assistant", "content": "I am Qwen"},
        {"role": "user", "content": " ".join(["document"] * 20) + "\nquestion?"},
        {"role": "assistant", "content": "corrupted unrelated answer"},
        {"role": "user", "content": "answer the previous document question in one sentence"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(standalone_user_min_tokens=10),
    )

    assert selected == [messages[0], messages[3], messages[-1]]


def test_raw_metadata_override_preserves_client_transcript():
    messages = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": " ".join(["document"] * 20)},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(standalone_user_min_tokens=10),
        metadata={"langburst_message_context": "raw"},
    )

    assert selected == messages


def test_latest_user_mode_is_explicit():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(mode="latest_user"),
    )

    assert selected == [messages[0], messages[-1]]


def test_unknown_context_policy_fails_closed():
    with pytest.raises(ValueError, match="MESSAGE_CONTEXT_POLICY"):
        select_messages_for_generation(
            [{"role": "user", "content": "hello"}],
            encoder=_encoder,
            policy=MessageContextPolicy(mode="mystery"),
        )
