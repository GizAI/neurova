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


def test_agent_context_promotes_prelude_user_messages_to_developer_context():
    messages = [
        {"role": "developer", "content": "follow repository rules"},
        {"role": "user", "content": "# AGENTS.md instructions\nkeep changes clean"},
        {"role": "user", "content": "build a budget app"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(),
        metadata={"langburst_agent_context": True},
    )

    assert selected == [
        messages[0],
        {"role": "developer", "content": "# AGENTS.md instructions\nkeep changes clean"},
        messages[-1],
    ]


def test_agent_context_does_not_rewrite_visible_conversation_history():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "continue"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(),
        metadata={"langburst_agent_context": True},
    )

    assert selected == messages


def test_agent_plain_text_context_keeps_latest_user_only_before_first_assistant():
    messages = [
        {"role": "developer", "content": "large tool-agent instructions"},
        {"role": "user", "content": "# AGENTS.md instructions\nrepo context"},
        {"role": "user", "content": "build a budget app"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(),
        metadata={"langburst_agent_context": "plain_text"},
    )

    assert selected == [messages[-1]]


def test_agent_plain_text_context_keeps_current_tool_loop_tail():
    messages = [
        {"role": "system", "content": "large agent instructions"},
        {"role": "user", "content": "environment context"},
        {"role": "user", "content": "create a file"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec_command", "arguments": '{"cmd":"cat file"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
    ]

    selected = select_messages_for_generation(
        messages,
        encoder=_encoder,
        policy=MessageContextPolicy(),
        metadata={"langburst_agent_context": "plain_text"},
    )

    assert selected == messages[2:]
