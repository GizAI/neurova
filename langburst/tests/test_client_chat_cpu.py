from __future__ import annotations

import json

from langburst import client_chat


def test_base_url_normalizes_v1_suffix():
    assert client_chat._base_url("http://host:8008") == "http://host:8008/v1"
    assert client_chat._base_url("http://host:8008/v1/") == "http://host:8008/v1"


def test_chat_payload_keeps_history():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
    ]

    payload = client_chat._chat_payload(
        model="m",
        messages=messages,
        text="next",
        max_tokens=None,
        temperature=0,
        top_p=None,
        reasoning_effort="none",
    )

    assert payload["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next"},
    ]
    assert payload["stream_options"] == {"include_usage": True}
    assert "metadata" not in payload
    assert "max_tokens" not in payload
    assert payload["reasoning_effort"] == "none"


def test_stateless_payload_keeps_history():
    messages = [{"role": "user", "content": "old"}]

    payload = client_chat._chat_payload(
        model="m",
        messages=messages,
        text="next",
        max_tokens=32,
        temperature=0,
        top_p=0.9,
        reasoning_effort="high",
    )

    assert payload["messages"] == [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "next"},
    ]
    assert payload["top_p"] == 0.9
    assert "enable_thinking" not in payload
    assert payload["reasoning_effort"] == "high"


def test_thinking_command_updates_state():
    effort, message = client_chat._set_thinking(" high ", reasoning_effort=None)
    assert effort == "high"
    assert message == "thinking: high"

    effort, message = client_chat._set_thinking("none", reasoning_effort=effort)
    assert effort == "none"
    assert message == "thinking: none"

    assert client_chat._thinking_status(effort) == "none"


def test_command_handler_uses_single_session_state():
    session = client_chat.ChatSession(system="sys", reasoning_effort=None)

    result = client_chat._handle_command("/thinking high", session)
    assert result.handled is True
    assert result.message == "thinking: high"
    assert session.reasoning_effort == "high"

    session.add_turn("u", "a")
    assert client_chat._handle_command("/history", session).message == "turns=1"

    result = client_chat._handle_command("/reset", session)
    assert result.message == "new conversation"
    assert session.messages == [{"role": "system", "content": "sys"}]

    session.add_turn("u2", "a2")
    result = client_chat._handle_command("/new", session)
    assert result.message == "new conversation"
    assert session.messages == [{"role": "system", "content": "sys"}]

    assert client_chat._handle_command("/exit", session).exit is True


def test_conversation_file_roundtrip(tmp_path):
    path = tmp_path / "conversation.json"
    session = client_chat.ChatSession(system="sys", reasoning_effort="low")
    session.add_turn("hello", "hi")

    client_chat._save_conversation(path, session)
    loaded = client_chat._load_conversation(path, system="sys", reasoning_effort=None)

    assert loaded.reasoning_effort == "low"
    assert loaded.messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_textual_app_can_be_constructed_when_dependency_available(tmp_path):
    if client_chat.App is None:
        return
    app = client_chat.LangBurstTextualApp(
        base_url="http://example/v1",
        model="m",
        max_tokens=None,
        temperature=0,
        top_p=None,
        timeout=1,
        system=None,
        reasoning_effort=None,
        conversation_file=tmp_path / "conversation.json",
        history_file=tmp_path / "history",
    )
    assert app.model == "m"


def test_textual_composer_key_contract():
    if client_chat.ChatComposer is None:
        return
    bindings = {binding.key: binding.action for binding in client_chat.ChatComposer.BINDINGS}
    assert bindings["enter"] == "submit"
    assert bindings["alt+enter"] == "newline"
    assert bindings["up"] == "history_previous"
    assert bindings["down"] == "history_next"


def test_textual_app_keeps_composer_as_auto_focus():
    if client_chat.App is None:
        return
    assert client_chat.LangBurstTextualApp.AUTO_FOCUS == "#composer"
    bindings = {key: action for key, action, *_rest in client_chat.LangBurstTextualApp.BINDINGS}
    assert bindings["ctrl+l"] == "focus_input"
    assert not hasattr(client_chat.LangBurstTextualApp, "_keep_focus_alive")
    assert client_chat.ChatOutput.can_focus is False


def test_textual_stream_updates_single_transcript(tmp_path):
    if client_chat.App is None:
        return

    async def run():
        app = client_chat.LangBurstTextualApp(
            base_url="http://example/v1",
            model="m",
            max_tokens=None,
            temperature=0,
            top_p=None,
            timeout=1,
            system=None,
            reasoning_effort=None,
            conversation_file=tmp_path / "conversation.json",
            history_file=tmp_path / "history",
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            app._append_log("assistant> ")
            app._append_delta("hello")
            app._append_delta(" world")
            assert "assistant> hello world" in app.transcript_text
            assert app.transcript_text.count("assistant>") == 1

    import asyncio

    asyncio.run(run())


def test_input_history_roundtrip(tmp_path):
    path = tmp_path / "history"
    client_chat._save_input_history(path, ["first", "first", "", "second"])

    assert client_chat._load_input_history(path) == ["first", "second"]


def test_textual_composer_uses_persistent_input_history(tmp_path):
    if client_chat.App is None:
        return

    history_file = tmp_path / "history"
    client_chat._save_input_history(history_file, ["first prompt", "second prompt"])

    async def run():
        app = client_chat.LangBurstTextualApp(
            base_url="http://example/v1",
            model="m",
            max_tokens=None,
            temperature=0,
            top_p=None,
            timeout=1,
            system=None,
            reasoning_effort=None,
            conversation_file=tmp_path / "conversation.json",
            history_file=history_file,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app.composer_history_previous() is True
            assert app._composer().text == "second prompt"
            assert app.composer_history_previous() is True
            assert app._composer().text == "first prompt"
            assert app.composer_history_next() is True
            assert app._composer().text == "second prompt"
            assert app.composer_history_next() is True
            assert app._composer().text == ""

    import asyncio

    asyncio.run(run())


class _FakeResponse:
    def __init__(self, lines: list[str]) -> None:
        self.lines = [line.encode("utf-8") for line in lines]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter(self.lines)


def test_stream_request_parses_sse(monkeypatch):
    chunks = [
        {"choices": [{"delta": {"content": "안녕"}}]},
        {"choices": [{"delta": {"content": "!"}}]},
    ]
    lines = [
        f"data: {json.dumps(chunks[0], ensure_ascii=False)}\n",
        "\n",
        f"data: {json.dumps(chunks[1], ensure_ascii=False)}\n",
        "\n",
        "data: [DONE]\n",
        "\n",
    ]

    monkeypatch.setattr(client_chat, "urlopen", lambda _req, timeout: _FakeResponse(lines))

    assert list(client_chat._stream_request("http://example/v1/chat/completions", {"stream": True}, timeout=1)) == chunks


def test_run_turn_prints_usage_metrics(monkeypatch, capsys):
    events = [
        {"choices": [{"delta": {"content": "안녕"}}]},
        {
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "prompt_tokens_details": {"cached_tokens": 1},
                "completion_tokens_details": {"accepted_prediction_tokens": 0, "rejected_prediction_tokens": 0},
                "performance": {"prefill_tok_s": 1000.0, "ttft_s": 0.1, "decode_tok_s": 20.0, "e2e_tok_s": 10.0, "finish_reason": "stop"},
            },
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        },
    ]

    monkeypatch.setattr(client_chat, "_stream_request", lambda *_args, **_kwargs: iter(events))

    answer, usage, cancelled = client_chat._run_turn("http://example/v1", {"stream": True}, timeout=1)
    out = capsys.readouterr().out

    assert answer == "안녕"
    assert usage is events[1]["usage"]
    assert cancelled is False
    assert "metrics> prefill=1000.00 tok/s decode=20.00 tok/s" in out
    assert "prompt=3" not in out
    assert "decode=20.00 tok/s" in out
