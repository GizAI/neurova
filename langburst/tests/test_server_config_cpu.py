from __future__ import annotations

import json
from pathlib import Path

from langburst.core.defaults import DEFAULT_SERVING_RECENT_WINDOW
from langburst.core.chat_template import resolve_chat_template_kwargs
from langburst.core.text_stream import LeadingRoleEchoFilter, StopTextFilter, ThinkingTextFilter, hide_thinking_text
from langburst.core.features import RuntimeFeatures
from langburst.engines.native.manager import load_model_specs
from langburst.server import (
    ChatCompletionRequest,
    _assistant_message_from_text,
    _final_response_text,
    _generation_chat_template_kwargs,
    _generation_messages,
    _long_document_stop_strings,
    _native_generation_config,
    _native_stop_token_sequences,
    _request_messages,
    _requested_generation_tokens,
    _validate_request_shape,
)


def test_load_model_specs_from_json(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_name": "toy-a",
                        "adapter": "toy",
                        "hf_model": str(tmp_path / "hf"),
                        "qb_model": str(tmp_path / "qb"),
                        "device": "cpu",
                        "recent_window": 128,
                        "runtime_profile": "original",
                        "block_prefill": False,
                        "state_pool": False,
                        "gpu_sampling": False,
                        "estimated_vram_mib": 1234,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = load_model_specs(path, RuntimeFeatures.from_profile("stateful"))
    assert len(specs) == 1
    assert specs[0].model_name == "toy-a"
    assert specs[0].adapter_id == "toy"
    assert specs[0].device == "cpu"
    assert specs[0].recent_window == 128
    assert specs[0].estimated_vram_mib == 1234
    assert not specs[0].runtime_features.block_prefill
    assert not specs[0].runtime_features.state_pool
    assert not specs[0].runtime_features.gpu_sampling


def test_load_model_specs_defaults_to_safe_server_window(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            [
                {
                    "model_name": "toy-a",
                    "adapter": "toy",
                    "hf_model": str(tmp_path / "hf"),
                    "qb_model": str(tmp_path / "qb"),
                    "device": "cpu",
                }
            ]
        ),
        encoding="utf-8",
    )
    specs = load_model_specs(path, RuntimeFeatures.from_profile("stateful"))
    assert specs[0].recent_window == DEFAULT_SERVING_RECENT_WINDOW


def test_load_model_specs_requires_explicit_adapter(tmp_path: Path):
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            [
                {
                    "model_name": "toy-a",
                    "hf_model": str(tmp_path / "hf"),
                    "qb_model": str(tmp_path / "qb"),
                    "device": "cpu",
                }
            ]
        ),
        encoding="utf-8",
    )
    try:
        load_model_specs(path, RuntimeFeatures.from_profile("stateful"))
    except ValueError as exc:
        assert "explicit adapter" in str(exc)
    else:
        raise AssertionError("missing adapter should fail fast")


def test_native_generation_defaults_disable_thinking_and_min_output(monkeypatch):
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {
                "<|im_start|>": [248047],
                "<|im_end|>": [248046],
                "<think>": [248068],
                "</think>": [248069],
            }[text]

    class Engine:
        tokenizer = Tokenizer()

        @staticmethod
        def eos_token_ids():
            return (248046,)

    monkeypatch.delenv("LANGBURST_DEFAULT_MIN_NEW_TOKENS", raising=False)
    monkeypatch.delenv("LANGBURST_SUPPRESS_THINK_TOKENS", raising=False)

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=128)
    cfg = _native_generation_config(Engine(), req)

    assert cfg.min_new_tokens == 0
    assert 248047 in cfg.suppress_tokens
    assert 248046 not in cfg.suppress_tokens
    assert 248068 not in cfg.suppress_tokens
    assert 248069 not in cfg.suppress_tokens


def test_native_generation_can_opt_into_think_token_suppression(monkeypatch):
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {
                "<|im_start|>": [248047],
                "<|im_end|>": [248046],
                "<think>": [248068],
                "</think>": [248069],
            }[text]

    class Engine:
        tokenizer = Tokenizer()

        @staticmethod
        def eos_token_ids():
            return (248046,)

    monkeypatch.setenv("LANGBURST_SUPPRESS_THINK_TOKENS", "1")

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=128)
    cfg = _native_generation_config(Engine(), req)

    assert 248047 in cfg.suppress_tokens
    assert 248046 not in cfg.suppress_tokens
    assert 248068 in cfg.suppress_tokens
    assert 248069 in cfg.suppress_tokens


def test_requested_generation_tokens_are_capped_by_runtime_policy(monkeypatch):
    monkeypatch.delenv("LANGBURST_MIN_COMPLETION_TOKENS", raising=False)
    monkeypatch.setenv("LANGBURST_DEFAULT_MAX_TOKENS", "256")
    monkeypatch.setenv("LANGBURST_MAX_GENERATION_TOKENS", "1024")

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=2048)

    assert _requested_generation_tokens(req) == 1024


def test_requested_generation_tokens_use_default_when_client_omits_max(monkeypatch):
    monkeypatch.delenv("LANGBURST_MIN_COMPLETION_TOKENS", raising=False)
    monkeypatch.setenv("LANGBURST_DEFAULT_MAX_TOKENS", "256")
    monkeypatch.setenv("LANGBURST_MAX_GENERATION_TOKENS", "1024")

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

    assert _requested_generation_tokens(req) == 256


def test_requested_generation_tokens_raise_small_client_budget_to_operating_floor(monkeypatch):
    monkeypatch.setenv("LANGBURST_DEFAULT_MAX_TOKENS", "256")
    monkeypatch.setenv("LANGBURST_MAX_GENERATION_TOKENS", "1024")
    monkeypatch.setenv("LANGBURST_MIN_COMPLETION_TOKENS", "384")

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=256)

    assert _requested_generation_tokens(req) == 384


def test_requested_generation_tokens_allow_strict_client_budget(monkeypatch):
    monkeypatch.setenv("LANGBURST_DEFAULT_MAX_TOKENS", "256")
    monkeypatch.setenv("LANGBURST_MAX_GENERATION_TOKENS", "1024")
    monkeypatch.setenv("LANGBURST_MIN_COMPLETION_TOKENS", "384")

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=256,
        metadata={"langburst_strict_max_tokens": True},
    )

    assert _requested_generation_tokens(req) == 256


def test_requested_generation_tokens_cap_still_wins_over_floor(monkeypatch):
    monkeypatch.setenv("LANGBURST_DEFAULT_MAX_TOKENS", "256")
    monkeypatch.setenv("LANGBURST_MAX_GENERATION_TOKENS", "320")
    monkeypatch.setenv("LANGBURST_MIN_COMPLETION_TOKENS", "384")

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=256)

    assert _requested_generation_tokens(req) == 320


def test_native_generation_includes_repetition_guard(monkeypatch):
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {
                "<|im_start|>": [248047],
                "<|im_end|>": [248046],
                "<think>": [248068],
                "</think>": [248069],
            }[text]

    class Engine:
        tokenizer = Tokenizer()

        @staticmethod
        def eos_token_ids():
            return (248046,)

    monkeypatch.setenv("LANGBURST_REPETITION_STOP_NGRAM_SIZE", "4")
    monkeypatch.setenv("LANGBURST_REPETITION_STOP_MIN_NGRAM_SIZE", "3")
    monkeypatch.setenv("LANGBURST_REPETITION_STOP_REPEATS", "6")

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=128)
    cfg = _native_generation_config(Engine(), req)

    assert cfg.repetition_stop_min_ngram_size == 3
    assert cfg.repetition_stop_ngram_size == 4
    assert cfg.repetition_stop_repeats == 6
    assert 248047 in cfg.stop_token_ids


def test_long_document_context_adds_markdown_separator_stop():
    selected = [{"role": "user", "content": "x" * 2500}]
    original = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        selected[0],
    ]

    assert _long_document_stop_strings(selected_messages=selected, original_messages=original, metadata=None) == ("\n\n---",)
    assert _long_document_stop_strings(selected_messages=selected, original_messages=selected, metadata=None) == ("\n\n---",)
    assert _long_document_stop_strings(
        selected_messages=[{"role": "user", "content": "short"}],
        original_messages=[{"role": "user", "content": "short"}],
        metadata=None,
    ) == ()
    assert _long_document_stop_strings(
        selected_messages=selected,
        original_messages=original,
        metadata={"langburst_message_context": "raw"},
    ) == ()


def test_native_stop_sequences_include_extra_stop_strings():
    class Tokenizer:
        def encode(self, text: str):
            return {
                "\n\n---": [198, 198, 14374],
            }[text]

    class Engine:
        tokenizer = Tokenizer()

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

    assert _native_stop_token_sequences(Engine(), req, extra_stop_strings=("\n\n---",)) == ((198, 198, 14374),)


def test_stop_text_filter_holds_split_stop_string():
    filt = StopTextFilter(("\n\n---",))

    assert filt.push("answer") == "an"
    assert filt.push("\n") == "s"
    assert filt.push("\n--") == "wer"
    assert filt.push("- trailing") == ""
    assert filt.stopped is True


def test_leading_role_echo_filter_removes_only_initial_chatml_echo():
    filt = LeadingRoleEchoFilter()

    assert filt.push("**\nuser\n") == ""
    assert filt.push("assistant\n\nAnswer") == "Answer"
    assert filt.push(" mentions user and assistant later") == " mentions user and assistant later"


def test_leading_role_echo_filter_preserves_normal_opening_on_final():
    filt = LeadingRoleEchoFilter()

    assert filt.push("assistant tools can help", final=True) == "assistant tools can help"


def test_final_response_text_filters_nonstream_split_stop_string():
    req = ChatCompletionRequest(
        model="langburst-qwen3.6-27b-q4",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
    )

    assert _final_response_text(
        "정상 답변\n\n--- hidden",
        req=req,
        model_name=req.model,
        extra_stop_strings=("\n\n---",),
    ) == "정상 답변"


def test_final_response_text_filters_leading_role_echo_nonstream():
    req = ChatCompletionRequest(
        model="langburst-qwen3.6-27b-q4",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=64,
    )

    assert _final_response_text(
        "user\nassistant\n\n안녕하세요",
        req=req,
        model_name=req.model,
    ) == "안녕하세요"


def test_native_generation_allows_think_tokens_when_explicitly_enabled(monkeypatch):
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {
                "<|im_start|>": [248047],
                "<|im_end|>": [248046],
                "<think>": [248068],
                "</think>": [248069],
            }[text]

    class Engine:
        tokenizer = Tokenizer()

        @staticmethod
        def eos_token_ids():
            return (248046,)

    monkeypatch.delenv("LANGBURST_SUPPRESS_THINK_TOKENS", raising=False)

    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=128,
        reasoning_effort="low",
        chat_template_kwargs={"enable_thinking": True, "preserve_thinking": True},
    )
    cfg = _native_generation_config(Engine(), req)

    assert 248047 in cfg.suppress_tokens
    assert 248046 not in cfg.suppress_tokens
    assert 248068 not in cfg.suppress_tokens
    assert 248069 not in cfg.suppress_tokens


def test_chat_template_policy_has_single_non_thinking_default():
    class Request:
        chat_template_kwargs = None
        reasoning_effort = None

    assert resolve_chat_template_kwargs()["enable_thinking"] is False
    assert resolve_chat_template_kwargs()["preserve_thinking"] is False
    req = Request()
    req.chat_template_kwargs = {"enable_thinking": True}
    assert resolve_chat_template_kwargs(req)["enable_thinking"] is True
    assert resolve_chat_template_kwargs(req)["preserve_thinking"] is True
    req.reasoning_effort = "none"
    assert resolve_chat_template_kwargs(req)["enable_thinking"] is False
    assert resolve_chat_template_kwargs(req)["preserve_thinking"] is False
    effort_only_req = Request()
    effort_only_req.reasoning_effort = "low"
    assert resolve_chat_template_kwargs(effort_only_req)["enable_thinking"] is False
    assert resolve_chat_template_kwargs(effort_only_req)["preserve_thinking"] is False


def test_tools_do_not_force_visible_thinking_template():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    kwargs = _generation_chat_template_kwargs(req, tools=req.tools)

    assert kwargs["enable_thinking"] is False
    assert kwargs["preserve_thinking"] is False


def test_request_messages_preserves_content_without_hidden_rewrite():
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "HEAD " + ("x" * 10000) + " TAIL"},
            {"role": "user", "content": [{"type": "text", "text": "kept"}]},
        ],
        max_tokens=16,
    )

    messages = _request_messages(req)

    assert messages[0]["content"] == "HEAD " + ("x" * 10000) + " TAIL"
    assert messages[1]["content"] == [{"type": "text", "text": "kept"}]


def test_request_messages_drops_empty_assistant_turns_only():
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "assistant", "content": "kept"},
            {"role": "user", "content": ""},
        ],
        max_tokens=16,
    )

    messages = _request_messages(req)

    assert messages == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "kept"},
        {"role": "user", "content": ""},
    ]


def test_native_request_shape_allows_advertised_tools_without_forced_choice():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "noop",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    _validate_request_shape(req)


def test_native_request_shape_rejects_forced_tool_choice():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "noop",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "noop"}},
    )

    try:
        _validate_request_shape(req)
    except ValueError as exc:
        assert "tool_choice" in str(exc)
    else:
        raise AssertionError("forced tool_choice must require a tool-capable engine")


def test_request_messages_preserves_empty_assistant_tool_calls():
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "create file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"echo ok > tool_test.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": ""},
            {"role": "user", "content": "what happened?"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    messages = _request_messages(req)

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert messages[2] == {"role": "tool", "content": "", "tool_call_id": "call_1"}


def test_generation_messages_keeps_tool_transcript_instead_of_plain_text_shortcut():
    class Engine:
        def encode_messages(self, messages, **_kwargs):
            return [0] * len(str(messages))

    req = ChatCompletionRequest(
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "create file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": '{"cmd":"echo ok > tool_test.txt"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            {"role": "user", "content": "report the result"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    selected = _generation_messages(Engine(), req)

    assert [message["role"] for message in selected] == ["user", "assistant", "tool", "user"]
    assert selected[1]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert selected[2]["content"] == "ok"


def test_qwen_tool_call_xml_becomes_openai_tool_calls():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        """I will call it.
<tool_call>
<function=write_file>
<parameter=path>
tool_test.txt
</parameter>
<parameter=content>
LangBurst tool call ok
</parameter>
</function>
</tool_call>""",
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["tool_calls"][0]["type"] == "function"
    assert message["tool_calls"][0]["function"]["name"] == "write_file"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "path": "tool_test.txt",
        "content": "LangBurst tool call ok",
    }


def test_qwen_tool_call_parser_accepts_name_line_variant():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        """</think>

<tool_call>
write_file
<parameter=path>
tool_test.txt
</parameter>
<parameter=content>
LangBurst tool call ok
</parameter>
</function>
</tool_call>""",
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "write_file"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "path": "tool_test.txt",
        "content": "LangBurst tool call ok",
    }


def test_qwen_tool_call_parser_accepts_schema_matched_json_block():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}, "workdir": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_stdin",
                    "parameters": {
                        "type": "object",
                        "properties": {"session_id": {"type": "integer"}, "chars": {"type": "string"}},
                        "required": ["session_id"],
                    },
                },
            },
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        """I will invoke the tool.
```json
{"cmd":"echo \\"LangBurst tool call ok\\" > tool_test.txt"}
```""",
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt',
    }


def test_qwen_tool_call_parser_accepts_exec_command_shell_block():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        """I'll use `exec_command`.
```bash
echo "LangBurst tool call ok" > tool_test.txt && cat tool_test.txt
```""",
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt && cat tool_test.txt',
    }


def test_qwen_tool_call_parser_accepts_tagged_json_tool_call():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '<exec_command>{"cmd": "echo \\"LangBurst tool call ok\\" > tool_test.txt"}',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt',
    }


def test_qwen_tool_call_parser_accepts_invoke_xml_tool_call():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '<invoke name="exec_command">\n'
        '<parameter name="cmd">echo "LangBurst tool call ok" > tool_test.txt</parameter>\n'
        "</invoke>",
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt',
    }


def test_qwen_tool_call_parser_accepts_tool_use_json_xml():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '<tool_use>\n{"name": "exec_command", "arguments": {"cmd": "cat tool_test.txt"}}\n</tool_use>',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"cmd": "cat tool_test.txt"}


def test_qwen_tool_call_parser_accepts_embedded_json_tool_call():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        'Now call it. {"cmd": "echo \\"LangBurst tool call ok\\" > tool_test.txt", "name": "exec_command"}',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt',
    }


def test_qwen_tool_call_parser_accepts_schema_matched_json_without_name():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_goal",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '{"cmd": "echo \\"LangBurst tool call ok\\" > tool_test.txt"}',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt',
    }


def test_tool_result_phase_does_not_reparse_tool_like_text_as_new_call():
    req = ChatCompletionRequest(
        messages=[
            {"role": "user", "content": "create file"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": '{"cmd":"cat tool_test.txt"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "LangBurst tool call ok"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '<exec_command>{"cmd":"cat tool_test.txt"}',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "stop"
    assert message == {"role": "assistant", "content": '<exec_command>{"cmd":"cat tool_test.txt"}'}


def test_qwen_tool_call_parser_accepts_named_tool_input_pair():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '<tool_uses>exec_command</tool_uses>\n'
        '<tool_input>echo "LangBurst tool call ok" > tool_test.txt && cat tool_test.txt</tool_input>',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt && cat tool_test.txt',
    }


def test_qwen_tool_call_parser_accepts_fenced_tool_name_block():
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "write"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
            }
        ],
    )

    message, finish_reason = _assistant_message_from_text(
        '```exec_command\necho "LangBurst tool call ok" > tool_test.txt && cat tool_test.txt',
        req=req,
        model_name="langburst-qwen3.6-27b-q4",
    )

    assert finish_reason == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "exec_command"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "cmd": 'echo "LangBurst tool call ok" > tool_test.txt && cat tool_test.txt',
    }


def test_hide_thinking_text_removes_reasoning_spans():
    assert hide_thinking_text("a <think>hidden</think> b") == "a b"
    assert hide_thinking_text("</think>\nanswer") == "answer"


def test_thinking_text_filter_handles_split_tags():
    filt = ThinkingTextFilter(enabled=True)
    out = []

    out.append(filt.push("hello <thi"))
    out.append(filt.push("nk>secret"))
    out.append(filt.push("</thi"))
    out.append(filt.push("nk> answer", final=True))

    assert "".join(out) == "hello answer"
