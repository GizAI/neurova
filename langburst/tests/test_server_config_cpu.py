from __future__ import annotations

import json
from pathlib import Path

from langburst.core.defaults import DEFAULT_SERVING_RECENT_WINDOW
from langburst.core.features import RuntimeFeatures
from langburst.engines.native.manager import load_model_specs
from langburst.server import ChatCompletionRequest, _native_generation_config, _request_messages, _thinking_visible_prefix, _with_visible_prefix_once


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


def test_native_generation_defaults_enable_thinking_and_min_output(monkeypatch):
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {"<think>": [248068], "</think>": [248069]}[text]

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
    assert 248068 not in cfg.suppress_tokens
    assert 248069 not in cfg.suppress_tokens


def test_native_generation_suppresses_think_tokens_when_explicitly_disabled(monkeypatch):
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {"<think>": [248068], "</think>": [248069]}[text]

    class Engine:
        tokenizer = Tokenizer()

        @staticmethod
        def eos_token_ids():
            return (248046,)

    monkeypatch.delenv("LANGBURST_SUPPRESS_THINK_TOKENS", raising=False)

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], max_tokens=128, enable_thinking=False)
    cfg = _native_generation_config(Engine(), req)

    assert 248068 in cfg.suppress_tokens
    assert 248069 in cfg.suppress_tokens


def test_visible_thinking_prefix_defaults_on_and_can_be_disabled():
    default_req = ChatCompletionRequest(
        model="langburst-qwen3.6-27b-q3",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=16,
    )
    disabled_req = ChatCompletionRequest(
        model="langburst-qwen3.6-27b-q3",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=16,
        enable_thinking=False,
    )
    non_qwen_req = ChatCompletionRequest(
        model="llama-test",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=16,
    )

    assert _thinking_visible_prefix(default_req) == "<think>\n"
    assert _thinking_visible_prefix(disabled_req) == ""
    assert _thinking_visible_prefix(non_qwen_req) == ""


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


def test_visible_thinking_prefix_is_prepended_once():
    text, emitted = _with_visible_prefix_once("reasoning", "<think>\n", emitted=False)
    assert text == "<think>\nreasoning"
    assert emitted

    text, emitted = _with_visible_prefix_once(" more", "<think>\n", emitted=emitted)
    assert text == " more"
    assert emitted

    text, emitted = _with_visible_prefix_once("<think>\nalready", "<think>\n", emitted=False)
    assert text == "<think>\nalready"
    assert emitted
