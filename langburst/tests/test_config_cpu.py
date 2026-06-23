from __future__ import annotations

from pathlib import Path

import pytest

from langburst.config import export_shell, load_serving_config, resolved_env


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "ml-dmc8-q4.yaml"


def test_ml_dmc8_q4_config_resolves_champion_serving_defaults():
    env = resolved_env(load_serving_config(CONFIG_PATH))

    assert env["MODEL_NAME"] == "langburst-qwen3.6-27b-q4"
    assert env["QB_DIR"] == "/home/user/models/Qwen3.6-27B-qb4-marlin-fused"
    assert env["LANGBURST_CONTEXT_WINDOW"] == "65536"
    assert env["LANGBURST_CONTEXT_TIERS"] == "4096,65536"
    assert env["LANGBURST_CONTEXT_TIER_SLOTS"] == "2,1"
    assert env["LANGBURST_KV_CACHE_DTYPE"] == "int4_bdr"
    assert env["LANGBURST_MAX_GENERATION_TOKENS"] == "8192"
    assert env["LANGBURST_DEFAULT_MAX_TOKENS"] == "8192"
    assert env["LANGBURST_MTP_MAX_DRAFT"] == "4"
    assert env["LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE"] == "2:1"
    assert env["LANGBURST_DEFAULT_MIN_NEW_TOKENS"] == "0"
    assert env["LANGBURST_MIN_COMPLETION_TOKENS"] == "0"
    assert env["LANGBURST_MESSAGE_CONTEXT_POLICY"] == "auto_long_user"
    assert env["LANGBURST_STANDALONE_USER_MIN_TOKENS"] == "1024"
    assert env["LANGBURST_MARLIN_INTERNAL_ARGMAX"] == "1"
    assert env["LANGBURST_GDN_SPEC_NORM_GATE_FUSED"] == "1"
    assert env["LANGBURST_MTP_LOCAL_TKH_ATTENTION"] == "0"
    assert env["LANGBURST_CUDA_GRAPH"] == "0"
    assert env["LANGBURST_REPETITION_STOP_MIN_NGRAM_SIZE"] == "32"
    assert env["LANGBURST_REPETITION_STOP_NGRAM_SIZE"] == "96"
    assert env["LANGBURST_REPETITION_STOP_REPEATS"] == "2"
    assert env["LANGBURST_SUPPRESS_THINK_TOKENS"] == "0"
    assert env["LANGBURST_SUPPRESS_CHAT_CONTROL_TOKENS"] == "0"
    assert env["PREFIX_CACHE"] == "on"


def test_server_env_key_manifest_covers_generated_runtime_env():
    env = resolved_env(load_serving_config(CONFIG_PATH))
    server_keys = set(env["LANGBURST_SERVER_ENV_KEYS"].split())

    for key in (
        "LANGBURST_CONTEXT_WINDOW",
        "LANGBURST_PAGED_KV",
        "LANGBURST_MARLIN_OUT_CACHE_POLICY",
        "LANGBURST_MTP_MAX_DRAFT",
        "LANGBURST_TRIM_MODEL_CACHE_BEFORE_PREFILL",
        "LANGBURST_MESSAGE_CONTEXT_POLICY",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        assert key in server_keys
        assert key in env


def test_yaml_config_rejects_tier_larger_than_context_window(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
serving:
  context_window: 4096
  context_tiers: [4096, 8192]
  context_tier_slots: [1, 1]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_window"):
        load_serving_config(path)


def test_shell_export_is_directly_eval_safe_for_start_script():
    script = export_shell(load_serving_config(CONFIG_PATH))

    assert "export MODEL_NAME=langburst-qwen3.6-27b-q4" in script
    assert "export LANGBURST_MTP_MAX_DRAFT_BY_ACTIVE=2:1" in script
    assert "LANGBURST_SERVER_ENV_KEYS=" in script
