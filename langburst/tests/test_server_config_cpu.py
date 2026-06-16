from __future__ import annotations

import json
from pathlib import Path

from langburst.core.defaults import DEFAULT_SERVING_RECENT_WINDOW
from langburst.core.features import RuntimeFeatures
from langburst.engines.native_impl.manager import load_model_specs


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
