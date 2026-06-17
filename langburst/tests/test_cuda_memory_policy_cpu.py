from __future__ import annotations

import torch

from langburst.engines.native.cuda_memory import CudaMemoryPolicy


def test_cuda_memory_policy_skips_when_free_memory_is_above_threshold(monkeypatch):
    calls: list[str] = []
    policy = CudaMemoryPolicy(trim_after_request=True, trim_free_below_mib=128)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (256 * 1024 * 1024, 1024 * 1024 * 1024))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    assert policy.release_idle_cache(active_requests=0) is False
    assert calls == []


def test_cuda_memory_policy_releases_when_idle_and_under_threshold(monkeypatch):
    calls: list[str] = []
    policy = CudaMemoryPolicy(trim_after_request=True, trim_free_below_mib=128)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (64 * 1024 * 1024, 1024 * 1024 * 1024))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: calls.append("ipc"))

    assert policy.release_idle_cache(active_requests=0) is True
    assert "empty_cache" in calls


def test_cuda_memory_policy_does_not_release_while_active(monkeypatch):
    calls: list[str] = []
    policy = CudaMemoryPolicy(trim_after_request=True, trim_free_below_mib=0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    assert policy.release_idle_cache(active_requests=1) is False
    assert calls == []


def test_cuda_memory_policy_from_env_accepts_legacy_names():
    policy = CudaMemoryPolicy.from_env(
        {
            "LANGBURST_EMPTY_CACHE_AFTER_REQUEST": "0",
            "LANGBURST_TRIM_CACHE_FREE_BELOW_MIB": "512",
        }
    )

    assert policy.trim_after_request is False
    assert policy.trim_free_below_mib == 512
