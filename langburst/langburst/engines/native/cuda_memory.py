from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from typing import Mapping

import torch


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 1/0, true/false, on/off")


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return int(default)
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


@dataclass(frozen=True)
class CudaMemoryPolicy:
    """One CUDA cache release policy for native serving.

    It avoids the old split where worker and runner both read different env
    names and could synchronize/empty-cache more often than intended.
    """

    trim_after_request: bool = True
    trim_free_below_mib: int = 768

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "CudaMemoryPolicy":
        source = os.environ if env is None else env
        enabled_raw = (
            source.get("LANGBURST_EMPTY_CACHE_AFTER_REQUEST")
            or source.get("LANGBURST_TRIM_CACHE_AFTER_REQUEST")
            or source.get("LANGBURST_TRIM_CACHE_DURING_PREFILL")
        )
        enabled = _env_bool({"enabled": enabled_raw or ""}, "enabled", True)
        return cls(
            trim_after_request=enabled,
            trim_free_below_mib=_env_int(source, "LANGBURST_TRIM_CACHE_FREE_BELOW_MIB", 768),
        )

    def release_idle_cache(self, *, active_requests: int = 0) -> bool:
        if not self.should_release(active_requests=active_requests):
            return False
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return True

    def should_release(self, *, active_requests: int = 0) -> bool:
        if active_requests > 0 or not self.trim_after_request:
            return False
        if not torch.cuda.is_available():
            return False
        if self.trim_free_below_mib > 0:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            if free_bytes >= self.trim_free_below_mib * 1024 * 1024:
                return False
        return True

    def summary(self) -> dict[str, int | bool]:
        return {
            "trim_after_request": self.trim_after_request,
            "trim_free_below_mib": self.trim_free_below_mib,
        }
