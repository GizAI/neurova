from __future__ import annotations

import os

LOWBIT_ROWS_PER_CTA_CHOICES = (4, 8, 16)
DEFAULT_LOWBIT_ROWS_PER_CTA = 8
DEFAULT_MARLIN_DIRECT_MAX_BATCH = 4
DEFAULT_FAST_RAW_BLOCK = True
DEFAULT_BATCH_STATE_KERNELS = True
DEFAULT_BATCH_PREFILL_STEPS = True
DEFAULT_PAGED_ATTENTION_KERNELS = False
DEFAULT_VERIFY_NEXTN_MODE = "fused"
VERIFY_NEXTN_MODE_CHOICES = ("sequential", "block", "fused")


def lowbit_rows_per_cta(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("LANGBURST_LOWBIT_ROWS_PER_CTA")
    if raw is None or raw == "":
        return DEFAULT_LOWBIT_ROWS_PER_CTA
    rows = int(raw)
    if rows not in LOWBIT_ROWS_PER_CTA_CHOICES:
        choices = ", ".join(str(v) for v in LOWBIT_ROWS_PER_CTA_CHOICES)
        raise ValueError(f"lowbit rows_per_cta must be one of: {choices}")
    return rows


def marlin_direct_max_batch(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("LANGBURST_MARLIN_DIRECT_MAX_BATCH")
    if raw is None or raw == "":
        return DEFAULT_MARLIN_DIRECT_MAX_BATCH
    batch = int(raw)
    if batch < 1:
        raise ValueError("Marlin direct max batch must be >= 1")
    return batch


def fast_raw_block_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_FAST_RAW_BLOCK")
    if raw is None or raw == "":
        return DEFAULT_FAST_RAW_BLOCK
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("LANGBURST_FAST_RAW_BLOCK must be one of: 1/0, true/false, on/off")


def batch_state_kernels_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_BATCH_STATE_KERNELS")
    if raw is None:
        return DEFAULT_BATCH_STATE_KERNELS
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("LANGBURST_BATCH_STATE_KERNELS must be one of: 1/0, true/false, on/off")


def batch_conv_kernels_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_BATCH_CONV_KERNELS")
    if raw is None:
        return batch_state_kernels_enabled()
    return _parse_env_bool(raw, "LANGBURST_BATCH_CONV_KERNELS")


def batch_gdn_kernels_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_BATCH_GDN_KERNELS")
    if raw is None:
        return batch_state_kernels_enabled()
    return _parse_env_bool(raw, "LANGBURST_BATCH_GDN_KERNELS")


def batch_prefill_steps_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_BATCH_PREFILL_STEPS")
    if raw is None:
        return DEFAULT_BATCH_PREFILL_STEPS
    return _parse_env_bool(raw, "LANGBURST_BATCH_PREFILL_STEPS")


def paged_attention_kernels_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_PAGED_ATTENTION_KERNELS")
    if raw is None:
        return DEFAULT_PAGED_ATTENTION_KERNELS
    return _parse_env_bool(raw, "LANGBURST_PAGED_ATTENTION_KERNELS")


def _parse_env_bool(raw: str | int | bool, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of: 1/0, true/false, on/off")


def verify_nextn_mode(value: str | None = None) -> str:
    raw = value if value is not None else os.environ.get("LANGBURST_VERIFY_NEXTN_MODE")
    if raw is None or raw == "":
        return DEFAULT_VERIFY_NEXTN_MODE
    mode = str(raw).strip().lower().replace("-", "_")
    if mode not in VERIFY_NEXTN_MODE_CHOICES:
        choices = ", ".join(VERIFY_NEXTN_MODE_CHOICES)
        raise ValueError(f"LANGBURST_VERIFY_NEXTN_MODE must be one of: {choices}")
    return mode
