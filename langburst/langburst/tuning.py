from __future__ import annotations

from dataclasses import dataclass
import os

LOWBIT_ROWS_PER_CTA_CHOICES = (4, 8, 16)
DEFAULT_LOWBIT_ROWS_PER_CTA = 8
DEFAULT_MARLIN_DIRECT_MAX_BATCH = 256
DEFAULT_FAST_RAW_BLOCK = True
DEFAULT_BATCH_STATE_KERNELS = True
DEFAULT_BATCH_PREFILL_STEPS = True
DEFAULT_RAW_PREFILL_BLOCK_TOKENS = 16
DEFAULT_PAGED_PREFILL_BLOCK = True
DEFAULT_PAGED_ATTENTION_KERNELS = False
DEFAULT_SHORT_PREFILL_SDPA_TOKENS = 2048
DEFAULT_SHORT_PREFILL_SDPA_MIN_FREE_MIB = 384
DEFAULT_ATTENTION_RECENT_TOKENS = 128
DEFAULT_PAGED_ATTENTION_BACKEND = "auto"
PAGED_ATTENTION_BACKEND_CHOICES = ("auto", "direct", "flash")
DEFAULT_INT4_KV_LAYOUT = "tiled"
INT4_KV_LAYOUT_CHOICES = ("token", "tiled")
DEFAULT_VERIFY_NEXTN_MODE = "block"
VERIFY_NEXTN_MODE_CHOICES = ("sequential", "block", "fused")


@dataclass(frozen=True)
class PrefillAttentionPolicy:
    fresh_sdpa_tokens: int
    extend_sdpa_tokens: int
    min_free_mib: int
    recent_tokens: int

    def allows_fresh_sdpa(self, *, tokens: int) -> bool:
        return self.fresh_sdpa_tokens > 0 and int(tokens) <= self.fresh_sdpa_tokens

    def allows_extend_sdpa(self, *, live_tokens: int) -> bool:
        return self.extend_sdpa_tokens > 0 and int(live_tokens) <= self.extend_sdpa_tokens


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


def raw_prefill_block_tokens(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("LANGBURST_RAW_PREFILL_BLOCK_TOKENS")
    if raw is None or raw == "":
        return DEFAULT_RAW_PREFILL_BLOCK_TOKENS
    tokens = int(raw)
    if tokens < 1:
        raise ValueError("LANGBURST_RAW_PREFILL_BLOCK_TOKENS must be >= 1")
    return tokens


def paged_prefill_block_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_PAGED_PREFILL_BLOCK")
    if raw is None:
        return DEFAULT_PAGED_PREFILL_BLOCK
    return _parse_env_bool(raw, "LANGBURST_PAGED_PREFILL_BLOCK")


def paged_attention_kernels_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_PAGED_ATTENTION_KERNELS")
    if raw is None:
        return DEFAULT_PAGED_ATTENTION_KERNELS
    return _parse_env_bool(raw, "LANGBURST_PAGED_ATTENTION_KERNELS")


def short_prefill_sdpa_tokens(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("LANGBURST_SHORT_PREFILL_SDPA_TOKENS")
    if raw is None or raw == "":
        return DEFAULT_SHORT_PREFILL_SDPA_TOKENS
    tokens = int(raw)
    if tokens < 0:
        raise ValueError("LANGBURST_SHORT_PREFILL_SDPA_TOKENS must be >= 0")
    return tokens


def short_prefill_sdpa_min_free_mib(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB")
    if raw is None or raw == "":
        return DEFAULT_SHORT_PREFILL_SDPA_MIN_FREE_MIB
    mib = int(raw)
    if mib < 0:
        raise ValueError("LANGBURST_SHORT_PREFILL_SDPA_MIN_FREE_MIB must be >= 0")
    return mib


def attention_recent_tokens(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("LANGBURST_ATTENTION_RECENT_TOKENS")
    if raw is None or raw == "":
        return DEFAULT_ATTENTION_RECENT_TOKENS
    tokens = int(raw)
    if tokens < 0:
        raise ValueError("LANGBURST_ATTENTION_RECENT_TOKENS must be >= 0")
    return tokens


def prefill_attention_policy() -> PrefillAttentionPolicy:
    # LANGBURST_SHORT_PREFILL_SDPA_TOKENS is kept as the compatibility knob for
    # both fresh and small-past extension prefill.  Model code consumes this
    # policy object instead of reading scattered env vars.
    return PrefillAttentionPolicy(
        fresh_sdpa_tokens=short_prefill_sdpa_tokens(),
        extend_sdpa_tokens=short_prefill_sdpa_tokens(),
        min_free_mib=short_prefill_sdpa_min_free_mib(),
        recent_tokens=attention_recent_tokens(),
    )


def paged_attention_backend(value: str | None = None) -> str:
    raw = value if value is not None else os.environ.get("LANGBURST_PAGED_ATTENTION_BACKEND")
    if raw is None or raw == "":
        return DEFAULT_PAGED_ATTENTION_BACKEND
    mode = str(raw).strip().lower().replace("-", "_")
    if mode not in PAGED_ATTENTION_BACKEND_CHOICES:
        choices = ", ".join(PAGED_ATTENTION_BACKEND_CHOICES)
        raise ValueError(f"LANGBURST_PAGED_ATTENTION_BACKEND must be one of: {choices}")
    return mode


def int4_kv_layout(value: str | None = None) -> str:
    raw = value if value is not None else os.environ.get("LANGBURST_INT4_KV_LAYOUT")
    if raw is None or raw == "":
        return DEFAULT_INT4_KV_LAYOUT
    layout = str(raw).strip().lower().replace("-", "_")
    if layout not in INT4_KV_LAYOUT_CHOICES:
        choices = ", ".join(INT4_KV_LAYOUT_CHOICES)
        raise ValueError(f"LANGBURST_INT4_KV_LAYOUT must be one of: {choices}")
    return layout


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


def verify_full_logits_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("LANGBURST_VERIFY_FULL_LOGITS")
    if raw is None or raw == "":
        return False
    return _parse_env_bool(raw, "LANGBURST_VERIFY_FULL_LOGITS")
