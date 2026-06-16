from __future__ import annotations

import os

LOWBIT_ROWS_PER_CTA_CHOICES = (4, 8, 16)
DEFAULT_LOWBIT_ROWS_PER_CTA = 8
DEFAULT_MARLIN_DIRECT_MAX_BATCH = 4
DEFAULT_FAST_RAW_BLOCK = True
DEFAULT_SPECULATIVE_VERIFIER = "transaction_block"
SPECULATIVE_VERIFIER_CHOICES = ("sequential", "transaction_block")


def lowbit_rows_per_cta(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("QWENBURST_LOWBIT_ROWS_PER_CTA")
    if raw is None or raw == "":
        return DEFAULT_LOWBIT_ROWS_PER_CTA
    rows = int(raw)
    if rows not in LOWBIT_ROWS_PER_CTA_CHOICES:
        choices = ", ".join(str(v) for v in LOWBIT_ROWS_PER_CTA_CHOICES)
        raise ValueError(f"lowbit rows_per_cta must be one of: {choices}")
    return rows


def marlin_direct_max_batch(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("QWENBURST_MARLIN_DIRECT_MAX_BATCH")
    if raw is None or raw == "":
        return DEFAULT_MARLIN_DIRECT_MAX_BATCH
    batch = int(raw)
    if batch < 1:
        raise ValueError("Marlin direct max batch must be >= 1")
    return batch


def fast_raw_block_enabled(value: str | int | bool | None = None) -> bool:
    raw = value if value is not None else os.environ.get("QWENBURST_FAST_RAW_BLOCK")
    if raw is None or raw == "":
        return DEFAULT_FAST_RAW_BLOCK
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError("QWENBURST_FAST_RAW_BLOCK must be one of: 1/0, true/false, on/off")


def speculative_verifier_mode(value: str | None = None) -> str:
    raw = value if value is not None else os.environ.get("QWENBURST_SPECULATIVE_VERIFIER")
    if raw is None or raw == "":
        return DEFAULT_SPECULATIVE_VERIFIER
    mode = str(raw).strip().lower().replace("-", "_")
    if mode not in SPECULATIVE_VERIFIER_CHOICES:
        choices = ", ".join(SPECULATIVE_VERIFIER_CHOICES)
        raise ValueError(f"QWENBURST_SPECULATIVE_VERIFIER must be one of: {choices}")
    return mode
