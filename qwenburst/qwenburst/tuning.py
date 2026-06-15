from __future__ import annotations

import os

LOWBIT_ROWS_PER_CTA_CHOICES = (4, 8, 16)
DEFAULT_LOWBIT_ROWS_PER_CTA = 8


def lowbit_rows_per_cta(value: int | str | None = None) -> int:
    raw = value if value is not None else os.environ.get("QWENBURST_LOWBIT_ROWS_PER_CTA")
    if raw is None or raw == "":
        return DEFAULT_LOWBIT_ROWS_PER_CTA
    rows = int(raw)
    if rows not in LOWBIT_ROWS_PER_CTA_CHOICES:
        choices = ", ".join(str(v) for v in LOWBIT_ROWS_PER_CTA_CHOICES)
        raise ValueError(f"lowbit rows_per_cta must be one of: {choices}")
    return rows
