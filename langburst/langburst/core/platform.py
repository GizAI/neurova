from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PLATFORM_NAME = "LangBurst"
PACKAGE_NAME = "langburst"
ENV_PREFIX = "LANGBURST"
INDEX_FILENAMES = ("langburst_index.json",)


def env(name: str, default: str | None = None) -> str | None:
    """Read a LangBurst env var."""

    return os.environ.get(f"{ENV_PREFIX}_{name}", default)


def env_flag(name: str, default: bool = False) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_index_file(root: str | Path, filenames: Iterable[str] = INDEX_FILENAMES) -> Path:
    base = Path(root)
    for filename in filenames:
        path = base / filename
        if path.exists():
            return path
    names = ", ".join(filenames)
    raise FileNotFoundError(f"runtime index not found in {base}; expected one of: {names}")
