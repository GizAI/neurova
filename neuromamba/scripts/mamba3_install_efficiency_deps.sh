#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U bitsandbytes deepspeed

python - <<'PY'
import importlib
for name in ("bitsandbytes", "deepspeed"):
    mod = importlib.import_module(name)
    print(f"{name} {getattr(mod, '__version__', 'unknown')}")
PY
