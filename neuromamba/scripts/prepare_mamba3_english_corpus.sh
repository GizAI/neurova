#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/user/workspace/neurova}"
OUT="${OUT:-${ROOT}/luma/data/english_bootstrap.txt}"
mkdir -p "$(dirname "${OUT}")"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT

urls=(
  "https://www.gutenberg.org/files/1342/1342-0.txt"
  "https://www.gutenberg.org/files/1661/1661-0.txt"
  "https://www.gutenberg.org/files/84/84-0.txt"
)

: > "${tmp}"
for url in "${urls[@]}"; do
  curl -L --fail --retry 3 "$url" >> "${tmp}"
  printf '\n\n' >> "${tmp}"
done

python - "$tmp" "$OUT" <<'PY'
from pathlib import Path
import re
import sys

src = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
src = re.sub(r"\r\n?", "\n", src)
src = re.sub(r"\n{3,}", "\n\n", src)
src = re.sub(r"[ \t]+", " ", src)
lines = []
for line in src.splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("***") or "Gutenberg" in line[:120]:
        continue
    if len(line) < 40:
        continue
    lines.append(line)
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {len(lines)} lines to {sys.argv[2]}")
PY
