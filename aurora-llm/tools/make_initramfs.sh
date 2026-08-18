#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "usage: $0 MODEL.ali OUTPUT.cpio.gz" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="$(realpath "$1")"
OUT="$(realpath -m "$2")"
make -C "$ROOT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$ROOT/aurora-llm" "$TMP/init"
cp "$MODEL" "$TMP/model.ali"
chmod 0755 "$TMP/init"
(
  cd "$TMP"
  printf '%s\n' init model.ali | cpio -o -H newc 2>/dev/null | gzip -1 > "$OUT"
)
echo "wrote $OUT"
echo "The guest Linux kernel must have the NIC driver and CONFIG_IP_PNP built in."
