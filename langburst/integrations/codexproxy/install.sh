#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIPROXYAPI_SRC="${CLIPROXYAPI_SRC:-/home/user/opensources/CLIProxyAPI}"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
CLIPROXY_CONFIG_DIR="${CLIPROXY_CONFIG_DIR:-$HOME/.cli-proxy-api}"
LOCAL_BIN="${LOCAL_BIN:-$HOME/.local/bin}"
LANGBURST_BASE_URL="${LANGBURST_BASE_URL:-http://192.168.0.47:8008/v1}"

if [ ! -d "$CLIPROXYAPI_SRC/.git" ]; then
  echo "CLIProxyAPI source not found: $CLIPROXYAPI_SRC" >&2
  exit 2
fi

mkdir -p "$CODEX_HOME" "$CLIPROXY_CONFIG_DIR" "$LOCAL_BIN"

cp "$SCRIPT_DIR/codexproxy" "$LOCAL_BIN/codexproxy"
chmod 0755 "$LOCAL_BIN/codexproxy"

if [ -f "$CODEX_HOME/proxy.config.toml" ]; then
  cp "$CODEX_HOME/proxy.config.toml" "$CODEX_HOME/proxy.config.toml.bak.$(date +%Y%m%d-%H%M%S)"
fi
install -m 0644 "$SCRIPT_DIR/proxy.config.toml" "$CODEX_HOME/proxy.config.toml"

python3 - "$SCRIPT_DIR/cli-proxy-api.config.yaml" "$CLIPROXY_CONFIG_DIR/config.yaml" "$LANGBURST_BASE_URL" <<'PY'
from pathlib import Path
import os
import sys
from datetime import datetime

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
base_url = sys.argv[3]
if dst.exists():
    backup = dst.with_name(f"{dst.name}.bak.{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    backup.write_text(dst.read_text())
text = src.read_text()
text = text.replace("${LANGBURST_BASE_URL:-http://192.168.0.47:8008/v1}", base_url)
text = text.replace("${LANGBURST_OPENAI_COMPAT_API_KEY:-local-langburst}", os.environ.get("LANGBURST_OPENAI_COMPAT_API_KEY", "local-langburst"))
dst.write_text(text)
PY

(
  cd "$CLIPROXYAPI_SRC"
  if git apply --check "$SCRIPT_DIR/patches/cliproxyapi-openai-responses-developer-role.patch"; then
    git apply "$SCRIPT_DIR/patches/cliproxyapi-openai-responses-developer-role.patch"
  elif git apply --check --reverse "$SCRIPT_DIR/patches/cliproxyapi-openai-responses-developer-role.patch"; then
    echo "CLIProxyAPI developer-role patch already applied"
  else
    echo "CLIProxyAPI developer-role patch does not apply cleanly" >&2
    exit 1
  fi
  go test ./internal/translator/openai/openai/responses
  go build -o /tmp/cli-proxy-api.langburst ./cmd/server
)

install -m 0755 /tmp/cli-proxy-api.langburst "$LOCAL_BIN/cli-proxy-api"

if pgrep -f "^$LOCAL_BIN/cli-proxy-api " >/dev/null 2>&1; then
  pkill -f "^$LOCAL_BIN/cli-proxy-api " || true
fi

nohup "$LOCAL_BIN/cli-proxy-api" \
  -config "$CLIPROXY_CONFIG_DIR/config.yaml" \
  -local-model \
  > /tmp/cli-proxy-api.log 2>&1 &

for _ in $(seq 1 40); do
  if curl -fsS -H 'Authorization: Bearer sk-codex-local' http://127.0.0.1:8317/v1/models >/dev/null; then
    echo "CLIProxyAPI ready: http://127.0.0.1:8317/v1"
    exit 0
  fi
  sleep 0.25
done

echo "CLIProxyAPI did not become ready; see /tmp/cli-proxy-api.log" >&2
tail -n 80 /tmp/cli-proxy-api.log >&2 || true
exit 1
