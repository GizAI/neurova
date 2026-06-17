#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
LangBurst runner

Usage:
  ./neurova.sh langburst help
  ./neurova.sh langburst server [...]
  ./neurova.sh langburst chat [...]
  ./neurova.sh langburst generate [...]
  ./neurova.sh langburst doctor [...]
  ./neurova.sh langburst bench [...]
  ./neurova.sh langburst install-cpu

Notes:
  chat talks to an OpenAI-compatible LangBurst API and streams replies.
  generate loads an engine/model in-process for one-shot local generation.
  server/bench expect LangBurst to be installed in the active Python env.
  install-cpu installs the package without building CUDA extensions.
EOF
}

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  help|-h|--help)
    usage
    ;;
  server|serve)
    exec langburst-server "$@"
    ;;
  chat)
    exec python -m langburst.client_chat "$@"
    ;;
  generate|local-chat)
    exec langburst-chat "$@"
    ;;
  doctor)
    exec langburst-doctor "$@"
    ;;
  bench|bench-serving)
    exec langburst-bench-serving "$@"
    ;;
  install-cpu)
    LANGBURST_SKIP_CUDA_EXT=1 exec python -m pip install -e .
    ;;
  *)
    echo "Unknown LangBurst command: $cmd" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
