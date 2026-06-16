#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

usage() {
  cat <<'EOF'
Neurova workspace router

Usage:
  ./neurova.sh help
  ./neurova.sh langburst <server|chat|doctor|bench|install-cpu> [...]
  ./neurova.sh saneflow [chat|generate|eval|status|train] [...]
  ./neurova.sh mamba3 [serve|chat|bench|status|...] [...]
  ./neurova.sh neuromamba [serve|chat|bench|status|...] [...]
  ./neurova.sh qwen-memory [bf16|4bit]
  NEUROVA_ALLOW_LUMA=1 ./neurova.sh luma [...]

Main project:
  langburst/ is the active serving-engine project.

Policy:
  This root script only routes to project-owned scripts.
  Project logic belongs under each project's scripts/ directory.
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
  langburst)
    exec langburst/scripts/run.sh "$@"
    ;;
  saneflow)
    exec saneflow/scripts/run.sh "$@"
    ;;
  mamba3|neuromamba)
    exec neuromamba/scripts/run.sh "$@"
    ;;
  qwen-memory|qwen_memory)
    exec qwen_memory/scripts/run.sh "$@"
    ;;
  luma)
    exec luma/scripts/run.sh "$@"
    ;;
  *)
    echo "Unknown project or command: $cmd" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac
