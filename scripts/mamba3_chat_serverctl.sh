#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-mamba3_siso}" ]]; then
  if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV:-mamba3_siso}"
  fi
fi

ACTION="${1:-status}"
HOST="${NEUROVA_MAMBA3_SERVER_HOST:-127.0.0.1}"
PORT="${NEUROVA_MAMBA3_SERVER_PORT:-8765}"
MODE="${NEUROVA_MAMBA3_MODE:-mamba3-siso-fast-0.3b-ds128}"
TOKENIZER="${NEUROVA_MAMBA3_TOKENIZER:-llama31}"
CHECKPOINT="${NEUROVA_MAMBA3_CHECKPOINT:-runs/mamba3_current/model.pt}"
FALLBACK_CHECKPOINT="${NEUROVA_MAMBA3_FALLBACK_CHECKPOINT:-runs/mamba3_neurova_chat_v1/chat.pt}"
SECOND_FALLBACK_CHECKPOINT="${NEUROVA_MAMBA3_SECOND_FALLBACK_CHECKPOINT:-runs/mamba3_neurova_speak_v1/sft.pt}"
SEQ_LEN="${NEUROVA_MAMBA3_SEQ:-128}"
MAX_NEW="${NEUROVA_MAMBA3_MAX_NEW:-24}"
DTYPE="${NEUROVA_MAMBA3_DTYPE:-bf16}"
DECODE_MODE="${NEUROVA_MAMBA3_DECODE_MODE:-safe}"
TOP_K="${NEUROVA_MAMBA3_TOP_K:-1}"
TOP_P="${NEUROVA_MAMBA3_TOP_P:-0.0}"
TEMP="${NEUROVA_MAMBA3_TEMP:-1.0}"
REPETITION_PENALTY="${NEUROVA_MAMBA3_REPETITION_PENALTY:-1.0}"
CACHE_PARITY_STEPS="${NEUROVA_MAMBA3_CACHE_PARITY_STEPS:-8}"
CUDA_GRAPH="${NEUROVA_MAMBA3_CUDA_GRAPH:-0}"
CACHE_PARITY_GUARD="${NEUROVA_MAMBA3_CACHE_PARITY_GUARD:-1}"
RUN_DIR="${NEUROVA_MAMBA3_SERVER_RUN_DIR:-runs/mamba3_chat_server}"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"
CONFIG_FILE="$RUN_DIR/server.config"

if [[ ! -f "$CHECKPOINT" && -f "$FALLBACK_CHECKPOINT" ]]; then
  CHECKPOINT="$FALLBACK_CHECKPOINT"
fi
if [[ ! -f "$CHECKPOINT" && -f "$SECOND_FALLBACK_CHECKPOINT" ]]; then
  CHECKPOINT="$SECOND_FALLBACK_CHECKPOINT"
fi

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

write_config() {
  mkdir -p "$RUN_DIR"
  cat > "$CONFIG_FILE.tmp" <<EOF
HOST=$HOST
PORT=$PORT
MODE=$MODE
TOKENIZER=$TOKENIZER
CHECKPOINT=$CHECKPOINT
SEQ_LEN=$SEQ_LEN
MAX_NEW=$MAX_NEW
DTYPE=$DTYPE
DECODE_MODE=$DECODE_MODE
TOP_K=$TOP_K
TOP_P=$TOP_P
TEMP=$TEMP
REPETITION_PENALTY=$REPETITION_PENALTY
CACHE_PARITY_STEPS=$CACHE_PARITY_STEPS
CUDA_GRAPH=$CUDA_GRAPH
CACHE_PARITY_GUARD=$CACHE_PARITY_GUARD
EOF
}

config_changed() {
  write_config
  [[ ! -f "$CONFIG_FILE" ]] && return 0
  ! cmp -s "$CONFIG_FILE" "$CONFIG_FILE.tmp"
}

wait_health() {
  local deadline="${1:-90}"
  python - "$HOST" "$PORT" "$deadline" <<'PY'
import sys, time, urllib.request
host, port, deadline = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
url = f"http://{host}:{port}/health"
end = time.time() + deadline
last = None
while time.time() < end:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as r:
            if r.status == 200:
                sys.exit(0)
    except Exception as exc:
        last = exc
    time.sleep(0.5)
raise SystemExit(f"server did not become healthy: {last}")
PY
}

case "$ACTION" in
  start)
    mkdir -p "$RUN_DIR"
    if is_running; then
      if config_changed; then
        echo "server config changed; restarting pid=$(cat "$PID_FILE")"
        kill "$(cat "$PID_FILE")" || true
        sleep 1
        rm -f "$PID_FILE"
      else
        rm -f "$CONFIG_FILE.tmp"
        echo "server running pid=$(cat "$PID_FILE")"
        wait_health "${NEUROVA_MAMBA3_SERVER_WAIT:-90}"
        exit 0
      fi
    fi
    write_config
    nohup python scripts/mamba3_chat_server.py \
      --host "$HOST" \
      --port "$PORT" \
      --mode "$MODE" \
      --tokenizer "$TOKENIZER" \
      --checkpoint "$CHECKPOINT" \
      --seq-len "$SEQ_LEN" \
      --max-new "$MAX_NEW" \
      --device cuda \
      --dtype "$DTYPE" \
      --decode-mode "$DECODE_MODE" \
      --top-k "$TOP_K" \
      --top-p "$TOP_P" \
      --temperature "$TEMP" \
      --repetition-penalty "$REPETITION_PENALTY" \
      --cache-parity-steps "$CACHE_PARITY_STEPS" \
      $(if [[ "$CUDA_GRAPH" == "1" ]]; then printf '%s' '--cuda-graph'; fi) \
      $(if [[ "$CACHE_PARITY_GUARD" != "1" ]]; then printf '%s' '--no-cache-parity-guard'; fi) \
      >"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    echo "server starting pid=$(cat "$PID_FILE") log=$LOG_FILE"
    wait_health "${NEUROVA_MAMBA3_SERVER_WAIT:-90}"
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PID_FILE")" || true
      sleep 1
    fi
    rm -f "$PID_FILE"
    echo "server stopped"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  logs)
    mkdir -p "$RUN_DIR"
    touch "$LOG_FILE"
    tail -n "${2:-80}" "$LOG_FILE"
    ;;
  status)
    if is_running; then
      echo "server running pid=$(cat "$PID_FILE")"
      python - <<PY || true
import json, urllib.request
url = "http://$HOST:$PORT/health"
try:
    with urllib.request.urlopen(url, timeout=1.5) as r:
        print(json.dumps(json.loads(r.read().decode("utf-8")), ensure_ascii=False, indent=2))
except Exception as exc:
    print(f"health failed: {exc}")
PY
    else
      echo "server stopped"
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs}" >&2
    exit 2
    ;;
esac
