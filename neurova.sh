#!/usr/bin/env bash
# Neurova V6 — Quick start
# Usage:
#   bash neurova.sh                                # interactive Mamba-3 chat
#   bash neurova.sh [mode]                         # legacy V6: bf16 | 4bit
#   bash neurova.sh mamba3                         # interactive chat
#   bash neurova.sh mamba3 [prompt]                # one-shot prompt
#   bash neurova.sh mamba3 [chat|official|eval|bench|bench-mcq|bench-mmlu|bench-mmlu-redux|bench-suite|tune|status|diagnose|probe|serve|server-start|server-stop|server-restart|server-status|research-start|research-status|research-stop|research-logs|research-tail|research-hybrid-start|research-hybrid-status|research-hybrid-stop|research-hybrid-tail|teacher-research-start|teacher-research-status|teacher-research-stop|teacher-research-tail] [prompt]

cd "$(dirname "$0")"

if [[ $# -eq 0 ]]; then
  set -- mamba3
fi

if [[ "${1:-}" == "mamba3" ]]; then
  shift
  if [[ $# -eq 0 ]]; then
    ACTION="serve"
  else
    ACTION="${1:-chat}"
    case "$ACTION" in
      chat|turbo|official|eval|bench|bench-mcq|bench-mmlu|bench-mmlu-redux|bench-suite|tune|status|diagnose|probe|serve|server-start|server-stop|server-restart|server-status|research-start|research-status|research-stop|research-logs|research-tail|research-hybrid-start|research-hybrid-status|research-hybrid-stop|research-hybrid-tail|teacher-research-start|teacher-research-status|teacher-research-stop|teacher-research-tail)
        shift || true
        ;;
      *)
        ACTION="chat"
        ;;
    esac
  fi

  HOST="${NEUROVA_MAMBA3_HOST:-ml-dmc8}"
  ROOT="${NEUROVA_MAMBA3_ROOT:-/home/user/workspace/neurova}"
  ENV_NAME="${NEUROVA_MAMBA3_ENV:-mamba3_siso}"
  MODE="${NEUROVA_MAMBA3_MODE:-mamba3-siso-fast-0.3b-ds128}"
  TOKENIZER="${NEUROVA_MAMBA3_TOKENIZER:-llama31}"
  CHECKPOINT="${NEUROVA_MAMBA3_CHECKPOINT:-runs/mamba3_current/model.pt}"
  FALLBACK_CHECKPOINT="${NEUROVA_MAMBA3_FALLBACK_CHECKPOINT:-runs/mamba3_current_training_chat/model.pt}"
  SECOND_FALLBACK_CHECKPOINT="${NEUROVA_MAMBA3_SECOND_FALLBACK_CHECKPOINT:-runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
  SEQ_LEN="${NEUROVA_MAMBA3_SEQ:-128}"
  MAX_NEW="${NEUROVA_MAMBA3_MAX_NEW:-512}"
  DTYPE="${NEUROVA_MAMBA3_DTYPE:-bf16}"
  SERVER_HOST="${NEUROVA_MAMBA3_SERVER_HOST:-127.0.0.1}"
  SERVER_PORT="${NEUROVA_MAMBA3_SERVER_PORT:-8765}"
  SERVER_RUN_DIR="${NEUROVA_MAMBA3_SERVER_RUN_DIR:-runs/mamba3_chat_server}"
  USE_SERVER="${NEUROVA_MAMBA3_USE_SERVER:-1}"
  DECODE_MODE="${NEUROVA_MAMBA3_DECODE_MODE:-cache}"
  if [[ -n "${NEUROVA_MAMBA3_CUDA_GRAPH:-}" ]]; then
    CUDA_GRAPH="${NEUROVA_MAMBA3_CUDA_GRAPH}"
  elif [[ "$SEQ_LEN" -le 128 ]]; then
    CUDA_GRAPH="1"
  else
    CUDA_GRAPH="0"
  fi
  CACHE_PARITY_GUARD="${NEUROVA_MAMBA3_CACHE_PARITY_GUARD:-0}"
  PROMPT="${*:-The main idea is}"
  if [[ "$ACTION" == "turbo" ]]; then
    ACTION="chat"
    SEQ_LEN="${NEUROVA_MAMBA3_TURBO_SEQ:-128}"
    MAX_NEW="${NEUROVA_MAMBA3_TURBO_MAX_NEW:-128}"
    CUDA_GRAPH="${NEUROVA_MAMBA3_TURBO_CUDA_GRAPH:-1}"
  fi
  LONG_CONTEXT_THRESHOLD_CHARS="${NEUROVA_MAMBA3_LONG_CONTEXT_THRESHOLD_CHARS:-2048}"
  if [[ "$ACTION" == "chat" && ${#PROMPT} -gt "$LONG_CONTEXT_THRESHOLD_CHARS" ]]; then
    SEQ_LEN="${NEUROVA_MAMBA3_LONG_SEQ:-16384}"
    MAX_NEW="${NEUROVA_MAMBA3_LONG_MAX_NEW:-512}"
    CUDA_GRAPH="${NEUROVA_MAMBA3_LONG_CUDA_GRAPH:-0}"
    SERVER_PORT="${NEUROVA_MAMBA3_LONG_SERVER_PORT:-8767}"
    SERVER_RUN_DIR="${NEUROVA_MAMBA3_LONG_SERVER_RUN_DIR:-runs/mamba3_chat_server_long}"
  fi
  if [[ "$ACTION" == "official" ]]; then
    ACTION="chat"
    DECODE_MODE="cache-verify"
  fi
  DEFAULT_STREAM="0"
  if [[ "$ACTION" == "serve" ]]; then
    DEFAULT_STREAM="1"
  fi
  STREAM_FLAG="--stream"
  if [[ "${NEUROVA_MAMBA3_STREAM:-$DEFAULT_STREAM}" == "0" ]]; then
    STREAM_FLAG=""
  fi

  q() { printf "%q" "$1"; }
  remote_exec() {
    local remote_cmd
    remote_cmd="cd $(q "$ROOT") && source ~/miniconda3/etc/profile.d/conda.sh && conda activate $(q "$ENV_NAME") && $*"
    if [[ "$HOST" == "local" || "$HOST" == "$(hostname)" ]]; then
      bash -lc "$remote_cmd"
    elif [[ "$ACTION" == "serve" && -t 0 ]]; then
      ssh -tt "$HOST" "bash -lc $(q "$remote_cmd")"
    else
      ssh "$HOST" "bash -lc $(q "$remote_cmd")"
    fi
  }

  PICK_CHECKPOINT='CHECKPOINT='"$(q "$CHECKPOINT")"'; if [[ ! -f "$CHECKPOINT" && -f '"$(q "$FALLBACK_CHECKPOINT")"' ]]; then CHECKPOINT='"$(q "$FALLBACK_CHECKPOINT")"'; fi; if [[ ! -f "$CHECKPOINT" && -f '"$(q "$SECOND_FALLBACK_CHECKPOINT")"' ]]; then CHECKPOINT='"$(q "$SECOND_FALLBACK_CHECKPOINT")"'; fi'
  COMMON="--mode $(q "$MODE") --tokenizer $(q "$TOKENIZER") --checkpoint \"\${CHECKPOINT}\" --seq-len $(q "$SEQ_LEN") --device cuda --dtype $(q "$DTYPE")"
  CUDA_GRAPH_FLAG="--cuda-graph"
  if [[ "$SEQ_LEN" -gt 128 ]]; then
    CUDA_GRAPH_FLAG=""
  fi
  DECODE="--top-k ${NEUROVA_MAMBA3_TOP_K:-40} --top-p ${NEUROVA_MAMBA3_TOP_P:-0.9} --temperature ${NEUROVA_MAMBA3_TEMP:-0.8} --repetition-penalty ${NEUROVA_MAMBA3_REPETITION_PENALTY:-1.15} $CUDA_GRAPH_FLAG"

  case "$ACTION" in
    chat)
      if [[ "$USE_SERVER" == "1" ]]; then
        CLIENT_STREAM_FLAG=""
        if [[ "${NEUROVA_MAMBA3_STREAM:-$DEFAULT_STREAM}" == "0" ]]; then
          CLIENT_STREAM_FLAG="--no-stream"
        fi
        remote_exec "$PICK_CHECKPOINT; export NEUROVA_MAMBA3_CHECKPOINT=\"\$CHECKPOINT\" NEUROVA_MAMBA3_SERVER_HOST=$(q "$SERVER_HOST") NEUROVA_MAMBA3_SERVER_PORT=$(q "$SERVER_PORT") NEUROVA_MAMBA3_SERVER_RUN_DIR=$(q "$SERVER_RUN_DIR") NEUROVA_MAMBA3_MODE=$(q "$MODE") NEUROVA_MAMBA3_TOKENIZER=$(q "$TOKENIZER") NEUROVA_MAMBA3_SEQ=$(q "$SEQ_LEN") NEUROVA_MAMBA3_MAX_NEW=$(q "$MAX_NEW") NEUROVA_MAMBA3_DTYPE=$(q "$DTYPE") NEUROVA_MAMBA3_DECODE_MODE=$(q "$DECODE_MODE") NEUROVA_MAMBA3_CUDA_GRAPH=$(q "$CUDA_GRAPH") NEUROVA_MAMBA3_CACHE_PARITY_GUARD=$(q "$CACHE_PARITY_GUARD"); scripts/mamba3_infer_guard.sh run scripts/mamba3_chat_request.sh --prompt $(q "$PROMPT") $CLIENT_STREAM_FLAG"
      else
        remote_exec "$PICK_CHECKPOINT; scripts/mamba3_infer_guard.sh run python scripts/mamba3_safe_chat.py $COMMON --prompt $(q "$PROMPT") --max-new $(q "$MAX_NEW") $STREAM_FLAG"
      fi
      ;;
    eval)
      remote_exec "$PICK_CHECKPOINT; python -m mamba3_kr.cli eval-english $COMMON --max-new $(q "$MAX_NEW") $DECODE"
      ;;
    bench)
      remote_exec "$PICK_CHECKPOINT; python -m mamba3_kr.cli bench-decode $COMMON --prompt $(q "$PROMPT") --max-new $(q "$MAX_NEW") --repeats 3 $DECODE"
      ;;
    bench-mcq)
      remote_exec "$PICK_CHECKPOINT; scripts/mamba3_exclusive_gpu_guard.sh run python scripts/mamba3_eval_mcq_bench.py --suite smoke --mode $(q "$MODE") --tokenizer $(q "$TOKENIZER") --checkpoint \"\$CHECKPOINT\" --seq-len $(q "$SEQ_LEN") --device cuda --dtype $(q "$DTYPE") --out runs/mamba3_benchmarks/latest_mcq_smoke.json"
      ;;
    bench-mmlu)
      remote_exec "$PICK_CHECKPOINT; scripts/mamba3_exclusive_gpu_guard.sh run python scripts/mamba3_eval_mcq_bench.py --suite mmlu --mmlu-subject ${NEUROVA_MMLU_SUBJECT:-all} --limit ${NEUROVA_MMLU_LIMIT:-100} --mode $(q "$MODE") --tokenizer $(q "$TOKENIZER") --checkpoint \"\$CHECKPOINT\" --seq-len $(q "$SEQ_LEN") --device cuda --dtype $(q "$DTYPE") --out runs/mamba3_benchmarks/latest_mmlu.json"
      ;;
    bench-mmlu-redux)
      remote_exec "$PICK_CHECKPOINT; scripts/mamba3_exclusive_gpu_guard.sh run python scripts/mamba3_eval_mcq_bench.py --suite mmlu_redux --mmlu-subject ${NEUROVA_MMLU_SUBJECT:-all} --redux-filter ${NEUROVA_MMLU_REDUX_FILTER:-ok} --limit ${NEUROVA_MMLU_REDUX_LIMIT:-100} --mode $(q "$MODE") --tokenizer $(q "$TOKENIZER") --checkpoint \"\$CHECKPOINT\" --seq-len $(q "$SEQ_LEN") --device cuda --dtype $(q "$DTYPE") --out runs/mamba3_benchmarks/latest_mmlu_redux.json"
      ;;
    bench-suite)
      remote_exec "$PICK_CHECKPOINT; scripts/mamba3_exclusive_gpu_guard.sh run env MODE=$(q "$MODE") TOKENIZER=$(q "$TOKENIZER") CHECKPOINT=\"\$CHECKPOINT\" SEQ_LEN=$(q "$SEQ_LEN") MMLU_LIMIT=${NEUROVA_MMLU_LIMIT:-100} MMLU_REDUX_LIMIT=${NEUROVA_MMLU_REDUX_LIMIT:-100} MMLU_SUBJECT=${NEUROVA_MMLU_SUBJECT:-all} MMLU_REDUX_FILTER=${NEUROVA_MMLU_REDUX_FILTER:-ok} scripts/mamba3_benchmark_suite.sh"
      ;;
    tune)
      remote_exec "$PICK_CHECKPOINT; python scripts/mamba3_decode_tune.py $COMMON --max-new $(q "$MAX_NEW")"
      ;;
    status)
      remote_exec "scripts/mamba3_status_dashboard.sh"
      ;;
    diagnose)
      remote_exec "$PICK_CHECKPOINT; python -m mamba3_kr.cli diagnose-decode $COMMON --prompt $(q "$PROMPT")"
      ;;
    probe)
      remote_exec "$PICK_CHECKPOINT; python -m mamba3_kr.cli probe-kernel $COMMON --batch-size 1 --data data/english_completion_bootstrap.txt data/english_instruction_bootstrap.txt"
      ;;
    serve)
      if [[ "$USE_SERVER" == "1" ]]; then
        remote_exec "$PICK_CHECKPOINT; export NEUROVA_MAMBA3_CHECKPOINT=\"\$CHECKPOINT\" NEUROVA_MAMBA3_SERVER_HOST=$(q "$SERVER_HOST") NEUROVA_MAMBA3_SERVER_PORT=$(q "$SERVER_PORT") NEUROVA_MAMBA3_SERVER_RUN_DIR=$(q "$SERVER_RUN_DIR") NEUROVA_MAMBA3_MODE=$(q "$MODE") NEUROVA_MAMBA3_TOKENIZER=$(q "$TOKENIZER") NEUROVA_MAMBA3_SEQ=$(q "$SEQ_LEN") NEUROVA_MAMBA3_MAX_NEW=$(q "$MAX_NEW") NEUROVA_MAMBA3_DTYPE=$(q "$DTYPE") NEUROVA_MAMBA3_DECODE_MODE=$(q "$DECODE_MODE") NEUROVA_MAMBA3_CUDA_GRAPH=$(q "$CUDA_GRAPH") NEUROVA_MAMBA3_CACHE_PARITY_GUARD=$(q "$CACHE_PARITY_GUARD"); scripts/mamba3_chat_repl.sh"
      else
        remote_exec "$PICK_CHECKPOINT; python scripts/mamba3_safe_chat.py $COMMON --max-new $(q "$MAX_NEW") $STREAM_FLAG"
      fi
      ;;
    server-start)
      remote_exec "$PICK_CHECKPOINT; export NEUROVA_MAMBA3_CHECKPOINT=\"\$CHECKPOINT\" NEUROVA_MAMBA3_SERVER_HOST=$(q "$SERVER_HOST") NEUROVA_MAMBA3_SERVER_PORT=$(q "$SERVER_PORT") NEUROVA_MAMBA3_SERVER_RUN_DIR=$(q "$SERVER_RUN_DIR") NEUROVA_MAMBA3_MODE=$(q "$MODE") NEUROVA_MAMBA3_TOKENIZER=$(q "$TOKENIZER") NEUROVA_MAMBA3_SEQ=$(q "$SEQ_LEN") NEUROVA_MAMBA3_MAX_NEW=$(q "$MAX_NEW") NEUROVA_MAMBA3_DTYPE=$(q "$DTYPE") NEUROVA_MAMBA3_DECODE_MODE=$(q "$DECODE_MODE") NEUROVA_MAMBA3_CUDA_GRAPH=$(q "$CUDA_GRAPH") NEUROVA_MAMBA3_CACHE_PARITY_GUARD=$(q "$CACHE_PARITY_GUARD"); scripts/mamba3_infer_guard.sh run scripts/mamba3_chat_serverctl.sh start"
      ;;
    server-stop)
      remote_exec "NEUROVA_MAMBA3_SERVER_HOST=$(q "$SERVER_HOST") NEUROVA_MAMBA3_SERVER_PORT=$(q "$SERVER_PORT") scripts/mamba3_chat_serverctl.sh stop"
      ;;
    server-restart)
      remote_exec "$PICK_CHECKPOINT; export NEUROVA_MAMBA3_CHECKPOINT=\"\$CHECKPOINT\" NEUROVA_MAMBA3_SERVER_HOST=$(q "$SERVER_HOST") NEUROVA_MAMBA3_SERVER_PORT=$(q "$SERVER_PORT") NEUROVA_MAMBA3_SERVER_RUN_DIR=$(q "$SERVER_RUN_DIR") NEUROVA_MAMBA3_MODE=$(q "$MODE") NEUROVA_MAMBA3_TOKENIZER=$(q "$TOKENIZER") NEUROVA_MAMBA3_SEQ=$(q "$SEQ_LEN") NEUROVA_MAMBA3_MAX_NEW=$(q "$MAX_NEW") NEUROVA_MAMBA3_DTYPE=$(q "$DTYPE") NEUROVA_MAMBA3_DECODE_MODE=$(q "$DECODE_MODE") NEUROVA_MAMBA3_CUDA_GRAPH=$(q "$CUDA_GRAPH") NEUROVA_MAMBA3_CACHE_PARITY_GUARD=$(q "$CACHE_PARITY_GUARD"); scripts/mamba3_infer_guard.sh run scripts/mamba3_chat_serverctl.sh restart"
      ;;
    server-status)
      remote_exec "NEUROVA_MAMBA3_SERVER_HOST=$(q "$SERVER_HOST") NEUROVA_MAMBA3_SERVER_PORT=$(q "$SERVER_PORT") scripts/mamba3_chat_serverctl.sh status"
      ;;
    research-start)
      remote_exec "GPU_POLICY=${NEUROVA_RESEARCH_GPU_POLICY:-train_priority} INTERVAL=${NEUROVA_RESEARCH_INTERVAL:-120} scripts/mamba3_research_autopilotctl.sh start"
      ;;
    research-status)
      remote_exec "scripts/mamba3_research_autopilotctl.sh status"
      ;;
    research-stop)
      remote_exec "scripts/mamba3_research_autopilotctl.sh stop"
      ;;
    research-logs)
      remote_exec "scripts/mamba3_research_autopilotctl.sh logs"
      ;;
    research-tail)
      remote_exec "scripts/mamba3_research_autopilotctl.sh tail"
      ;;
    research-hybrid-start)
      remote_exec "STEPS=${NEUROVA_HYBRID_STEPS:-2500} MMLU_REDUX_LIMIT=${NEUROVA_HYBRID_MMLU_REDUX_LIMIT:-100} MODES_CSV=${NEUROVA_HYBRID_MODES:-mamba3-siso-fast-0.3b-ds128,mamba3-siso-hybrid-0.3b} scripts/mamba3_autonomous_hybrid_researchctl.sh start"
      ;;
    research-hybrid-status)
      remote_exec "scripts/mamba3_autonomous_hybrid_researchctl.sh status"
      ;;
    research-hybrid-stop)
      remote_exec "scripts/mamba3_autonomous_hybrid_researchctl.sh stop"
      ;;
    research-hybrid-tail)
      remote_exec "scripts/mamba3_autonomous_hybrid_researchctl.sh tail"
      ;;
    teacher-research-start)
      remote_exec "MAX_ROUNDS=${NEUROVA_TEACHER_MAX_ROUNDS:-3} BOOTSTRAP_EXISTING=${NEUROVA_TEACHER_BOOTSTRAP_EXISTING:-1} DEEPSEEK_RECORDS=${NEUROVA_TEACHER_RECORDS:-20000} DEEPSEEK_RECORDS_CSV=${NEUROVA_TEACHER_RECORDS_CSV:-3000,12000,30000} DEEPSEEK_BATCH_SIZE=${NEUROVA_TEACHER_BATCH_SIZE:-24} MCQ_STEPS=${NEUROVA_TEACHER_MCQ_STEPS:-2500} MMLU_REDUX_LIMIT=${NEUROVA_TEACHER_MMLU_REDUX_LIMIT:-200} CHAT_REPAIR=${NEUROVA_TEACHER_CHAT_REPAIR:-1} CHAT_STEPS=${NEUROVA_TEACHER_CHAT_STEPS:-1200} scripts/mamba3_teacher_researchctl.sh start"
      ;;
    teacher-research-status)
      remote_exec "scripts/mamba3_teacher_researchctl.sh status"
      ;;
    teacher-research-stop)
      remote_exec "scripts/mamba3_teacher_researchctl.sh stop"
      ;;
    teacher-research-tail)
      remote_exec "scripts/mamba3_teacher_researchctl.sh tail"
      ;;
  esac
  exit $?
fi

MODE="${1:-bf16}"
export V6_MODE="$MODE"
exec python3 neurova_v6.py
