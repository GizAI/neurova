#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TOKENIZER="${TOKENIZER:-llama31}"
MODES_CSV="${MODES_CSV:-mamba3-siso-hybrid-95m,mamba3-siso-hybrid-0.3b,siso,transformer-tiny,mimo-r2-fast-tiny,mimo-r4-fast-tiny}"
SEQ_LENS_CSV="${SEQ_LENS_CSV:-128,512}"
TOKEN_BUDGET="${TOKEN_BUDGET:-131072}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
LR="${LR:-1e-4}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
TRAIN_DATA="${TRAIN_DATA:-data/splits/base_general_recovery_strat_train.txt}"
VALID_DATA="${VALID_DATA:-data/splits/base_general_recovery_strat_valid.txt}"
RUN_ROOT="${RUN_ROOT:-runs/mamba3_arch_context_compare}"
EVAL_BATCHES="${EVAL_BATCHES:-16}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"
MAX_NEW="${MAX_NEW:-48}"

cd "${ROOT}"
mkdir -p "${RUN_ROOT}"

IFS=',' read -r -a MODES <<< "${MODES_CSV}"
IFS=',' read -r -a SEQ_LENS <<< "${SEQ_LENS_CSV}"

summary="${RUN_ROOT}/summary.jsonl"
: > "${summary}"

for seq_len in "${SEQ_LENS[@]}"; do
  for mode in "${MODES[@]}"; do
    tokens_per_step=$(( seq_len * BATCH_SIZE * GRAD_ACCUM_STEPS ))
    steps=$(( TOKEN_BUDGET / tokens_per_step ))
    if (( steps < 1 )); then
      steps=1
    fi
    run_dir="${RUN_ROOT}/${mode}_seq${seq_len}"
    ckpt="${run_dir}/base.pt"
    mkdir -p "${run_dir}"
    echo "== compare mode=${mode} seq_len=${seq_len} steps=${steps} token_budget=${TOKEN_BUDGET} =="

    train_ok=true
    python -m mamba3_kr.cli train-packed \
      --mode "${mode}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint "${ckpt}" \
      --data "${TRAIN_DATA}" \
      --steps "${steps}" \
      --lr "${LR}" \
      --save-every "${steps}" \
      --no-resume \
      --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
      --optimizer "${OPTIMIZER}" \
      --seq-len "${seq_len}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      > "${run_dir}/train.log" 2>&1 || train_ok=false

    if [[ "${train_ok}" != "true" ]]; then
      python - <<PY | tee -a "${summary}"
import json
from pathlib import Path
log = Path(${run_dir@Q}) / "train.log"
print(json.dumps({
    "mode": ${mode@Q},
    "seq_len": int(${seq_len@Q}),
    "steps": int(${steps@Q}),
    "checkpoint": ${ckpt@Q},
    "train_ok": False,
    "train_tail": log.read_text(errors="replace").splitlines()[-20:] if log.exists() else [],
}))
PY
      continue
    fi

    python -m mamba3_kr.cli eval-loss \
      --mode "${mode}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint "${ckpt}" \
      --data "${VALID_DATA}" \
      --batches "${EVAL_BATCHES}" \
      --seq-len "${seq_len}" \
      --batch-size "${BATCH_SIZE}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      > "${run_dir}/eval_loss.json" 2>&1 || true

    python -m mamba3_kr.cli quality-gate \
      --mode "${mode}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint "${ckpt}" \
      --max-new "${MAX_NEW}" \
      --seq-len "${seq_len}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --top-k 1 \
      --top-p 0 \
      --temperature 1.0 \
      --repetition-penalty 1.0 \
      > "${run_dir}/quality_gate.json" 2>&1 || true

    python -m mamba3_kr.cli bench-decode \
      --mode "${mode}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint "${ckpt}" \
      --prompt "The main idea is" \
      --max-new "${MAX_NEW}" \
      --repeats "${BENCH_REPEATS}" \
      --seq-len "${seq_len}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --top-k 1 \
      --top-p 0 \
      --temperature 1.0 \
      --repetition-penalty 1.0 \
      > "${run_dir}/bench_decode.json" 2>&1 || true

    python - <<PY | tee -a "${summary}"
import json
from pathlib import Path

run_dir = Path(${run_dir@Q})
payload = {
    "mode": ${mode@Q},
    "seq_len": int(${seq_len@Q}),
    "steps": int(${steps@Q}),
    "token_budget": int(${TOKEN_BUDGET@Q}),
    "checkpoint": ${ckpt@Q},
    "train_ok": True,
}
for name in ("eval_loss", "quality_gate", "bench_decode"):
    path = run_dir / f"{name}.json"
    if not path.exists():
        payload[name] = {"ok": False, "error": "missing"}
        continue
    raw = path.read_text(errors="replace")
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        try:
            payload[name] = json.loads(raw[start:end + 1])
            continue
        except Exception as exc:
            payload[name] = {"ok": False, "error": str(exc), "raw_tail": raw.splitlines()[-20:]}
            continue
    payload[name] = {"ok": False, "error": "no_json", "raw_tail": raw.splitlines()[-20:]}
print(json.dumps(payload, ensure_ascii=False))
PY
  done
done

echo "summary=${summary}"
