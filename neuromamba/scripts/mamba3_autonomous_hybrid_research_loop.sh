#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

TOKENIZER="${TOKENIZER:-llama31}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-neuromamba/runs/mamba3_siso_fast_0_3b_ds128_v3/model.pt}"
TRAIN_DATA="${TRAIN_DATA:-neuromamba/data/splits/no_cheat_knowledge_v1_train.jsonl}"
VALID_DATA="${VALID_DATA:-neuromamba/data/splits/no_cheat_knowledge_v1_valid.jsonl}"
PROGRAMMATIC_DATA="${PROGRAMMATIC_DATA:-neuromamba/data/mamba3_programmatic_curriculum_eval.jsonl}"
RUN_ROOT="${RUN_ROOT:-neuromamba/runs/mamba3_autonomous_hybrid_research/$(date -u +%Y%m%dT%H%M%SZ)}"
MODES_CSV="${MODES_CSV:-mamba3-siso-fast-0.3b-ds128,mamba3-siso-hybrid-0.3b}"
SEQ_LEN="${SEQ_LEN:-1024}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
STEPS="${STEPS:-2500}"
LR="${LR:-4e-6}"
SAVE_EVERY="${SAVE_EVERY:-500}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
EVAL_BATCHES="${EVAL_BATCHES:-64}"
MMLU_REDUX_LIMIT="${MMLU_REDUX_LIMIT:-100}"
PROGRAMMATIC_LIMIT="${PROGRAMMATIC_LIMIT:-128}"
QUALITY_MAX_NEW="${QUALITY_MAX_NEW:-64}"
BENCH_MAX_NEW="${BENCH_MAX_NEW:-96}"
BENCH_REPEATS="${BENCH_REPEATS:-3}"
START_MODE="${START_MODE:-mamba3-siso-fast-0.3b-ds128}"
SEED_POLICY="${SEED_POLICY:-same_mode_only}"

mkdir -p "${RUN_ROOT}" neuromamba/data/splits
summary="${RUN_ROOT}/summary.jsonl"
: > "${summary}"

echo "run_root=${RUN_ROOT}"
echo "summary=${summary}"

if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
  echo "missing BASE_CHECKPOINT=${BASE_CHECKPOINT}" >&2
  exit 2
fi

if [[ ! -f "${TRAIN_DATA}" || ! -f "${VALID_DATA}" ]]; then
  echo "missing no-cheat splits; building from neuromamba/data/no_cheat_knowledge_v1.jsonl" >&2
  python neuromamba/scripts/mamba3_make_splits.py \
    --inputs neuromamba/data/no_cheat_knowledge_v1.jsonl \
    --train-out "${TRAIN_DATA}" \
    --valid-out "${VALID_DATA}" \
    --valid-ratio 0.01 \
    --seed 20260613
fi

if [[ ! -f "${PROGRAMMATIC_DATA}" ]]; then
  python neuromamba/scripts/mamba3_generate_programmatic_curriculum.py \
    --out "${PROGRAMMATIC_DATA}" \
    --records 512 \
    --seed 20260614
fi

IFS=',' read -r -a MODES <<< "${MODES_CSV}"

for mode in "${MODES[@]}"; do
  mode_slug="${mode//[^A-Za-z0-9_.-]/_}"
  run_dir="${RUN_ROOT}/${mode_slug}"
  ckpt="${run_dir}/model.pt"
  mkdir -p "${run_dir}"

  echo "== candidate mode=${mode} run_dir=${run_dir} =="
  if [[ "${SEED_POLICY}" == "same_mode_only" && "${mode}" == "${START_MODE}" ]]; then
    cp "${BASE_CHECKPOINT}" "${ckpt}"
    echo "seeded ${mode} from ${BASE_CHECKPOINT}"
  elif [[ "${SEED_POLICY}" == "all" ]]; then
    cp "${BASE_CHECKPOINT}" "${ckpt}"
    echo "seeded ${mode} from ${BASE_CHECKPOINT}; load_or_new will skip incompatible payloads"
  else
    rm -f "${ckpt}"
    echo "fresh candidate ${mode}; no cross-architecture checkpoint transplant"
  fi

  train_ok=true
  python -m neuromamba.cli train-packed \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --data "${TRAIN_DATA}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --save-every "${SAVE_EVERY}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --optimizer "${OPTIMIZER}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --shuffle-texts \
    --data-seed 20260614 \
    > "${run_dir}/train.log" 2>&1 || train_ok=false

  if [[ "${train_ok}" != "true" ]]; then
    python - <<PY | tee -a "${summary}"
import json
from pathlib import Path
log = Path(${run_dir@Q}) / "train.log"
print(json.dumps({
    "mode": ${mode@Q},
    "checkpoint": ${ckpt@Q},
    "train_ok": False,
    "train_tail": log.read_text(errors="replace").splitlines()[-80:] if log.exists() else [],
}, ensure_ascii=False))
PY
    continue
  fi

  python -m neuromamba.cli eval-loss \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --data "${VALID_DATA}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --batches "${EVAL_BATCHES}" \
    > "${run_dir}/eval_loss.json" 2>&1 || true

  python -m neuromamba.cli quality-gate \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --max-new "${QUALITY_MAX_NEW}" \
    --seq-len 128 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 \
    > "${run_dir}/quality_gate.json" 2>&1 || true

  python neuromamba/scripts/mamba3_eval_programmatic.py \
    --data "${PROGRAMMATIC_DATA}" \
    --limit "${PROGRAMMATIC_LIMIT}" \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --seq-len 128 \
    --max-new 32 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    > "${run_dir}/programmatic_eval.json" 2>&1 || true

  python neuromamba/scripts/mamba3_eval_mcq_bench.py \
    --suite mmlu_redux \
    --mmlu-subject all \
    --redux-filter ok \
    --limit "${MMLU_REDUX_LIMIT}" \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --seq-len 128 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --out "${run_dir}/mmlu_redux.json" \
    > "${run_dir}/mmlu_redux.stdout" 2>&1 || true

  python -m neuromamba.cli bench-decode \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --prompt "Explain why clean data matters for language models." \
    --max-new "${BENCH_MAX_NEW}" \
    --repeats "${BENCH_REPEATS}" \
    --seq-len 128 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --top-k 1 \
    --top-p 0 \
    --temperature 1.0 \
    --repetition-penalty 1.0 \
    > "${run_dir}/bench_decode.json" 2>&1 || true

  python neuromamba/scripts/mamba3_recurrent_parity.py \
    --mode "${mode}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint "${ckpt}" \
    --seq-len 128 \
    --steps 12 \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    > "${run_dir}/recurrent_parity.json" 2>&1 || true

  python - <<PY | tee -a "${summary}"
import json
from pathlib import Path

run_dir = Path(${run_dir@Q})
payload = {
    "mode": ${mode@Q},
    "checkpoint": ${ckpt@Q},
    "train_ok": True,
    "steps": int(${STEPS@Q}),
    "seq_len": int(${SEQ_LEN@Q}),
    "batch_size": int(${BATCH_SIZE@Q}),
    "grad_accum_steps": int(${GRAD_ACCUM_STEPS@Q}),
    "lr": float(${LR@Q}),
    "seed_policy": ${SEED_POLICY@Q},
    "base_checkpoint": ${BASE_CHECKPOINT@Q},
    "start_mode": ${START_MODE@Q},
}
for name in ("eval_loss", "quality_gate", "programmatic_eval", "mmlu_redux", "bench_decode", "recurrent_parity"):
    path = run_dir / f"{name}.json"
    if not path.exists() and name == "mmlu_redux":
        path = run_dir / "mmlu_redux.json"
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
            payload[name] = {"ok": False, "error": str(exc), "raw_tail": raw.splitlines()[-40:]}
            continue
    payload[name] = {"ok": False, "error": "no_json", "raw_tail": raw.splitlines()[-40:]}
print(json.dumps(payload, ensure_ascii=False))
PY
done

python - <<PY | tee "${RUN_ROOT}/verdict.json"
import json
from pathlib import Path

summary = Path(${summary@Q})
rows = [json.loads(line) for line in summary.read_text().splitlines() if line.strip()]

def get(row, path, default=None):
    cur = row
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

verdicts = []
for row in rows:
    eval_loss = get(row, ("eval_loss", "loss"))
    redux = get(row, ("mmlu_redux", "choice_accuracy"), get(row, ("mmlu_redux", "accuracy")))
    prog = get(row, ("programmatic_eval", "accuracy"))
    toks = get(row, ("bench_decode", "tokens_per_sec"), get(row, ("bench_decode", "tok_per_sec")))
    quality_ok = bool(get(row, ("quality_gate", "ok"), False))
    parity_ok = bool(get(row, ("recurrent_parity", "ok"), False))
    score = 0.0
    if isinstance(redux, (int, float)):
        score += redux * 100.0
    if isinstance(prog, (int, float)):
        score += prog * 20.0
    if isinstance(eval_loss, (int, float)):
        score += max(0.0, 10.0 - eval_loss)
    if isinstance(toks, (int, float)):
        score += min(toks / 100.0, 20.0)
    if quality_ok:
        score += 5.0
    if parity_ok:
        score += 5.0
    verdicts.append({
        "mode": row.get("mode"),
        "checkpoint": row.get("checkpoint"),
        "score_for_ranking_only": round(score, 4),
        "eval_loss": eval_loss,
        "mmlu_redux_choice_accuracy": redux,
        "programmatic_accuracy": prog,
        "decode_tokens_per_sec": toks,
        "quality_ok": quality_ok,
        "recurrent_parity_ok": parity_ok,
        "promote_default": False,
        "reason": "research-only; MMLU-Redux is a held-out report metric, not a training target",
    })
verdicts.sort(key=lambda x: x["score_for_ranking_only"], reverse=True)
print(json.dumps({
    "run_root": ${RUN_ROOT@Q},
    "summary": ${summary@Q},
    "best_research_candidate": verdicts[0] if verdicts else None,
    "all_candidates": verdicts,
}, ensure_ascii=False, indent=2))
PY

echo "done run_root=${RUN_ROOT}"
