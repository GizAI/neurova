#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TOKENIZER="${TOKENIZER:-llama31}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bf16}"
DATA="${DATA:-neuromamba/data/splits/base_doc_cont_train.txt}"
OUT="${OUT:-neuromamba/runs/mamba3_train_grid_probe.jsonl}"
STEPS="${STEPS:-3}"
LR="${LR:-5e-5}"
OPTIMIZER="${OPTIMIZER:-adamw8bit}"
MODES_CSV="${MODES_CSV:-mimo-r4-moe-520m,mimo-r4-moe-900m,mimo-r4-moe-1.1b,mimo-r4-moe-1.3b,mimo-r4-moe-1.7b,mimo-r4-moe-2.1b,mimo-r4-moe-2.3b,mimo-r4-moe-2.4b,mimo-r4-moe-2.5b,mimo-r4-moe-2.9b,mimo-r4-440m,mimo-r4-880m,mimo-r4-1.5b}"
SEQ_LENS_CSV="${SEQ_LENS_CSV:-512,1024,2048}"
BATCH_SIZES_CSV="${BATCH_SIZES_CSV:-1,2,3,4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MIN_FREE_MB_AFTER="${MIN_FREE_MB_AFTER:-1200}"

cd "${ROOT}"
mkdir -p "$(dirname "${OUT}")"
: > "${OUT}"

IFS=',' read -r -a MODES <<< "${MODES_CSV}"
IFS=',' read -r -a SEQ_LENS <<< "${SEQ_LENS_CSV}"
IFS=',' read -r -a BATCH_SIZES <<< "${BATCH_SIZES_CSV}"

for mode in "${MODES[@]}"; do
  for seq_len in "${SEQ_LENS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      ckpt="neuromamba/runs/_probe_${mode}_s${seq_len}_b${batch_size}.pt"
      rm -f "${ckpt}"
      echo "== probe mode=${mode} seq_len=${seq_len} batch_size=${batch_size} =="
      before_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
      set +e
      output=$(
        python -m neuromamba.cli train-packed \
          --mode "${mode}" \
          --tokenizer "${TOKENIZER}" \
          --checkpoint "${ckpt}" \
          --data "${DATA}" \
          --steps "${STEPS}" \
          --lr "${LR}" \
          --save-every 0 \
          --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
          --optimizer "${OPTIMIZER}" \
          --seq-len "${seq_len}" \
          --batch-size "${batch_size}" \
          --device "${DEVICE}" \
          --dtype "${DTYPE}" \
          --no-resume 2>&1
      )
      status=$?
      set -e
      after_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
      printf '%s\n' "${output}"
      PROBE_OUTPUT="${output}" python - "${OUT}" "${mode}" "${seq_len}" "${batch_size}" "${status}" "${before_free}" "${after_free}" "${MIN_FREE_MB_AFTER}" <<'PY'
import json, re, sys, time
import os

out, mode, seq_len, batch_size, status, before_free, after_free, min_free = sys.argv[1:]
text = os.environ.get("PROBE_OUTPUT", "")
steps = []
for line in text.splitlines():
    m = re.search(r"step=(\d+) loss=([0-9.]+) tok_s=([0-9.]+) peak_vram_gb=([0-9.]+)", line)
    if m:
        steps.append({
            "step": int(m.group(1)),
            "loss": float(m.group(2)),
            "tok_s": float(m.group(3)),
            "peak_vram_gb": float(m.group(4)),
        })
payload = {
    "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "mode": mode,
    "seq_len": int(seq_len),
    "batch_size": int(batch_size),
    "status": int(status),
    "ok": int(status) == 0,
    "before_free_mb": int(before_free or 0),
    "after_free_mb": int(after_free or 0),
    "min_free_mb_after": int(min_free),
    "steps": steps,
    "last_tok_s": steps[-1]["tok_s"] if steps else None,
    "peak_vram_gb": max((s["peak_vram_gb"] for s in steps), default=None),
    "error_tail": "\n".join(text.splitlines()[-20:]) if int(status) != 0 else "",
}
payload["fits_margin"] = payload["ok"] and payload["after_free_mb"] >= payload["min_free_mb_after"]
with open(out, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY
      rm -f "${ckpt}"
      if [[ "${status}" -ne 0 ]]; then
        echo "probe failed; continuing"
      fi
    done
  done
done

echo "probe_log=${OUT}"
