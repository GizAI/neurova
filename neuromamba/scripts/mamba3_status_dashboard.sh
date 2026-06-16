#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT}"

section() {
  printf '\n== %s ==\n' "$1"
}

section "chat server"
neuromamba/scripts/mamba3_chat_serverctl.sh status || true

section "current auto-apply"
neuromamba/scripts/mamba3_current_applyctl.sh status || true

section "teacher research"
neuromamba/scripts/mamba3_teacher_researchctl.sh status || true

section "siso intel ablation"
neuromamba/scripts/mamba3_siso_intel_ablationctl.sh status || true

if [[ "${MAMBA3_STATUS_FULL:-0}" == "1" ]]; then
  section "research autopilot"
  neuromamba/scripts/mamba3_research_autopilotctl.sh status || true

  section "autonomous hybrid research"
  neuromamba/scripts/mamba3_autonomous_hybrid_researchctl.sh status || true

  section "siso fast 0.3B intelligence training"
  neuromamba/scripts/mamba3_siso_fast_intel_trainctl.sh status || true

  section "siso hybrid 0.3B training"
  RUN_DIR="${ROOT}/neuromamba/runs/mamba3_siso_hybrid_0_3b_v1" neuromamba/scripts/mamba3_siso_hybrid_trainctl.sh status || true

  section "chat autoloop"
  neuromamba/scripts/mamba3_chat_autoloopctl.sh status || true

  section "mamba3 training"
  neuromamba/scripts/mamba3_moe24_trainctl.sh status || true

  section "chat training"
  neuromamba/scripts/mamba3_chat_trainctl.sh status || true

  section "speak training"
  neuromamba/scripts/mamba3_speak_trainctl.sh status || true

  section "latest decode tune"
  if [[ -f neuromamba/runs/mamba3_neurova_speak_v1/decode_tune/latest.json ]]; then
    python -c 'import json; p="neuromamba/runs/mamba3_neurova_speak_v1/decode_tune/latest.json"; d=json.load(open(p)); print(json.dumps({"path": p, "best": d.get("best")}, ensure_ascii=False, indent=2))'
  else
    echo "no decode tune result yet"
  fi
else
  section "archived research"
  echo "hidden by default; run MAMBA3_STATUS_FULL=1 ./neurova.sh mamba3 status for older experiments"
fi

section "gpu"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
