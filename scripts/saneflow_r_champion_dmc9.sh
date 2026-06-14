#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

profile="dmc9-r-champion-delta-landmark-long"
champion_key="D_delta_landmark"
decision_path="runs/saneflow_r_champion/decision.json"
report_path="runs/saneflow_r_ablation/report.json"
run_path="runs/saneflow_r_champion/d_delta_landmark_long"

python scripts/saneflow_r_ablation_report.py >/tmp/saneflow_r_ablation_report_latest.json

python - <<'PY'
import json
from pathlib import Path

report_path = Path("runs/saneflow_r_ablation/report.json")
decision_path = Path("runs/saneflow_r_champion/decision.json")
report = json.loads(report_path.read_text(encoding="utf-8"))
losses = {
    name: row["best_valid_loss"]
    for name, row in report.items()
    if row.get("best_valid_loss") is not None
}
if not losses:
    raise SystemExit("no completed R ablation valid losses found")
winner = min(losses, key=losses.get)
decision = {
    "winner": winner,
    "winner_valid_loss": losses[winner],
    "expected_winner": "D_delta_landmark",
    "action": "continue champion pretrain, no SFT",
    "profile": "dmc9-r-champion-delta-landmark-long",
    "source_checkpoint": "runs/saneflow_r_ablation/d_delta_landmark/model.pt",
    "output": "runs/saneflow_r_champion/d_delta_landmark_long",
    "losses": losses,
}
decision_path.parent.mkdir(parents=True, exist_ok=True)
decision_path.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(decision, indent=2, ensure_ascii=False))
if winner != "D_delta_landmark":
    raise SystemExit(f"winner is {winner}, not D_delta_landmark; refusing automatic continuation")
PY

if pgrep -af "${run_path}" >/dev/null; then
  echo "champion run already active: ${run_path}"
  exit 0
fi

if [[ -f "${run_path}/model.pt" ]]; then
  echo "champion run already completed: ${run_path}/model.pt"
  exit 0
fi

python scripts/saneflow_run.py start "${profile}"
