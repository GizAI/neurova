#!/usr/bin/env python3
"""Summarize the 2.4B MoE 100M-token gate and print the next action.

This script is intentionally conservative: it does not promote checkpoints or
start new training. It converts the gate logs into an explicit decision so the
long-running background job can be managed without relying on memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Decision:
    status: str
    action: str
    rationale: list[str]
    required_next_steps: list[str]


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "loss" in payload:
            records.append(payload)
    return records


def summarize(records: list[dict[str, Any]], planned_tokens: int) -> dict[str, Any]:
    losses = [float(item["loss"]) for item in records]
    passed = [item for item in records if item.get("passed") is True]
    collapsed = [item for item in records if item.get("collapsed") is True]
    rounds = [int(item.get("round", idx + 1)) for idx, item in enumerate(records)]
    latest = records[-1] if records else {}
    latest_round = int(latest.get("round", rounds[-1] if rounds else 0) or 0)
    steps_per_round = int(latest.get("steps_per_round", 2000) or 2000)
    seq_len = int(latest.get("seq_len", 2048) or 2048)
    batch_size = int(latest.get("batch_size", 1) or 1)
    trained_tokens = latest_round * steps_per_round * seq_len * batch_size
    return {
        "records": len(records),
        "latest_round": latest_round,
        "trained_tokens_estimate": trained_tokens,
        "planned_tokens": planned_tokens,
        "progress_pct_estimate": round(trained_tokens / planned_tokens * 100, 2) if planned_tokens else 0.0,
        "loss_first": losses[0] if losses else None,
        "loss_latest": losses[-1] if losses else None,
        "loss_best": min(losses) if losses else None,
        "loss_delta_first_to_latest": (losses[-1] - losses[0]) if len(losses) >= 2 else None,
        "passed_records": len(passed),
        "collapsed_records": len(collapsed),
        "latest_collapsed": latest.get("collapsed") if latest else None,
        "latest_distinct_words": latest.get("distinct_words") if latest else None,
        "latest_longest_repeated_word_run": latest.get("longest_repeated_word_run") if latest else None,
    }


def make_decision(summary: dict[str, Any], min_loss_for_base_gate: float) -> Decision:
    records = int(summary["records"])
    latest_loss = summary["loss_latest"]
    loss_delta = summary["loss_delta_first_to_latest"]
    latest_collapsed = summary["latest_collapsed"]
    progress = float(summary["progress_pct_estimate"])

    if records == 0:
        return Decision(
            status="no_gate_records",
            action="keep_or_resume_current_100m_block",
            rationale=["No held-out loss/collapse gate records were found."],
            required_next_steps=[
                "Verify the controller log path.",
                "Keep the watchdog active so training resumes from the latest checkpoint if interrupted.",
            ],
        )

    if progress < 99.0:
        return Decision(
            status="in_progress",
            action="do_not_change_architecture_yet",
            rationale=[
                f"Only about {progress:.2f}% of the planned 100M-token diagnostic block is complete.",
                "Changing architecture mid-block would destroy the value of the collapse/loss trend.",
            ],
            required_next_steps=[
                "Let the current 100M-token block finish.",
                "Keep SFT, QA templates, and promotion blocked until raw continuation is collapse-free.",
            ],
        )

    if latest_loss is not None and latest_loss <= min_loss_for_base_gate and latest_collapsed is False:
        return Decision(
            status="base_gate_passed",
            action="extend_base_to_300m_then_1b_tokens_before_sft",
            rationale=[
                f"Latest held-out loss {latest_loss:.4f} is below the base gate {min_loss_for_base_gate:.4f}.",
                "The latest decode gate is collapse-free.",
            ],
            required_next_steps=[
                "Continue raw document base training to 300M local tokens.",
                "Then continue to 1B local tokens if validation and collapse gates keep improving.",
                "Run the full eval matrix before any SFT.",
            ],
        )

    if latest_loss is not None and latest_loss < 8.5 and latest_collapsed is False:
        return Decision(
            status="collapse_free_but_undertrained",
            action="extend_base_training_not_sft",
            rationale=[
                f"Latest held-out loss {latest_loss:.4f} is improved but still above the strict base gate.",
                "Collapse appears resolved, so the correct next move is more clean base tokens.",
            ],
            required_next_steps=[
                "Extend to 300M local tokens with the same raw document objective.",
                "Add downstream evals, but keep chat/SFT separate until base loss and continuation quality are stronger.",
            ],
        )

    if latest_collapsed is True and loss_delta is not None and loss_delta < 0:
        return Decision(
            status="loss_down_but_collapse_persists",
            action="stop_blind_continuation_after_100m_and_diagnose_active_compute",
            rationale=[
                f"Loss improved by {loss_delta:.4f}, but the latest continuation is still collapsed.",
                "The current 2.4B total/top-1 MoE path should not be treated as dense 2.4B intelligence.",
            ],
            required_next_steps=[
                "Measure router/expert usage histogram, router entropy, and expert skew.",
                "Add or test load-balancing auxiliary loss before longer MoE continuation.",
                "Separate model collapse from greedy decode collapse with full-forward, sampling, and recurrent-parity probes.",
                "Compare against higher-active-compute baselines: dense-ish 900M/1.3B, trainable/offloaded dense 1.3B-2.7B, and top-2/top-4 load-balanced MoE.",
                "Preserve optimizer state if feasible through smaller active models, offload, or ZeRO rather than repeated weight-only Adam resets.",
            ],
        )

    return Decision(
        status="loss_plateau_or_unclear",
        action="run_diagnostic_matrix_before_more_tokens",
        rationale=[
            "The 100M block did not produce a clean base-model gate pass.",
            "More tokens without an active-compute/router/decode diagnosis risks extending a broken run.",
        ],
        required_next_steps=[
            "Run perplexity, continuation, repetition, copy/retrieval, JSON extraction, math/code, and long-context probes.",
            "Choose the next architecture by active parameters and verified training stability, not by total sparse parameters.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/mamba3_clean_doc_base_moe24_v1/until_gate/summary.jsonl"),
    )
    parser.add_argument("--planned-tokens", type=int, default=102_400_000)
    parser.add_argument("--min-loss-for-base-gate", type=float, default=5.0)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    records = load_records(args.summary)
    summary = summarize(records, args.planned_tokens)
    decision = make_decision(summary, args.min_loss_for_base_gate)
    payload = {
        "summary": summary,
        "decision": {
            "status": decision.status,
            "action": decision.action,
            "rationale": decision.rationale,
            "required_next_steps": decision.required_next_steps,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("== 100M decision summary ==")
    for key, value in summary.items():
        print(f"{key}={value}")
    print("== decision ==")
    print(f"status={decision.status}")
    print(f"action={decision.action}")
    print("rationale:")
    for item in decision.rationale:
        print(f"- {item}")
    print("required_next_steps:")
    for item in decision.required_next_steps:
        print(f"- {item}")


if __name__ == "__main__":
    main()
