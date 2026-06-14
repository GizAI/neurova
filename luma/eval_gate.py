from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def memory_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False, "pass": False, "reason": "missing memory eval"}
    results = {row.get("ablation"): row for row in payload.get("results", [])}
    normal = float(results.get("normal", {}).get("overall", 0.0))
    no_slots = float(results.get("no_slots", {}).get("overall", 0.0))
    random_keys = float(results.get("random_slot_keys", {}).get("overall", 0.0))
    no_copy = float(results.get("no_copy", {}).get("overall", 0.0))
    no_slots_no_copy = float(results.get("no_slots_no_copy", {}).get("overall", 0.0))
    no_local_attention = float(results.get("no_local_attention", {}).get("overall", 0.0))
    best_ablation = max(no_slots, random_keys, no_copy, no_slots_no_copy, no_local_attention)
    gap = normal - best_ablation
    by_task = results.get("normal", {}).get("by_task", {})
    copy_acc = float(by_task.get("copy", {}).get("accuracy", 0.0))
    recall_acc = float(by_task.get("recall", {}).get("accuracy", 0.0))
    json_acc = float(by_task.get("json_field", {}).get("accuracy", 0.0))
    passed = copy_acc >= 0.50 and recall_acc >= 0.60 and json_acc >= 0.70 and gap >= 0.20
    return {
        "available": True,
        "pass": passed,
        "normal": normal,
        "no_slots": no_slots,
        "random_slot_keys": random_keys,
        "no_copy": no_copy,
        "no_slots_no_copy": no_slots_no_copy,
        "no_local_attention": no_local_attention,
        "ablation_gap": gap,
        "copy": copy_acc,
        "recall": recall_acc,
        "json_field": json_acc,
    }


def chat_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False, "pass": False, "reason": "missing chat eval"}
    pass_rate = float(payload.get("pass_rate", 0.0))
    bad_repetition = any(int(row.get("repeat4_max", 0)) >= 5 for row in payload.get("rows", []))
    replacement = any(bool(row.get("replacement_char")) for row in payload.get("rows", []))
    passed = pass_rate >= 0.80 and not bad_repetition and not replacement
    return {
        "available": True,
        "pass": passed,
        "pass_rate": pass_rate,
        "bad_repetition": bad_repetition,
        "replacement_char": replacement,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize LUMA promotion gates.")
    parser.add_argument("--chat", default="")
    parser.add_argument("--memory", default="")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chat = chat_summary(load_json(args.chat))
    memory = memory_summary(load_json(args.memory))
    promote = bool(chat["pass"] and memory["pass"])
    summary = {
        "promote_to_luma_current": promote,
        "verdict": "pass" if promote else "research_only",
        "chat": chat,
        "memory": memory,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
