#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: dict, schema: dict) -> list[str]:
    errors: list[str] = []
    for field in schema["required_fields"]:
        if field not in record:
            errors.append(f"missing:{field}")
    if errors:
        return errors

    hard_blocks = schema.get("hard_blocks", {})
    blocked_licenses = set(hard_blocks.get("license", []))
    if str(record.get("license", "")).lower() in blocked_licenses:
        errors.append("blocked:license")
    if record.get("benchmark_contamination_flag") is True:
        errors.append("blocked:benchmark_contamination")
    if float(record.get("toxicity_score", 0.0)) > float(hard_blocks.get("toxicity_score_greater_than", 1.0)):
        errors.append("blocked:toxicity")
    if float(record.get("pii_score", 0.0)) > float(hard_blocks.get("pii_score_greater_than", 1.0)):
        errors.append("blocked:pii")
    if not str(record.get("text", "")).strip():
        errors.append("blocked:empty_text")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate governed Mamba-3 training JSONL.")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--schema", type=Path, default=Path("neuromamba/configs/mamba3_data_governance_schema.json"))
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    total = 0
    accepted = 0
    rejected = 0
    samples = []
    with args.jsonl.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors = [f"invalid_json:{exc.msg}"]
            else:
                errors = validate_record(record, schema)
            if errors:
                rejected += 1
                if len(samples) < args.max_errors:
                    samples.append({"line": line_no, "errors": errors})
            else:
                accepted += 1

    result = {
        "file": str(args.jsonl),
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "ok": rejected == 0 and total > 0,
        "sample_errors": samples,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
