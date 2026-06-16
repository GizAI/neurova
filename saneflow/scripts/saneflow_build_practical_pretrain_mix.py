#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", errors="ignore")).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                text = obj.get("text", "")
                if isinstance(text, str) and text.strip():
                    rows.append(obj)
    return rows


def text_quality_score(text: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        return 0.0, ["empty"]
    length = len(stripped)
    alpha = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    spaces = sum(ch.isspace() for ch in stripped)
    unique_words = len(set(re.findall(r"[A-Za-z][A-Za-z0-9_'-]+", stripped.lower())))
    words = max(1, len(re.findall(r"\S+", stripped)))
    alpha_ratio = alpha / max(1, length)
    digit_ratio = digits / max(1, length)
    whitespace_ratio = spaces / max(1, length)
    unique_ratio = unique_words / words
    long_repeat = max((len(m.group(0)) for m in re.finditer(r"(.)\1{5,}", stripped)), default=0)
    score = 1.0
    if length < 160:
        score -= 0.35
        reasons.append("too_short")
    if alpha_ratio < 0.28:
        score -= 0.30
        reasons.append("low_alpha")
    if digit_ratio > 0.35:
        score -= 0.20
        reasons.append("digit_heavy")
    if whitespace_ratio < 0.08:
        score -= 0.15
        reasons.append("low_whitespace")
    if unique_ratio < 0.20:
        score -= 0.25
        reasons.append("low_unique_words")
    if long_repeat >= 8:
        score -= 0.35
        reasons.append("char_repetition")
    boilerplate = (
        "cookie policy",
        "enable javascript",
        "all rights reserved",
        "subscribe to our newsletter",
        "sign up for our newsletter",
    )
    lowered = stripped.lower()
    hits = sum(1 for phrase in boilerplate if phrase in lowered)
    if hits:
        score -= min(0.35, 0.15 * hits)
        reasons.append("boilerplate")
    return max(0.0, min(1.0, score)), reasons


def pii_like_score(text: str) -> float:
    email = bool(re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text))
    phone = bool(re.search(r"(?:\+?\d[\s().-]?){10,}", text))
    return min(1.0, 0.6 * email + 0.4 * phone)


def ratio_map(recipe: dict[str, Any]) -> dict[str, float]:
    ratios = {source["name"]: float(source["ratio"]) for source in recipe["sources"]}
    override_path = recipe.get("ratio_override")
    if override_path and Path(override_path).exists():
        override = json.loads(Path(override_path).read_text(encoding="utf-8"))
        raw = override.get("ratios", override)
        for name, value in raw.items():
            if name in ratios:
                ratios[name] = float(value)
    total = sum(max(0.0, value) for value in ratios.values())
    if total <= 0:
        raise ValueError("source ratios sum to zero")
    return {name: max(0.0, value) / total for name, value in ratios.items()}


def sample_split(recipe: dict[str, Any], split: str, target: int, rng: random.Random) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    reports = []
    ratios = ratio_map(recipe)
    quality = recipe.get("quality_filter", {})
    min_quality = float(quality.get("min_quality_score", 0.0))
    max_pii = float(quality.get("max_pii_score", 1.0))

    def accept_row(row: dict[str, Any], source_name: str) -> dict[str, Any] | None:
        text = row.get("text", "")
        if not isinstance(text, str):
            return None
        text = text.strip()
        if not text:
            return None
        key = row.get("dedup_hash") or digest(text)
        if key in seen:
            return None
        heuristic_quality, quality_reasons = text_quality_score(text)
        upstream_quality = row.get("quality_score", None)
        try:
            upstream_quality_f = float(upstream_quality) if upstream_quality is not None else heuristic_quality
        except (TypeError, ValueError):
            upstream_quality_f = heuristic_quality
        final_quality = min(1.0, max(0.0, 0.5 * heuristic_quality + 0.5 * upstream_quality_f))
        pii_score = max(float(row.get("pii_score", 0.0) or 0.0), pii_like_score(text))
        if final_quality < min_quality or pii_score > max_pii:
            return None
        seen.add(key)
        return {
            "text": text,
            "source": row.get("source") or source_name,
            "dataset": row.get("dataset") or source_name,
            "language": row.get("language", ""),
            "domain": row.get("domain", ""),
            "quality_score": round(final_quality, 4),
            "pii_score": round(pii_score, 4),
            "quality_reasons": quality_reasons,
            "dedup_hash": key,
        }

    for source in recipe["sources"]:
        path = Path(source[split])
        rows = read_rows(path)
        rng.shuffle(rows)
        want = int(round(target * float(ratios[source["name"]])))
        taken = 0
        filtered = 0
        for row in rows:
            clean = accept_row(row, source["name"])
            if clean is None:
                filtered += 1
                continue
            selected.append(clean)
            taken += 1
            if taken >= want:
                break
        reports.append({
            "source": source["name"],
            "split": split,
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "available": len(rows),
            "wanted": want,
            "taken": taken,
            "filtered": filtered,
            "ratio": ratios[source["name"]],
            "missing_or_empty": not path.exists() or (path.exists() and path.stat().st_size == 0),
        })

    if len(selected) < target:
        remainder = []
        for source in recipe["sources"]:
            for row in read_rows(Path(source[split])):
                row["_source_name"] = source["name"]
                remainder.append(row)
        rng.shuffle(remainder)
        for row in remainder:
            clean = accept_row(row, str(row.get("_source_name") or row.get("dataset") or "unknown"))
            if clean is None:
                continue
            selected.append(clean)
            if len(selected) >= target:
                break
    rng.shuffle(selected)
    return selected[:target], reports


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ratio-controlled practical base pretrain mix.")
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_practical_pretrain_mix.json"))
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    train, train_reports = sample_split(recipe, "train", int(recipe["train_records"]), rng)
    valid, valid_reports = sample_split(recipe, "valid", int(recipe["valid_records"]), rng)
    write_jsonl(Path(recipe["train_out"]), train)
    write_jsonl(Path(recipe["valid_out"]), valid)
    manifest = {
        "name": recipe["name"],
        "ratio_override": recipe.get("ratio_override", ""),
        "effective_ratios": ratio_map(recipe),
        "quality_filter": recipe.get("quality_filter", {}),
        "train": recipe["train_out"],
        "valid": recipe["valid_out"],
        "train_records": len(train),
        "valid_records": len(valid),
        "train_counts": Counter(row.get("dataset", "") for row in train),
        "valid_counts": Counter(row.get("dataset", "") for row in valid),
        "reports": train_reports + valid_reports,
    }
    out = Path(recipe["manifest_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
