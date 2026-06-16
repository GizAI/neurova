#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


HANGUL_RE = re.compile(r"[가-힣]")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?:\+?\d[\s().-]?){10,}")
IM_START_RE = re.compile(r"<\|im_start\|>(system|user|assistant|tool)?\n?", re.IGNORECASE)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8", errors="ignore")).hexdigest()


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def extract_text(row: dict[str, Any], text_field: str = "text") -> str:
    for key in (text_field, "text", "content", "raw_content", "document"):
        if key in row:
            return as_text(row[key])
    return ""


def chatml_to_document(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("<|im_end|>", "\n")
    text = re.sub(r"<\|im_start\|>system\n.*?(?=<\|im_start\|>user\n)", "", text, flags=re.DOTALL)
    text = text.replace("<|im_start|>user\n", "\nProblem:\n")
    text = text.replace("<|im_start|>assistant\n", "\nSolution:\n")
    text = text.replace("<|im_start|>tool\n", "\nTool Output:\n")
    text = IM_START_RE.sub("\n", text)
    return text


def pii_like_score(text: str) -> float:
    return min(1.0, 0.6 * bool(EMAIL_RE.search(text)) + 0.4 * bool(PHONE_RE.search(text)))


def text_quality_score(text: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        return 0.0, ["empty"]
    length = len(stripped)
    alpha = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    spaces = sum(ch.isspace() for ch in stripped)
    words = re.findall(r"[A-Za-z][A-Za-z0-9_'-]+", stripped.lower())
    unique_ratio = len(set(words)) / max(1, len(words))
    alpha_ratio = alpha / max(1, length)
    digit_ratio = digits / max(1, length)
    whitespace_ratio = spaces / max(1, length)
    score = 1.0
    if alpha_ratio < 0.25:
        score -= 0.30
        reasons.append("low_alpha")
    if digit_ratio > 0.45:
        score -= 0.20
        reasons.append("digit_heavy")
    if whitespace_ratio < 0.06:
        score -= 0.15
        reasons.append("low_whitespace")
    if words and unique_ratio < 0.18:
        score -= 0.25
        reasons.append("low_unique_words")
    if re.search(r"(.)\1{7,}", stripped):
        score -= 0.35
        reasons.append("char_repetition")
    lowered = stripped.lower()
    for phrase in ("enable javascript", "cookie policy", "subscribe to our newsletter", "all rights reserved"):
        if phrase in lowered:
            score -= 0.15
            reasons.append("boilerplate")
    return max(0.0, min(1.0, score)), reasons


def clean_text(text: str, filters: dict[str, Any], *, chatml: bool = False) -> tuple[str | None, float, float, list[str]]:
    if chatml:
        text = chatml_to_document(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text = text.strip()
    if filters.get("reject_hangul", True) and HANGUL_RE.search(text):
        return None, 0.0, 0.0, ["hangul"]
    if len(text) < int(filters.get("min_chars", 180)):
        return None, 0.0, 0.0, ["too_short"]
    if len(text) > int(filters.get("max_chars", 14000)):
        text = text[: int(filters.get("max_chars", 14000))].rsplit("\n", 1)[0].strip() or text[: int(filters.get("max_chars", 14000))]
    quality, reasons = text_quality_score(text)
    pii = pii_like_score(text)
    if quality < float(filters.get("min_quality_score", 0.0)):
        return None, quality, pii, reasons
    if pii > float(filters.get("max_pii_score", 1.0)):
        reasons.append("pii_like")
        return None, quality, pii, reasons
    return text, quality, pii, reasons


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def record(text: str, source: dict[str, Any], digest: str, quality: float, pii: float, reasons: list[str]) -> dict[str, Any]:
    return {
        "text": text,
        "source": source["name"],
        "dataset": source["name"],
        "dataset_ref": source.get("dataset") or source.get("path", ""),
        "language": source.get("language", "en"),
        "domain": source.get("domain", ""),
        "quality_score": round(quality, 4),
        "toxicity_score": None,
        "pii_score": round(pii, 4),
        "dedup_hash": digest,
        "benchmark_contamination_flag": None,
        "teacher_model": None,
        "generation_date": None,
        "quality_reasons": reasons,
    }


def load_hf_stream(source: dict[str, Any]):
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    kwargs: dict[str, Any] = {
        "path": source["dataset"],
        "split": source.get("split", "train"),
        "streaming": True,
    }
    if source.get("config"):
        kwargs["name"] = source["config"]
    if token:
        kwargs["token"] = token
    return load_dataset(**kwargs)


def collect_hf(source: dict[str, Any], filters: dict[str, Any], *, progress_every: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    max_train = int(source["max_train_records"])
    max_valid = int(source["max_valid_records"])
    valid_every = int(source.get("valid_every", 101))
    text_field = source.get("text_field", "text")
    seen: set[str] = set()
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    scanned = filtered = 0
    for row in load_hf_stream(source):
        if len(train) >= max_train and len(valid) >= max_valid:
            break
        scanned += 1
        if progress_every > 0 and scanned % progress_every == 0:
            print(
                json.dumps(
                    {
                        "source": source["name"],
                        "kind": source["kind"],
                        "scanned": scanned,
                        "train": len(train),
                        "valid": len(valid),
                        "filtered": filtered,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        text, quality, pii, reasons = clean_text(extract_text(row, text_field), filters)
        if not text:
            filtered += 1
            continue
        digest = stable_hash(text)
        if digest in seen:
            filtered += 1
            continue
        seen.add(digest)
        out = record(text, source, digest, quality, pii, reasons)
        if scanned % valid_every == 0 and len(valid) < max_valid:
            valid.append(out)
        elif len(train) < max_train:
            train.append(out)
    return train, valid, {"scanned": scanned, "filtered": filtered, "train": len(train), "valid": len(valid)}


def collect_local_chatml(source: dict[str, Any], filters: dict[str, Any], *, progress_every: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    max_train = int(source["max_train_records"])
    max_valid = int(source["max_valid_records"])
    path = ROOT / source["path"]
    seen: set[str] = set()
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    scanned = filtered = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if len(train) >= max_train and len(valid) >= max_valid:
                break
            scanned += 1
            if progress_every > 0 and scanned % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "source": source["name"],
                            "kind": source["kind"],
                            "scanned": scanned,
                            "train": len(train),
                            "valid": len(valid),
                            "filtered": filtered,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                filtered += 1
                continue
            text, quality, pii, reasons = clean_text(as_text(obj.get("text", "")), filters, chatml=True)
            if not text:
                filtered += 1
                continue
            digest = stable_hash(text)
            if digest in seen:
                filtered += 1
                continue
            seen.add(digest)
            out = record(text, source, digest, quality, pii, reasons)
            if scanned % 101 == 0 and len(valid) < max_valid:
                valid.append(out)
            elif len(train) < max_train:
                train.append(out)
    return train, valid, {"scanned": scanned, "filtered": filtered, "train": len(train), "valid": len(valid)}


def write_source(source: dict[str, Any], recipe: dict[str, Any]) -> dict[str, Any]:
    filters = recipe["filters"]
    progress_every = int(recipe.get("progress_every", 1000))
    if source.get("valid_every") is None:
        source["valid_every"] = recipe.get("valid_every", 101)
    if source["kind"] == "hf_stream":
        train, valid, stats = collect_hf(source, filters, progress_every=progress_every)
    elif source["kind"] == "local_jsonl_chatml":
        train, valid, stats = collect_local_chatml(source, filters, progress_every=progress_every)
    else:
        raise ValueError(f"unknown source kind: {source['kind']}")
    out_dir = ROOT / recipe["output_root"] / source["name"]
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    manifest = {
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train": str(train_path.relative_to(ROOT)),
        "valid": str(valid_path.relative_to(ROOT)),
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"source": source["name"], **stats, "train_path": str(train_path.relative_to(ROOT))}, ensure_ascii=False), flush=True)
    return manifest


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def sample_mix(recipe: dict[str, Any], split: str, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = int(recipe[f"{split}_records"])
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    ratios = {s["name"]: float(s["ratio"]) for s in recipe["sources"]}
    ratio_override = ROOT / recipe["mix_dir"] / "doremi_ratios.json"
    if ratio_override.exists():
        override = json.loads(ratio_override.read_text(encoding="utf-8"))
        raw = override.get("ratios", override)
        for name, value in raw.items():
            if name in ratios:
                ratios[name] = float(value)
    total = sum(ratios.values())
    ratios = {k: v / total for k, v in ratios.items()}
    seen: set[str] = set()
    for source in recipe["sources"]:
        path = ROOT / recipe["output_root"] / source["name"] / f"{split}.jsonl"
        source_rows = read_rows(path)
        rng.shuffle(source_rows)
        want = int(round(target * ratios[source["name"]]))
        taken = 0
        for row in source_rows:
            digest = row.get("dedup_hash") or stable_hash(row.get("text", ""))
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(row)
            taken += 1
            if taken >= want:
                break
        reports.append({"source": source["name"], "path": str(path.relative_to(ROOT)), "available": len(source_rows), "wanted": want, "taken": taken})
    if len(rows) < target:
        pool: list[dict[str, Any]] = []
        for source in recipe["sources"]:
            pool.extend(read_rows(ROOT / recipe["output_root"] / source["name"] / f"{split}.jsonl"))
        rng.shuffle(pool)
        for row in pool:
            digest = row.get("dedup_hash") or stable_hash(row.get("text", ""))
            if digest in seen:
                continue
            seen.add(digest)
            rows.append(row)
            if len(rows) >= target:
                break
    rng.shuffle(rows)
    return rows[:target], reports


def build_mix(recipe: dict[str, Any], *, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    train, train_reports = sample_mix(recipe, "train", rng)
    valid, valid_reports = sample_mix(recipe, "valid", rng)
    mix_dir = ROOT / recipe["mix_dir"]
    train_path = mix_dir / "train.jsonl"
    valid_path = mix_dir / "valid.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    manifest = {
        "name": recipe["name"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filters": recipe["filters"],
        "train": str(train_path.relative_to(ROOT)),
        "valid": str(valid_path.relative_to(ROOT)),
        "train_records": len(train),
        "valid_records": len(valid),
        "train_counts": Counter(row.get("dataset", "") for row in train),
        "valid_counts": Counter(row.get("dataset", "") for row in valid),
        "reports": train_reports + valid_reports,
    }
    (mix_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare English-only SaneFlow practical pretraining v2 sources and mix.")
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_pretrain_sources_v2.json"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    hf_env = recipe.get("hf_env")
    if hf_env:
        load_env_file(ROOT / hf_env)
    for fallback in recipe.get("hf_env_fallbacks", []):
        load_env_file(ROOT / fallback)
    selected = set(args.only or [])
    manifests = []
    if not args.skip_download:
        for source in recipe["sources"]:
            if selected and source["name"] not in selected:
                continue
            manifests.append(write_source(source, recipe))
    manifest = build_mix(recipe, seed=args.seed)
    (ROOT / recipe["output_root"] / "manifest.json").write_text(
        json.dumps({"recipe": str(args.recipe), "sources": manifests, "mix": manifest}, indent=2, ensure_ascii=False, default=dict) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
