#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Iterable


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ").strip()
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def text_from_row(row: dict) -> str:
    text = row.get("text") or row.get("content") or row.get("document") or ""
    if not text and row.get("prompt") and row.get("response"):
        text = f"{row['prompt']}\n\n{row['response']}"
    return clean_text(text)


def chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    cur = ""
    for para in paragraphs:
        if len(cur) + len(para) + 2 <= max_chars:
            cur = f"{cur}\n\n{para}".strip()
        else:
            if cur:
                parts.append(cur)
            cur = para[:max_chars]
    if cur:
        parts.append(cur)
    return parts


def synthetic_textbook_rows(recipe: dict) -> list[dict]:
    cfg = recipe.get("synthetic_textbook", {})
    topics = cfg.get("topics", [])
    rows = []
    for topic, text in topics:
        rows.append({"text": f"Short explanation about {topic}\n\n{text}", "source": "synthetic_textbook"})
    return rows


def build(args: argparse.Namespace, recipe: dict) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(args.seed)
    source_rows: dict[str, list[dict]] = {}
    reports = []
    sources = recipe.get("sources", [])
    for spec in sources:
        name = spec["name"]
        path = Path(spec["path"])
        ratio = float(spec["ratio"])
        rows = []
        scanned = 0
        for row in iter_jsonl(path):
            scanned += 1
            text = text_from_row(row)
            if len(text) < args.min_chars:
                continue
            if "<|im_start|>" in text or "<|im_end|>" in text:
                continue
            for part in chunk_text(text, args.max_chars):
                if len(part) >= args.min_chars:
                    rows.append({"text": part, "source": row.get("source") or name, "dedup_hash": digest(part)})
            if scanned >= args.max_scan_per_source:
                break
        rng.shuffle(rows)
        source_rows[name] = rows
        reports.append({"source": name, "path": str(path), "ratio": ratio, "scanned": scanned, "usable": len(rows)})
    synth = synthetic_textbook_rows(recipe) if recipe.get("synthetic_textbook", {}).get("enabled", True) else []
    synth_rows = [{"text": r["text"], "source": r["source"], "dedup_hash": digest(r["text"])} for r in synth]
    source_rows["synthetic_textbook"] = synth_rows
    reports.append({"source": "synthetic_textbook", "usable": len(synth_rows)})

    selected: list[dict] = []
    seen: set[str] = set()
    for spec in sources:
        name = spec["name"]
        ratio = float(spec["ratio"])
        want = int(args.train_records * ratio)
        taken = 0
        for row in source_rows.get(name, []):
            if row["dedup_hash"] in seen:
                continue
            seen.add(row["dedup_hash"])
            selected.append(row)
            taken += 1
            if taken >= want:
                break
    synth_floor = int(recipe.get("synthetic_textbook", {}).get("ratio_floor_records", 0))
    for row in (source_rows["synthetic_textbook"] * max(1, synth_floor // max(1, len(source_rows["synthetic_textbook"]))))[:synth_floor]:
        key = digest(row["text"] + str(len(selected)))
        selected.append({"text": row["text"], "source": row["source"], "dedup_hash": key})
    remainder = [r for rows in source_rows.values() for r in rows]
    rng.shuffle(remainder)
    for row in remainder:
        if len(selected) >= args.train_records + args.valid_records:
            break
        if row["dedup_hash"] in seen:
            continue
        seen.add(row["dedup_hash"])
        selected.append(row)

    rng.shuffle(selected)
    valid = selected[: args.valid_records]
    train = selected[args.valid_records : args.valid_records + args.train_records]
    manifest = {
        "name": recipe.get("name", "saneflow_speak_pretrain_v1"),
        "goal": "natural sentence continuation before ChatML SFT",
        "train_records": len(train),
        "valid_records": len(valid),
        "reports": reports,
        "rules": ["no ChatML", "no benchmark eval rows", "short coherent prose first"],
    }
    return train, valid, manifest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({"text": row["text"], "source": row["source"], "dedup_hash": row["dedup_hash"]}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SaneFlow speak-base continuation corpus.")
    parser.add_argument("--recipe", type=Path, default=Path("saneflow/configs/saneflow_speak_pretrain_mix.json"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--train-records", type=int, default=0)
    parser.add_argument("--valid-records", type=int, default=0)
    parser.add_argument("--max-scan-per-source", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    args.train_records = args.train_records or int(recipe["train_records"])
    args.valid_records = args.valid_records or int(recipe["valid_records"])
    args.max_scan_per_source = args.max_scan_per_source or int(recipe["max_scan_per_source"])
    args.min_chars = args.min_chars or int(recipe["min_chars"])
    args.max_chars = args.max_chars or int(recipe["max_chars"])
    out_dir = args.out_dir or Path(recipe["train_out"]).parent
    train, valid, manifest = build(args, recipe)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)
    manifest_path = Path(recipe.get("manifest_out") or out_dir / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["recipe"] = str(args.recipe)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
