#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


BENCHMARK_DATASETS = (
    ("edinburgh-dawg/mmlu-redux-2.0", "test"),
)

DEFAULT_STREAMS = (
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text"),
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_benchmark_needles(min_chars: int = 48) -> set[str]:
    try:
        from datasets import get_dataset_config_names, load_dataset
    except Exception as exc:
        print(f"warning: datasets unavailable for contamination filter: {exc}", file=sys.stderr)
        return set()

    needles: set[str] = set()
    for dataset_name, split in BENCHMARK_DATASETS:
        try:
            configs = get_dataset_config_names(dataset_name)
        except Exception as exc:
            print(f"warning: cannot list benchmark configs {dataset_name}: {exc}", file=sys.stderr)
            continue
        for config in configs:
            try:
                ds = load_dataset(dataset_name, config, split=split)
            except Exception as exc:
                print(f"warning: cannot load benchmark {dataset_name}/{config}: {exc}", file=sys.stderr)
                continue
            for row in ds:
                parts = []
                question = row.get("question")
                if isinstance(question, str):
                    parts.append(question)
                choices = row.get("choices")
                if isinstance(choices, list):
                    parts.extend(str(choice) for choice in choices)
                for part in parts:
                    text = normalize(str(part))
                    if len(text) >= min_chars:
                        needles.add(text[: min(180, len(text))])
    return needles


def contaminated(text: str, needles: set[str]) -> bool:
    if not needles:
        return False
    norm = normalize(text)
    return any(needle in norm for needle in needles)


def valid_text(text: str, min_chars: int, max_chars: int) -> bool:
    if len(text) < min_chars or len(text) > max_chars:
        return False
    alpha = sum(ch.isalpha() for ch in text)
    if alpha < min_chars * 0.45:
        return False
    lowered = text.lower()
    bad = ("answer the following multiple choice", "mmlu", "benchmark", "choose the correct answer")
    return not any(term in lowered for term in bad)


def iter_hf_stream(dataset_name: str, config: str, split: str, text_key: str) -> Iterable[str]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config, split=split, streaming=True)
    for row in ds:
        value = row.get(text_key)
        if isinstance(value, str):
            yield value


def record_for(text: str, source: str) -> dict[str, Any]:
    digest = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
    return {
        "text": f'<doc source="{source}" license="dataset" language="en" benchmark_contamination_flag="false">\n{text.strip()}\n</doc>',
        "source": source,
        "license": "dataset",
        "language": "en",
        "domain": "knowledge",
        "quality_score": None,
        "toxicity_score": None,
        "pii_score": None,
        "dedup_hash": digest,
        "benchmark_contamination_flag": False,
        "teacher_model": None,
        "generation_date": None,
    }


def write_local_fallback(out: Path, max_records: int, min_chars: int, max_chars: int, needles: set[str]) -> int:
    paths = [
        Path("neuromamba/data/splits/base_doc_cont_v3_train.jsonl"),
        Path("luma/data/english_completion_bootstrap.txt"),
        Path("luma/data/english_instruction_bootstrap.txt"),
    ]
    seen: set[str] = set()
    count = 0
    with out.open("a", encoding="utf-8") as fh:
        for path in paths:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                    if isinstance(obj, dict):
                        text = str(obj.get("text") or obj.get("content") or "")
                except Exception:
                    pass
                text = text.strip()
                if not valid_text(text, min_chars, max_chars) or contaminated(text, needles):
                    continue
                digest = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                fh.write(json.dumps(record_for(text, f"local:{path}"), ensure_ascii=False) + "\n")
                count += 1
                if count >= max_records:
                    return count
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-cheat knowledge corpus with MMLU-Redux contamination filtering.")
    parser.add_argument("--out", type=Path, default=Path("neuromamba/data/no_cheat_knowledge_v1.jsonl"))
    parser.add_argument("--max-records", type=int, default=200_000)
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--stream-limit-per-source", type=int, default=300_000)
    parser.add_argument("--no-benchmark-filter", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    needles = set() if args.no_benchmark_filter else load_benchmark_needles()
    seen: set[str] = set()
    accepted = 0
    scanned = 0
    sources: dict[str, dict[str, int]] = {}

    with args.out.open("w", encoding="utf-8") as fh:
        for dataset_name, config, split, text_key in DEFAULT_STREAMS:
            source = f"{dataset_name}:{config}:{split}"
            sources[source] = {"scanned": 0, "accepted": 0}
            try:
                stream = iter_hf_stream(dataset_name, config, split, text_key)
                for text in stream:
                    scanned += 1
                    sources[source]["scanned"] += 1
                    if sources[source]["scanned"] > args.stream_limit_per_source:
                        break
                    if not valid_text(text, args.min_chars, args.max_chars):
                        continue
                    if contaminated(text, needles):
                        continue
                    digest = hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    fh.write(json.dumps(record_for(text, source), ensure_ascii=False) + "\n")
                    accepted += 1
                    sources[source]["accepted"] += 1
                    if accepted >= args.max_records:
                        break
            except Exception as exc:
                print(f"warning: stream failed {source}: {exc}", file=sys.stderr)
            if accepted >= args.max_records:
                break

    if accepted < max(1000, args.max_records // 20):
        accepted += write_local_fallback(args.out, args.max_records - accepted, args.min_chars, args.max_chars, needles)

    sidecar = args.out.with_suffix(args.out.suffix + ".manifest.json")
    manifest = {
        "out": str(args.out),
        "accepted": accepted,
        "scanned": scanned,
        "benchmark_filter": not args.no_benchmark_filter,
        "benchmark_needles": len(needles),
        "sources": sources,
    }
    sidecar.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
