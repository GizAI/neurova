#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from saneflow.chat_format import DEFAULT_SYSTEM, IM_END, IM_START, format_chatml_pair


RAW = Path("data/corpus/raw_hf_sft_v3")
OUT = Path("data/corpus/sft_sources")


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def chatml_messages(messages: list[dict[str, Any]], system: str = DEFAULT_SYSTEM) -> str | None:
    out: list[str] = []
    has_system = any(str(m.get("role") or m.get("from") or "").lower() == "system" for m in messages)
    if not has_system:
        out.append(f"{IM_START}system\n{system.strip()}\n{IM_END}")
    for msg in messages:
        role = str(msg.get("role") or msg.get("from") or "user").lower().strip()
        if role in {"human"}:
            role = "user"
        elif role in {"gpt", "model"}:
            role = "assistant"
        if role not in {"system", "user", "assistant"}:
            role = "user"
        content = str(msg.get("content") or msg.get("value") or "").strip()
        if content:
            out.append(f"{IM_START}{role}\n{content}\n{IM_END}")
    text = "\n".join(out)
    return text if f"{IM_START}assistant" in text else None


def clean_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in str(text).strip().splitlines()).strip()


def make_row(text: str | None, source: str, bucket: str) -> dict[str, Any] | None:
    if not text or f"{IM_START}assistant" not in text:
        return None
    text = clean_text(text)
    return {"text": text, "source": source, "bucket": bucket, "dedup_hash": digest(text)}


def write_rows(path: Path, rows: Iterable[dict[str, Any] | None], limit: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if not row:
                continue
            key = row["dedup_hash"]
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if count >= limit:
                break
    return count


def iter_json_any(path: Path) -> Iterable[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return
    if text[0] == "[":
        for row in json.loads(text):
            yield row
        return
    for line in text.splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def parquet_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("pyarrow is required only for source preparation. Install it locally or use prebuilt sft_sources JSONL.") from exc
    for path in paths:
        if path.exists():
            for row in pq.read_table(path).to_pylist():
                yield row


def prepare_openr1() -> int:
    paths = sorted((RAW / "open-r1__OpenR1-Math-220k" / "all").glob("*.parquet"))

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in parquet_rows(paths):
            problem = clean_text(row.get("problem") or "")
            solution = clean_text(row.get("solution") or "")
            answer = clean_text(row.get("answer") or "")
            if answer and answer not in solution[-500:]:
                solution = f"{solution}\n\nFinal answer: {answer}"
            yield make_row(format_chatml_pair(problem, solution) if problem and solution else None, "openr1_math_220k", "reasoning")

    return write_rows(OUT / "openr1_math_220k.jsonl", rows(), 2500)


def prepare_bespoke() -> int:
    paths = [RAW / "bespokelabs__Bespoke-Stratos-17k" / "data" / "train-00000-of-00001.parquet"]

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in parquet_rows(paths):
            conv = row.get("conversations") or []
            yield make_row(chatml_messages(conv, system=DEFAULT_SYSTEM), "bespoke_stratos_17k", "reasoning")

    return write_rows(OUT / "bespoke_stratos_17k.jsonl", rows(), 2500)


def prepare_limo() -> int:
    path = RAW / "GAIR__LIMO" / "limo.jsonl"

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in iter_json_any(path):
            question = clean_text(row.get("question") or "")
            solution = clean_text(row.get("solution") or "")
            yield make_row(format_chatml_pair(question, solution) if question and solution else None, "limo", "reasoning")

    return write_rows(OUT / "limo.jsonl", rows(), 817)


def prepare_smoltalk2() -> int:
    paths = sorted((RAW / "HuggingFaceTB__smoltalk2" / "SFT").glob("*.parquet"))

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in parquet_rows(paths):
            yield make_row(chatml_messages(row.get("messages") or []), "smoltalk2_direct", "smoltalk2")

    return write_rows(OUT / "smoltalk2_direct.jsonl", rows(), 5000)


def prepare_opencode() -> int:
    paths = sorted((RAW / "nvidia__OpenCodeInstruct" / "data").glob("train-*.parquet"))

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in parquet_rows(paths):
            user = clean_text(row.get("input") or "")
            answer = clean_text(row.get("output") or "")
            yield make_row(format_chatml_pair(user, answer) if user and answer else None, "opencodeinstruct_direct", "code")

    return write_rows(OUT / "opencodeinstruct_direct.jsonl", rows(), 3200)


def prepare_toolace() -> int:
    path = RAW / "Team-ACE__ToolACE" / "data.json"

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in iter_json_any(path):
            conv = row.get("conversations") or []
            if isinstance(conv, list) and conv:
                user = ""
                answer = ""
                for msg in conv:
                    role = str(msg.get("from") or msg.get("role") or "").lower()
                    value = clean_text(msg.get("value") or msg.get("content") or "")
                    if role == "user" and not user:
                        user = value
                    elif role == "assistant" and user:
                        answer = value
                        break
                yield make_row(format_chatml_pair(user, answer) if user and answer else None, "toolace", "tool")
                continue
            question = clean_text(row.get("question") or row.get("query") or "")
            answer = clean_text(row.get("answer") or row.get("output") or "")
            yield make_row(format_chatml_pair(question, answer) if question and answer else None, "toolace", "tool")

    return write_rows(OUT / "toolace.jsonl", rows(), 1200)


def prepare_apigen() -> int:
    paths = [RAW / "argilla__apigen-function-calling" / "data" / "train-00000-of-00001.parquet"]

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in parquet_rows(paths):
            query = clean_text(row.get("query") or "")
            tools = clean_text(row.get("tools") or "")
            answer = clean_text(row.get("answers") or "")
            user = f"Available tools:\n{tools}\n\nUser request: {query}\nReturn the tool call JSON."
            yield make_row(format_chatml_pair(user, answer) if query and tools and answer else None, "apigen_function_calling", "tool")

    return write_rows(OUT / "apigen_function_calling.jsonl", rows(), 1000)


def prepare_kit19() -> int:
    path = RAW / "Junmai__kit-19-instruction-100000" / "toolkit_100000.csv"

    def rows() -> Iterable[dict[str, Any] | None]:
        with path.open(encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.DictReader(f):
                inst = clean_text(row.get("instruction") or "")
                inp = clean_text(row.get("input") or "")
                out = clean_text(row.get("output") or "")
                user = f"{inst}\n\n{inp}".strip()
                yield make_row(format_chatml_pair(user, out) if user and out else None, "kit19", "korean")

    return write_rows(OUT / "kit19.jsonl", rows(), 1400)


def prepare_polyguard() -> int:
    paths = sorted((RAW / "ToxicityPrompts__PolyGuardMix" / "data").glob("*.parquet"))

    def rows() -> Iterable[dict[str, Any] | None]:
        for row in parquet_rows(paths):
            prompt = clean_text(row.get("prompt") or "")
            response = clean_text(row.get("response") or "")
            refusal = str(row.get("response_refusal_label") or "").lower()
            harmful = str(row.get("prompt_harm_label") or "").lower()
            if not response:
                response = "I can't help with harmful requests, but I can help with a safe alternative." if "yes" in {refusal, harmful} else "I do not know."
            yield make_row(format_chatml_pair(prompt, response) if prompt else None, "polyguardmix", "safety")

    return write_rows(OUT / "polyguardmix.jsonl", rows(), 900)


PREPARERS = {
    "openr1_math_220k": prepare_openr1,
    "bespoke_stratos_17k": prepare_bespoke,
    "limo": prepare_limo,
    "smoltalk2_direct": prepare_smoltalk2,
    "opencodeinstruct_direct": prepare_opencode,
    "toolace": prepare_toolace,
    "apigen_function_calling": prepare_apigen,
    "kit19": prepare_kit19,
    "polyguardmix": prepare_polyguard,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical ChatML SFT source JSONL files from raw HF caches.")
    parser.add_argument("--only", nargs="*", choices=sorted(PREPARERS), default=None)
    args = parser.parse_args()

    selected = args.only or sorted(PREPARERS)
    manifest: dict[str, int] = {}
    for name in selected:
        count = PREPARERS[name]()
        manifest[name] = count
        print(f"{name}: {count}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
