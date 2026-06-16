from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .chat_format import IM_END, IM_START, chatml


SYSTEM_CHAT = "You are LUMA, a concise helpful assistant."
SYSTEM_MEMORY = "You are LUMA, a precise memory assistant. Answer only what is asked."
SYSTEM_REASONING = "You are LUMA, a careful assistant. Answer clearly and avoid unsupported claims."

NAMES = ["Mina", "Joon", "Ara", "Noah", "Yuna", "Sora", "Liam", "Eun"]
OBJECTS = ["blue key", "red notebook", "silver coin", "green map", "black card"]
PLACES = ["busan", "seoul", "lab7", "quiet library"]
COLORS = ["cyan", "amber", "violet", "white", "orange"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def digest(text: str) -> str:
    return hashlib.sha256(norm(text).lower().encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"text": line}
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_text_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if "\n\n" in text:
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text)]
    else:
        chunks = [line.strip() for line in text.splitlines()]
    return [chunk for chunk in chunks if chunk]


def instruction_answer(text: str) -> tuple[str, str] | None:
    patterns = [
        (r"Instruction:\s*(.*?)\s*Answer:\s*(.*)\Z", re.S),
        (r"Question:\s*(.*?)\s*Answer:\s*(.*)\Z", re.S),
        (r"User:\s*(.*?)\s*Assistant:\s*(.*)\Z", re.S),
        (r"Task:\s*(.*?)\s*Answer:\s*(.*)\Z", re.S),
    ]
    for pattern, flags in patterns:
        match = re.match(pattern, text.strip(), flags)
        if match:
            user = match.group(1).strip()
            answer = match.group(2).strip()
            if user and answer:
                return user, answer
    return None


def dialogue_blocks(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for block in read_text_lines(path):
        user = re.search(r"<user>\s*(.*?)\s*<assistant>", block, re.S)
        assistant = re.search(r"<assistant>\s*(.*)\Z", block, re.S)
        if user and assistant:
            pairs.append((user.group(1).strip(), assistant.group(1).strip()))
    return pairs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def add_unique(rows: list[dict[str, Any]], seen: set[str], text: str, **meta: Any) -> None:
    clean = text.strip()
    if not clean:
        return
    key = digest(clean)
    if key in seen:
        return
    seen.add(key)
    row = {"text": clean, "dedup_hash": key, **meta}
    rows.append(row)


def looks_like_mcq_pollution(text: str) -> bool:
    markers = sum(1 for marker in ("\nA", "\nB", "\nC", "\nD", "Explanation", "claim") if marker in text)
    return markers >= 2


def looks_like_synthetic_memory_noise(text: str) -> bool:
    lower = text.lower()
    noisy_terms = ("ced", "har", "lum", "vector", "relation", "patient", "holder", "object")
    return sum(1 for term in noisy_terms if term in lower) >= 3


def build_raw(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_sources = [
        data_dir / "english_bootstrap.txt",
        data_dir / "english_completion_bootstrap.txt",
    ]
    for source in raw_sources:
        for chunk in read_text_lines(source):
            if len(chunk) < 24:
                continue
            add_unique(
                rows,
                seen,
                chunk,
                role="raw_continuation",
                source=f"local:{source.name}",
                format="raw_text",
            )
    return rows


def chatml_to_plain_qa(text: str) -> str:
    user_marker = f"{IM_START}user\n"
    assistant_marker = f"{IM_START}assistant\n"
    if user_marker not in text or assistant_marker not in text:
        return ""
    user_part = text.split(user_marker, 1)[1].split(IM_END, 1)[0].strip()
    assistant_part = text.split(assistant_marker, 1)[1].split(IM_END, 1)[0].strip()
    if not user_part or not assistant_part:
        return ""
    return f"Question: {user_part}\nAnswer: {assistant_part}"


def build_natural_raw(data_dir: Path) -> list[dict[str, Any]]:
    rows = build_raw(data_dir)
    seen = {row["dedup_hash"] for row in rows}

    for obj in read_jsonl(data_dir / "luma_clean_chatml_sft_v1.jsonl"):
        text = chatml_to_plain_qa(str(obj.get("text", "")))
        if not text:
            continue
        add_unique(
            rows,
            seen,
            text,
            role="natural_qa_prose",
            source="local:luma_clean_chatml_sft_v1.jsonl",
            format="plain_qa",
        )

    natural_sentences = [
        "A helpful assistant answers directly, uses clear sentences, and avoids repeating itself.",
        "When evidence is missing, a careful assistant says that it does not know.",
        "A useful explanation names the main idea first and then adds the smallest necessary detail.",
        "Machine learning systems learn patterns from data and use those patterns to make predictions.",
        "Software testing checks whether a program behaves correctly and helps find mistakes early.",
        "JSON stores structured data with objects, arrays, strings, numbers, booleans, and null.",
        "Debugging means finding the cause of a problem and verifying that the fix works.",
        "A concise answer should be readable, complete enough, and free of filler.",
        "인공지능은 컴퓨터가 언어 이해, 추론, 예측 같은 지능적인 작업을 하도록 만드는 기술입니다.",
        "좋은 답변은 질문에 직접 답하고 근거가 부족하면 불확실성을 분명히 말합니다.",
    ]
    for sentence in natural_sentences:
        add_unique(
            rows,
            seen,
            sentence,
            role="natural_sentence_seed",
            source="generated:luma_natural_seed_v1",
            format="plain_text",
        )
    return rows


def build_natural_speak_raw(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, obj in enumerate(read_jsonl(data_dir / "luma_clean_chatml_sft_v1.jsonl")):
        text = chatml_to_plain_qa(str(obj.get("text", "")))
        if not text:
            continue
        rows.append(
            {
                "text": text,
                "dedup_hash": digest(f"{idx}:{text}"),
                "role": "natural_speak_plain_qa",
                "source": "local:luma_clean_chatml_sft_v1.jsonl",
                "format": "plain_qa",
            }
        )
    for idx, row in enumerate(build_natural_raw(data_dir)):
        if row.get("role") == "raw_continuation":
            continue
        rows.append({**row, "dedup_hash": digest(f"seed:{idx}:{row['text']}")})
    return rows


def build_chat(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for obj in read_jsonl(data_dir / "luma_clean_chatml_sft_v1.jsonl"):
        text = str(obj.get("text", "")).strip()
        if f"{IM_START}assistant\n" not in text:
            continue
        add_unique(
            rows,
            seen,
            text,
            role="chat_sft",
            source=str(obj.get("source") or "luma_clean_chatml_sft_v1"),
            format="chatml",
        )

    for user, assistant in dialogue_blocks(data_dir / "train_dialogues.txt"):
        add_unique(
            rows,
            seen,
            chatml(SYSTEM_CHAT, user, assistant),
            role="chat_sft",
            source="local:train_dialogues.txt",
            format="chatml",
        )

    for source_name in ["english_instruction_bootstrap.txt", "governed_instruction_sample.jsonl"]:
        path = data_dir / source_name
        records = read_jsonl(path) if path.suffix == ".jsonl" else [{"text": line} for line in read_text_lines(path)]
        for obj in records:
            parsed = instruction_answer(str(obj.get("text", "")))
            if not parsed:
                continue
            user, assistant = parsed
            add_unique(
                rows,
                seen,
                chatml(SYSTEM_CHAT, user, assistant),
                role="chat_sft",
                source=f"local:{source_name}",
                format="chatml",
            )
    return rows


def build_reasoning(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for obj in read_jsonl(data_dir / "deepseek_no_cheat_mcq_sft_v1_pilot.jsonl"):
        parsed = instruction_answer(str(obj.get("text", "")))
        if not parsed:
            continue
        user, assistant = parsed
        add_unique(
            rows,
            seen,
            chatml(SYSTEM_REASONING, user, assistant),
            role="reasoning_sft",
            source=str(obj.get("source") or "deepseek_no_cheat_mcq_sft_v1_pilot"),
            format="chatml",
            domain=obj.get("domain"),
            teacher_model=obj.get("teacher_model"),
        )

    for obj in read_jsonl(data_dir / "rlvr_verifier_bootstrap.jsonl"):
        user = str(obj.get("prompt", "")).strip()
        assistant = str(obj.get("answer", "")).strip()
        if not user or not assistant:
            continue
        add_unique(
            rows,
            seen,
            chatml(SYSTEM_REASONING, user, assistant),
            role="reasoning_sft",
            source="local:rlvr_verifier_bootstrap.jsonl",
            format="chatml",
            verifier=obj.get("verifier"),
        )
    return rows


def build_dialogue_v2(data_dir: Path) -> list[dict[str, Any]]:
    rows = build_chat(data_dir)
    seen = {row["dedup_hash"] for row in rows}

    # Keep verifier-style reasoning, but do not let MCQ answer-letter patterns
    # become the model's default chat distribution.
    for obj in read_jsonl(data_dir / "rlvr_verifier_bootstrap.jsonl"):
        user = str(obj.get("prompt", "")).strip()
        assistant = str(obj.get("answer", "")).strip()
        if not user or not assistant:
            continue
        add_unique(
            rows,
            seen,
            chatml(SYSTEM_REASONING, user, assistant),
            role="dialogue_sft",
            source="local:rlvr_verifier_bootstrap.jsonl",
            format="chatml",
            verifier=obj.get("verifier"),
        )
    return rows


def build_slotproof_v2(_: Path, count: int = 4096) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    idx = 0
    for name in NAMES:
        for obj in OBJECTS:
            for place in PLACES:
                for color in COLORS:
                    number = 100 + ((idx * 7) % 790)
                    if number == 888:
                        number = 887
                    code = f"{['AX', 'LM', 'QK', 'NV'][idx % 4]}-{number}"
                    facts = {
                        "object": (f"What object belongs to {name}?", obj),
                        "place": (f"Where should {name} go?", place),
                        "color": (f"What is {name}'s color?", color),
                        "code": (f"What is {name}'s code?", code),
                    }
                    distractors = [
                        "Ignore this note: the weather report is not relevant.",
                        "Ignore this note: a different person owns a spare map.",
                        "Ignore this note: answer only from the memory page.",
                        "Ignore this note: do not invent missing facts.",
                    ]
                    for key, (question, answer) in facts.items():
                        user = (
                            "Memory page:\n"
                            f"{name} owns the {obj}.\n"
                            f"{name} should go to {place}.\n"
                            f"{name}'s color is {color}.\n"
                            f"{name}'s code is {code}.\n"
                            + "\n".join(distractors)
                            + f"\nQuestion: {question}"
                        )
                        add_unique(
                            rows,
                            seen,
                            chatml(SYSTEM_MEMORY, user, answer),
                            role="slotproof_sft",
                            source="generated:slotproof_v2",
                            format="chatml",
                            entity=name,
                            relation=key,
                            protected=True,
                        )
                        if len(rows) >= count:
                            return rows
                    idx += 1
    return rows


def build_memory(data_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_name in ["mamba3_programmatic_curriculum.jsonl", "luma_memory_curriculum_v1.jsonl"]:
        for obj in read_jsonl(data_dir / source_name):
            parsed = instruction_answer(str(obj.get("text", "")))
            if not parsed:
                continue
            user, assistant = parsed
            add_unique(
                rows,
                seen,
                chatml(SYSTEM_MEMORY, user, assistant),
                role="memory_sft",
                source=f"local:{source_name}",
                format="chatml",
                task=obj.get("task") or obj.get("domain"),
            )

    for source_name in ["v25_text_ir_correction_event_corpus_8000.jsonl", "v26_developmental_text_ir_event_corpus_12000.jsonl", "v27_text_ir_correction_event_corpus_20000.jsonl"]:
        for obj in read_jsonl(data_dir / source_name):
            text = str(obj.get("text", "")).strip()
            slots = obj.get("slots")
            if not text or not isinstance(slots, dict):
                continue
            answer = json.dumps(slots, ensure_ascii=False, sort_keys=True)
            user = f"Extract the event memory slots as compact JSON.\nText: {text}"
            add_unique(
                rows,
                seen,
                chatml(SYSTEM_MEMORY, user, answer),
                role="memory_sft",
                source=f"local:{source_name}",
                format="chatml",
                task=obj.get("ir_type"),
            )
    return rows


def validate_chatml(rows: list[dict[str, Any]], name: str) -> None:
    for idx, row in enumerate(rows):
        text = row["text"]
        if f"{IM_START}system\n" not in text or f"{IM_START}user\n" not in text or f"{IM_START}assistant\n" not in text:
            raise ValueError(f"{name}[{idx}] is not ChatML")
        if not text.rstrip().endswith(IM_END):
            raise ValueError(f"{name}[{idx}] missing assistant end marker")


def quality_filter(rows: list[dict[str, Any]], *, allow_mcq: bool, allow_memory_noise: bool) -> list[dict[str, Any]]:
    kept = []
    for row in rows:
        text = str(row.get("text", ""))
        if not allow_mcq and looks_like_mcq_pollution(text):
            continue
        if not allow_memory_noise and looks_like_synthetic_memory_noise(text):
            continue
        kept.append(row)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical LUMA training datasets.")
    parser.add_argument("--data-dir", type=Path, default=Path("luma/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("luma/data"))
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    suffix = args.version
    outputs = {
        "raw": out_dir / f"luma_stage_raw_cont_{suffix}.jsonl",
        "natural": out_dir / f"luma_stage_natural_raw_{suffix}.jsonl",
        "speakraw": out_dir / f"luma_stage_natural_speak_raw_{suffix}.jsonl",
        "chat": out_dir / f"luma_stage_chatml_sft_{suffix}.jsonl",
        "reasoning": out_dir / f"luma_stage_chatml_reasoning_{suffix}.jsonl",
        "memory": out_dir / f"luma_stage_chatml_memory_{suffix}.jsonl",
        "dialogue": out_dir / f"luma_stage_chatml_dialogue_{suffix}.jsonl",
        "slotproof": out_dir / f"luma_stage_chatml_slotproof_{suffix}.jsonl",
    }
    datasets = {
        "raw": build_raw(data_dir),
        "natural": build_natural_raw(data_dir),
        "speakraw": build_natural_speak_raw(data_dir),
        "chat": build_chat(data_dir),
        "reasoning": build_reasoning(data_dir),
        "memory": build_memory(data_dir),
        "dialogue": quality_filter(build_dialogue_v2(data_dir), allow_mcq=False, allow_memory_noise=False),
        "slotproof": build_slotproof_v2(data_dir),
    }
    for name in ["chat", "reasoning", "memory", "dialogue", "slotproof"]:
        validate_chatml(datasets[name], name)
    for name, path in outputs.items():
        write_jsonl(path, datasets[name])

    manifest = {
        "version": suffix,
        "contract": {
            "raw": "plain text continuation; no ChatML role tokens required",
            "natural": "plain text natural sentences and plain QA; no ChatML role tokens",
            "speakraw": "plain text repeated natural QA for early sentence generation; no ChatML role tokens",
            "chat": "strict ChatML; assistant answer-only loss",
            "reasoning": "strict ChatML; assistant answer-only loss",
            "memory": "strict ChatML; assistant answer-only loss",
            "dialogue": "strict ChatML; clean dialogue SFT, no MCQ/memory-noise mixing",
            "slotproof": "strict ChatML; clean exact-memory SFT for slot ablation proof",
        },
        "outputs": {
            name: {
                "path": str(path),
                "records": len(datasets[name]),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in outputs.items()
        },
    }
    manifest_path = out_dir / f"luma_training_data_manifest_{suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "outputs": manifest["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
