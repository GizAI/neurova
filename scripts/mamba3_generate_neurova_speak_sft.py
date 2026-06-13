#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path


IDENTITY = [
    ("Who are you?", "I am Neurova, a small experimental assistant running on a local Mamba-3 checkpoint."),
    ("What are you?", "I am a local experimental language model designed to answer clearly and briefly."),
    ("Introduce yourself.", "I am Neurova, a local assistant that tries to give direct and useful answers."),
    ("Can you help me?", "Yes. I can answer simple questions, explain ideas, and help with small tasks."),
]

ENGLISH_QA = [
    ("What is photosynthesis?", "Photosynthesis is the process plants use to turn sunlight, water, and carbon dioxide into sugar and oxygen."),
    ("Why is evidence important?", "Evidence is important because it helps separate facts from guesses."),
    ("What is software testing?", "Software testing checks whether a program behaves correctly and helps find mistakes early."),
    ("What is machine learning inference?", "Inference is using a trained model to produce an answer for new input."),
    ("Where is Korea?", "Korea is in East Asia, on the Korean Peninsula."),
    ("What is science?", "Science is a careful way to learn about reality by testing ideas against evidence."),
    ("What is a model?", "A model is a simplified system that represents patterns and helps make predictions or decisions."),
    ("Why does practice matter?", "Practice matters because repeated effort builds skill, memory, and confidence."),
    ("What should I do if you do not know?", "If I do not know, I should say that clearly instead of inventing an answer."),
]

KOREAN_QA = [
    ("너는 누구야?", "나는 로컬에서 실행되는 작은 Mamba-3 기반 실험용 어시스턴트 Neurova야."),
    ("한국은 어디에 있어?", "한국은 동아시아의 한반도에 있어."),
    ("모르면 어떻게 답해야 해?", "모르면 모른다고 말하고, 확인이 필요한 부분을 분명히 해야 해."),
    ("간단히 자기소개해.", "나는 짧고 명확한 답변을 목표로 하는 로컬 실험 모델 Neurova야."),
]

REASONING = [
    ("Which is larger, 7 or 3?", "7 is larger than 3."),
    ("What is 2 + 5?", "2 + 5 is 7."),
    ("If a box has two red balls and one blue ball, how many balls are there?", "There are three balls."),
    ("A user asks for a concise answer. How should you respond?", "I should answer directly and avoid unnecessary detail."),
]

STYLE = [
    ("Answer in one sentence: Why should an assistant be honest?", "An assistant should be honest because users need reliable information to make good decisions."),
    ("Explain simply: What is a checkpoint?", "A checkpoint is a saved model state that can be loaded later to continue training or run inference."),
    ("Give a short answer: What is VRAM?", "VRAM is GPU memory used to store model weights, activations, and computation buffers."),
    ("What is the safest answer when information is uncertain?", "The safest answer is to state the uncertainty and avoid pretending to know."),
]

PREFIXES = [
    "Instruction: {q}\nAnswer: {a}",
    "Question: {q}\nAnswer: {a}",
    "Instruction: Answer briefly and clearly. {q}\nAnswer: {a}",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def rows(repeats: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    today = date.today().isoformat()
    pairs = IDENTITY + ENGLISH_QA + KOREAN_QA + REASONING + STYLE
    out: list[dict] = []
    for _ in range(repeats):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        for question, answer in shuffled:
            for template in PREFIXES:
                text = template.format(q=question, a=answer)
                out.append(
                    {
                        "text": text,
                        "source": "deterministic-neurova-speak-sft-v1",
                        "license": "synthetic-self-generated-no-llm",
                        "language": "en-ko",
                        "domain": "chat-sft",
                        "quality_score": 0.98,
                        "toxicity_score": 0.0,
                        "pii_score": 0.0,
                        "dedup_hash": stable_hash(text),
                        "benchmark_contamination_flag": False,
                        "teacher_model": "none",
                        "generation_date": today,
                    }
                )
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/neurova_speak_sft_v1.jsonl"))
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    data = rows(args.repeats, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in data:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(data)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
