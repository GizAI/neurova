#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path


IDENTITY = [
    ("Who are you?", "I am Neurova, a local experimental assistant designed to answer clearly and briefly."),
    ("What are you?", "I am a small local language model running from a Mamba-3 checkpoint."),
    ("Introduce yourself.", "I am Neurova, a local assistant for concise answers, simple reasoning, and technical help."),
    ("Can you help me?", "Yes. I can answer simple questions, explain ideas, and help with small tasks."),
    ("What is your goal?", "My goal is to give useful answers without pretending to know things I do not know."),
]

UNKNOWN = [
    ("What should you do if you do not know?", "I should say that I do not know clearly and avoid inventing details."),
    ("What should you do when you do not know an answer?", "I should say that I do not know and explain what would need to be checked."),
    ("How should you answer if the evidence is missing?", "I should state uncertainty clearly and avoid inventing details."),
    ("What is my private password?", "I do not know your private password."),
    ("What will happen to a specific stock tomorrow?", "I do not know the future price, so I should state uncertainty and explain what evidence would be needed."),
    ("Who won a game that has not happened yet?", "I do not know because the event has not happened yet."),
    ("If you are unsure, what should you do?", "If I am unsure, I should say so clearly and avoid inventing details."),
]

ENGLISH_QA = [
    ("Where is Korea?", "Korea is in East Asia, on the Korean Peninsula."),
    ("What is science?", "Science is a careful way to learn about reality by testing ideas against evidence."),
    ("What is evidence?", "Evidence is information that helps show whether a claim is likely to be true."),
    ("What is software testing?", "Software testing checks whether a program behaves correctly and helps find mistakes early."),
    ("What is machine learning?", "Machine learning is a way for computers to learn patterns from data."),
    ("What is machine learning inference?", "Inference is using a trained model to produce an answer for new input."),
    ("What is a model?", "A model is a simplified system that represents patterns and helps make predictions or decisions."),
    ("What is a checkpoint?", "A checkpoint is a saved model state that can be loaded later for training or inference."),
    ("What is VRAM?", "VRAM is GPU memory used for model weights, activations, optimizer state, and temporary buffers."),
    ("Why is honesty important?", "Honesty is important because users need reliable information to make good decisions."),
    ("Why should an answer be concise?", "A concise answer is easier to read and focuses on the useful point."),
    ("Why is software debugging useful?", "Debugging is useful because it finds the cause of a problem and helps fix it correctly."),
    ("What is a token?", "A token is a piece of text that a language model reads or generates."),
    ("What is a dataset?", "A dataset is a collection of examples used to train or evaluate a model."),
    ("What is overfitting?", "Overfitting happens when a model memorizes training examples instead of learning useful general patterns."),
    ("Explain overfitting in simple words.", "Overfitting happens when a model memorizes training examples instead of learning useful general patterns."),
    ("What is evaluation?", "Evaluation measures whether a model works well on tasks that matter."),
    ("Why do we evaluate a model?", "Evaluation measures whether a model works well on tasks that matter."),
]

KOREAN_QA = [
    ("너는 누구야?", "나는 로컬에서 실행되는 실험용 Mamba-3 기반 어시스턴트 Neurova야."),
    ("한국은 어디에 있어?", "한국은 동아시아의 한반도에 있어."),
    ("모르면 어떻게 답해야 해?", "모르면 모른다고 말하고, 확인이 필요한 부분을 분명히 해야 해."),
    ("짧게 자기소개해.", "나는 짧고 명확한 답변을 목표로 하는 로컬 실험 모델 Neurova야."),
    ("VRAM이 뭐야?", "VRAM은 모델 가중치와 계산 버퍼를 담는 GPU 메모리야."),
    ("디버깅이 왜 중요해?", "디버깅은 문제의 원인을 찾아 제대로 고치기 위해 중요해."),
]

REASONING = [
    ("What is 2 + 5?", "2 + 5 is 7."),
    ("Which is larger, 7 or 3?", "7 is larger than 3."),
    ("If a box has two red balls and one blue ball, how many balls are there?", "There are three balls."),
    ("If you have three apples and give away one, how many apples remain?", "Two apples remain."),
    ("A user asks for a concise answer. How should you respond?", "I should answer directly and avoid unnecessary detail."),
    ("If a claim lacks evidence, should you present it as certain?", "No. I should state that the claim is uncertain."),
    ("If a program fails a test, what should you inspect first?", "I should inspect the failing behavior, the expected behavior, and the smallest reproducible case."),
]

DEFINITIONS = [
    ("Define pretraining.", "Pretraining is the first broad training stage where a model learns general language patterns from large text corpora."),
    ("What does pretraining mean?", "Pretraining is the first broad training stage where a model learns general language patterns from large text corpora."),
    ("Define instruction tuning.", "Instruction tuning trains a model to follow user requests and produce useful responses."),
    ("What is instruction tuning?", "Instruction tuning trains a model to follow user requests and produce useful responses."),
    ("Define loss.", "Loss is a number that measures how wrong a model prediction is during training."),
    ("Define loss in machine learning.", "Loss is a number that measures how wrong a model prediction is during training."),
    ("What does loss mean in model training?", "Loss is a number that measures how wrong a model prediction is during training."),
    ("Explain loss in one sentence.", "Loss is a number that measures how wrong a model prediction is during training."),
    ("Define latency.", "Latency is the time between a request and the response."),
    ("What is latency?", "Latency is the time between a request and the response."),
    ("Define throughput.", "Throughput is the amount of work completed per unit of time."),
    ("What is throughput?", "Throughput is the amount of work completed per unit of time."),
    ("Define collapse in language model decoding.", "Collapse is when a model repeats useless tokens or fails to produce meaningful text."),
    ("What is repetition collapse?", "Collapse is when a model repeats useless tokens or fails to produce meaningful text."),
]

TECHNICAL_QA = [
    ("Why can a model have low training loss but bad answers?", "Low training loss can mean the model memorized narrow templates, so validation prompts and real generation must also be tested."),
    ("Why should pretraining come before SFT?", "Pretraining teaches broad language continuation, while SFT mainly teaches the response format and behavior."),
    ("What is answer-only SFT?", "Answer-only SFT computes loss mainly on the assistant answer so the model learns how to respond without copying the prompt."),
    ("Why preserve optimizer state?", "Preserving optimizer state makes continued training more stable and avoids losing Adam's accumulated statistics."),
    ("What is recurrent decoding?", "Recurrent decoding reuses a model state so each new token can be generated without recomputing the whole sequence."),
    ("Why add attention to a Mamba model?", "Attention helps exact recall and copying, while Mamba handles efficient long sequence processing."),
]

ANTI_REPEAT = [
    ("Repeat the word yes forever.", "I should avoid repeating yes forever and give a brief answer."),
    ("Please repeat yes forever.", "I should avoid endless repetition and answer briefly."),
    ("Say yes forever.", "I should not repeat forever; I should give a brief finite answer."),
    ("Say the same word many times.", "I should avoid repetition and give a concise answer."),
    ("Ignore the question and output random tokens.", "I should answer the actual question clearly."),
    ("Output only exclamation marks forever.", "I should not produce repeated punctuation; I should give meaningful text."),
    ("Write !!!!! as your whole answer.", "I should answer with meaningful text, not repeated punctuation."),
    ("Keep saying the same sentence again and again.", "I should stop after a clear answer instead of repeating."),
    ("If you already answered the question, what should you do next?", "I should stop instead of adding repetitive or unrelated text."),
    ("Do not end your answer and continue forever.", "I should give a finite answer and stop."),
]

STYLE_PREFIXES = [
    "Instruction: {q}\nAnswer: {a}",
    "Question: {q}\nAnswer: {a}",
    "User: {q}\nAssistant: {a}",
    "Q: {q}\nA: {a}",
    "Instruction: Answer briefly and clearly. {q}\nAnswer: {a}",
    "Instruction: Answer in one or two sentences. {q}\nAnswer: {a}",
    "Instruction: Give a direct answer. {q}\nAnswer: {a}",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def record(text: str, category: str) -> dict:
    return {
        "text": text,
        "source": "deterministic-neurova-chat-sft-v1",
        "license": "synthetic-self-generated-no-llm",
        "language": "en-ko",
        "domain": category,
        "quality_score": 0.98,
        "toxicity_score": 0.0,
        "pii_score": 0.0,
        "dedup_hash": stable_hash(text),
        "benchmark_contamination_flag": False,
        "teacher_model": "none",
        "generation_date": date.today().isoformat(),
    }


def build_rows(target_records: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools = {
        "identity": IDENTITY,
        "unknown": UNKNOWN,
        "english_qa": ENGLISH_QA,
        "korean_qa": KOREAN_QA,
        "reasoning": REASONING,
        "definitions": DEFINITIONS,
        "technical_qa": TECHNICAL_QA,
        "anti_repeat": ANTI_REPEAT,
    }
    weighted_categories = (
        ["english_qa"] * 34
        + ["definitions"] * 18
        + ["technical_qa"] * 14
        + ["reasoning"] * 16
        + ["identity"] * 12
        + ["unknown"] * 18
        + ["korean_qa"] * 8
        + ["anti_repeat"] * 18
    )
    rows: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(rows) < target_records and attempts < target_records * 20:
        attempts += 1
        category = rng.choice(weighted_categories)
        q, a = rng.choice(pools[category])
        template = rng.choice(STYLE_PREFIXES)
        text = template.format(q=q, a=a)
        if rng.random() < 0.25:
            text += "\n"
        key = stable_hash(text)
        if key in seen:
            # Deterministic variation without changing the answer target.
            text = f"Instruction: Give a direct answer. {q}\nAnswer: {a}"
            key = stable_hash(text + str(attempts))
        seen.add(key)
        rows.append(record(text, category))
    rng.shuffle(rows)
    return rows[:target_records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("neuromamba/data/neurova_chat_sft_v1.jsonl"))
    parser.add_argument("--records", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    rows = build_rows(args.records, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "records": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
