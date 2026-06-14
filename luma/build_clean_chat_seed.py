from __future__ import annotations

import argparse
import json
from pathlib import Path

from .chat_format import chatml


SYSTEM = "You are LUMA, a concise helpful assistant."


BASE_PAIRS = [
    ("hi", "Hello. I am LUMA. How can I help?"),
    ("hello", "Hello. I am LUMA. How can I help?"),
    ("who are you?", "I am LUMA, a local experimental assistant built to answer clearly and remember useful context."),
    ("너는 누구야?", "저는 LUMA입니다. 짧고 명확하게 답하고 필요한 정보를 기억하도록 설계된 실험용 로컬 어시스턴트입니다."),
    ("what is machine learning?", "Machine learning is a way to train computers to learn patterns from data and use them to make predictions or decisions."),
    ("한국어로 짧게 인공지능이 뭐야?", "인공지능은 컴퓨터가 언어 이해, 추론, 예측 같은 지능적인 작업을 하도록 만드는 기술입니다."),
    ("what is evaluation?", "Evaluation measures whether a model works well on tasks that matter."),
    ("what is overfitting?", "Overfitting is when a model memorizes training examples too closely and performs poorly on new inputs."),
    ("what should you do if you do not know?", "If I do not know, I should say so and explain what evidence would be needed."),
    ("answer briefly: why avoid repetition?", "Repetition makes answers hard to read and usually means the model is losing control."),
    ("2 + 3 = ?", "2 + 3 = 5."),
    ("If all roses are flowers and red roses are roses, are red roses flowers?", "Yes. Red roses are flowers because they are roses, and all roses are flowers."),
    ("Summarize: Memory helps models keep useful facts across a long conversation.", "Memory helps a model retain useful facts over long conversations."),
    ("Give JSON with key ok true.", "{\"ok\": true}"),
    ("Say you are uncertain when evidence is missing.", "I do not have enough evidence to answer confidently."),
    ("코덱스와 모델 학습의 차이를 짧게 설명해줘.", "코덱스는 코드를 다루는 실행 에이전트이고, 모델 학습은 데이터로 모델 파라미터를 조정하는 과정입니다."),
]


DEFINITIONS = {
    "attention": "Attention lets a model compare tokens or states and focus on relevant information.",
    "tokenizer": "A tokenizer converts text into token ids that a model can process.",
    "checkpoint": "A checkpoint is a saved model state that can be loaded later.",
    "loss": "Loss measures how wrong a model is during training.",
    "ablation": "An ablation removes or changes one component to test whether it matters.",
    "memory slot": "A memory slot is a persistent vector state used to store and update information.",
    "copy task": "A copy task checks whether a model can reproduce given text exactly.",
    "hallucination": "Hallucination is when a model states unsupported or false information as if it were true.",
    "gradient descent": "Gradient descent is an optimization method that improves parameters by moving them in the direction that lowers loss.",
    "JSON": "JSON is a text format for structured data using objects, arrays, strings, numbers, booleans, and null.",
    "inference": "Inference is running a trained model on input to produce an output.",
    "evidence": "Evidence is information that supports or challenges a claim.",
    "debugging": "Debugging is the process of finding the cause of a problem and verifying a fix.",
    "generalization": "Generalization is when a model performs well on examples it did not memorize.",
}


KOREAN = [
    ("짧게 답해. 좋은 답변이란?", "좋은 답변은 질문에 직접 답하고, 근거가 부족하면 불확실성을 분명히 말하는 답변입니다."),
    ("반복 붕괴가 뭐야?", "반복 붕괴는 모델이 같은 말이나 무의미한 토큰을 계속 반복하는 실패 상태입니다."),
    ("모르면 어떻게 해야 해?", "모르면 모른다고 말하고, 어떤 정보가 필요할지 설명해야 합니다."),
    ("메모리 슬롯은 왜 필요해?", "메모리 슬롯은 긴 문맥에서 중요한 사실을 저장하고 다시 쓰기 위해 필요합니다."),
    ("한국어로 짧게 머신러닝을 설명해줘.", "머신러닝은 데이터에서 패턴을 배워 새로운 입력에 대해 예측하거나 판단하는 기술입니다."),
    ("짧게 말해. 모르는 질문을 받으면?", "모르면 추측하지 말고 모른다고 말한 뒤 필요한 근거를 설명해야 합니다."),
    ("JSON으로 ok가 true인 값을 줘.", "{\"ok\": true}"),
    ("한 문장으로 디버깅이 뭐야?", "디버깅은 문제의 원인을 찾고 수정이 실제로 효과가 있는지 확인하는 과정입니다."),
]


UNCERTAINTY = [
    ("Who won the private test that is not described here?", "I do not have enough evidence to answer that."),
    ("Answer only if the evidence is present: What is Mira's access code?", "I do not have enough evidence to answer that."),
    ("If a user asks for an unsupported fact, what should you do?", "I should say that I do not know and ask for or look for reliable evidence."),
    ("Should you invent a citation when no source is available?", "No. I should not invent citations or unsupported evidence."),
]


JSON_TASKS = [
    ("Return JSON with name LUMA and ok true.", "{\"name\": \"LUMA\", \"ok\": true}"),
    ("Return compact JSON for count three.", "{\"count\": 3}"),
    ("Given status ready, return JSON with status only.", "{\"status\": \"ready\"}"),
    ("Return JSON: task is copy and success is false.", "{\"success\": false, \"task\": \"copy\"}"),
]


REASONING = [
    ("If every compiler is a tool and Nova is a compiler, is Nova a tool?", "Yes. Nova is a tool because every compiler is a tool."),
    ("A box has 3 red keys and 2 blue keys. How many keys are there?", "There are 5 keys."),
    ("If the rule says answer with one word and the answer is Seoul, what should you output?", "Seoul"),
    ("Choose the better answer: supported by evidence or made up confidently?", "Supported by evidence is better."),
    ("A user says 9 + 4 = 12. Is that correct?", "No. 9 + 4 = 13."),
]


STYLE = [
    ("Answer in one sentence. What makes an answer useful?", "A useful answer directly addresses the question, includes the needed detail, and avoids unsupported claims."),
    ("Make this shorter: The system should continue to operate in a reliable way even when the input changes.", "The system should stay reliable when input changes."),
    ("What should a concise assistant avoid?", "A concise assistant should avoid repetition, filler, and unsupported guesses."),
    ("How should you answer a simple factual question?", "I should answer directly and add only the context needed to make the answer clear."),
]


COPY_AND_EXTRACT = [
    ("Copy exactly: AX-1042-BETA", "AX-1042-BETA"),
    ("Return only the value after code=: code=NV-778", "NV-778"),
    ("Extract the city from JSON {\"city\":\"Seoul\",\"score\":7}.", "Seoul"),
    ("Extract the score from JSON {\"city\":\"Oslo\",\"score\":42}.", "42"),
]


def build_pairs() -> list[tuple[str, str]]:
    pairs = list(BASE_PAIRS)
    for term, definition in DEFINITIONS.items():
        pairs.append((f"What is {term}?", definition))
        pairs.append((f"Define {term} briefly.", definition))
    pairs.extend(KOREAN)
    pairs.extend(UNCERTAINTY)
    pairs.extend(JSON_TASKS)
    pairs.extend(REASONING)
    pairs.extend(STYLE)
    pairs.extend(COPY_AND_EXTRACT)
    for a in range(1, 10):
        for b in range(1, 6):
            pairs.append((f"What is {a} + {b}?", f"{a} + {b} = {a + b}."))
            pairs.append((f"Answer with the number only: {a} + {b}", str(a + b)))
    for a in range(2, 10):
        for b in range(1, 5):
            pairs.append((f"What is {a} times {b}?", f"{a} times {b} is {a * b}."))
    topics = ["attention", "tokenizer", "checkpoint", "memory slot", "debugging", "evidence"]
    for topic in topics:
        pairs.append((f"Explain {topic} in one short sentence.", DEFINITIONS[topic]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean single-format ChatML SFT set.")
    parser.add_argument("--out", default="data/luma_clean_chatml_sft_v1.jsonl")
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    pairs = build_pairs()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for _ in range(args.repeat):
            for user, assistant in pairs:
                row = {"text": chatml(SYSTEM, user, assistant), "source": "luma-clean-chatml-sft-v1"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    print(json.dumps({"out": str(out), "records": count, "unique_pairs": len(pairs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
