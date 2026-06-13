from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import LUMAConfig, LUMALM
from .tokenizer import ByteTokenizer


WORDS = [
    "amber", "brisk", "cedar", "delta", "ember", "fable", "glade", "harbor",
    "ion", "juniper", "lumen", "mosaic", "nova", "onyx", "quartz", "raven",
]


@dataclass
class EvalCase:
    task: str
    prompt: str
    answer: str


def rand_word(rng: random.Random) -> str:
    return rng.choice(WORDS) + rng.choice("abcdefghijklmnopqrstuvwxyz")


def make_copy(rng: random.Random) -> EvalCase:
    items = [rand_word(rng) for _ in range(rng.randint(2, 5))]
    answer = " ".join(items)
    return EvalCase("copy", f"Instruction: Copy this sequence exactly: {answer}\nAnswer:", answer)


def make_phonebook(rng: random.Random) -> EvalCase:
    pairs = [(rand_word(rng), f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}") for _ in range(3)]
    key, value = rng.choice(pairs)
    book = "; ".join(f"{name}: {phone}" for name, phone in pairs)
    return EvalCase("phonebook", f"Instruction: Phonebook: {book}. What is the phone number for {key}?\nAnswer:", value)


def make_json_field(rng: random.Random) -> EvalCase:
    obj = {"city": rng.choice(["Seoul", "Boston", "Oslo"]), "score": rng.randint(10, 99), "tag": rand_word(rng)}
    key = rng.choice(list(obj.keys()))
    body = json.dumps(obj, sort_keys=True)
    return EvalCase("json_field", f"Instruction: Given JSON {body}, return the value of field `{key}` only.\nAnswer:", str(obj[key]))


def make_recall(rng: random.Random) -> EvalCase:
    name = rng.choice(["Mina", "Joon", "Ara", "Noah"])
    obj = rng.choice(["blue key", "red notebook", "silver coin", "green map"])
    place = rng.choice(["seoul", "busan", "lab7", "quiet library"])
    prompt = (
        f"Instruction: Facts: {name} owns the {obj}. A distractor says {rand_word(rng)} owns a spare map. "
        f"{name} should go to {place}. What object belongs to {name}?\nAnswer:"
    )
    return EvalCase("recall", prompt, obj)


CASE_BUILDERS = [make_copy, make_phonebook, make_json_field, make_recall]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LUMA on exact memory tasks.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cases", type=int, default=100)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def generate(model: LUMALM, tokenizer: ByteTokenizer, prompt: str, max_new: int, device: str) -> str:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    for _ in range(max_new):
        x = torch.tensor([ids[-512:]], dtype=torch.long, device=device)
        logits = model(x).logits[0, -1]
        logits[tokenizer.pad_id] = -torch.inf
        logits[tokenizer.bos_id] = -torch.inf
        next_id = int(torch.argmax(logits).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)


def extract_answer(text: str) -> str:
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    return text.strip().splitlines()[0].strip()


def main() -> None:
    args = parse_args()
    tokenizer = ByteTokenizer()
    payload = torch.load(Path(args.ckpt), map_location=args.device, weights_only=True)
    model = LUMALM(LUMAConfig(**payload["config"])).to(args.device)
    model.load_state_dict(payload["model"])
    model.eval()
    rng = random.Random(args.seed)
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    samples = []
    for idx in range(args.cases):
        case = rng.choice(CASE_BUILDERS)(rng)
        raw = generate(model, tokenizer, case.prompt, args.max_new, args.device)
        pred = extract_answer(raw)
        ok = pred == case.answer
        totals[case.task] = totals.get(case.task, 0) + 1
        correct[case.task] = correct.get(case.task, 0) + int(ok)
        if len(samples) < 12:
            samples.append({"task": case.task, "expected": case.answer, "pred": pred, "ok": ok})
    by_task = {
        task: {"correct": correct.get(task, 0), "total": total, "accuracy": correct.get(task, 0) / total}
        for task, total in sorted(totals.items())
    }
    overall_total = sum(totals.values())
    overall_correct = sum(correct.values())
    print(json.dumps({"overall": overall_correct / overall_total, "by_task": by_task, "samples": samples}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
