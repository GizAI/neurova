from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from .model import LUMAConfig, LUMALM
from .tokenizer import LUMATokenizer, assert_tokenizer_contract, build_tokenizer
from .chat_format import IM_END, chatml


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
    return EvalCase("copy", chatml("You are LUMA, a precise memory assistant.", f"Copy this sequence exactly: {answer}"), answer)


def make_phonebook(rng: random.Random) -> EvalCase:
    pairs = [(rand_word(rng), f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}") for _ in range(3)]
    key, value = rng.choice(pairs)
    book = "; ".join(f"{name}: {phone}" for name, phone in pairs)
    return EvalCase("phonebook", chatml("You are LUMA, a precise memory assistant.", f"Phonebook: {book}. What is the phone number for {key}?"), value)


def make_json_field(rng: random.Random) -> EvalCase:
    obj = {"city": rng.choice(["Seoul", "Boston", "Oslo"]), "score": rng.randint(10, 99), "tag": rand_word(rng)}
    key = rng.choice(list(obj.keys()))
    body = json.dumps(obj, sort_keys=True)
    return EvalCase("json_field", chatml("You are LUMA, a precise memory assistant.", f"Given JSON {body}, return the value of field `{key}` only."), str(obj[key]))


def make_recall(rng: random.Random) -> EvalCase:
    name = rng.choice(["Mina", "Joon", "Ara", "Noah"])
    obj = rng.choice(["blue key", "red notebook", "silver coin", "green map"])
    place = rng.choice(["seoul", "busan", "lab7", "quiet library"])
    prompt = (
        f"Facts: {name} owns the {obj}. A distractor says {rand_word(rng)} owns a spare map. "
        f"{name} should go to {place}. What object belongs to {name}?"
    )
    return EvalCase("recall", chatml("You are LUMA, a precise memory assistant.", prompt), obj)


CASE_BUILDERS = [make_copy, make_phonebook, make_json_field, make_recall]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LUMA on exact memory tasks.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cases", type=int, default=100)
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--gap-lines", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260614)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16" if torch.cuda.is_available() else "fp32")
    p.add_argument("--out", default="")
    p.add_argument(
        "--ablation",
        choices=["normal", "no_slots", "random_slot_keys", "no_copy", "no_slots_no_copy", "no_local_attention"],
        default="normal",
    )
    p.add_argument("--compare-ablations", action="store_true")
    return p.parse_args()


def stop_ids(tokenizer: LUMATokenizer) -> set[int]:
    ids = {int(tokenizer.eos_id)}
    for text in (IM_END, "<|im_start|>user", "<|im_start|>system"):
        encoded = tokenizer.encode(text, add_bos=False, add_eos=False)
        if len(encoded) == 1:
            ids.add(int(encoded[0]))
    return ids


def add_memory_gap(case: EvalCase, rng: random.Random, gap_lines: int) -> EvalCase:
    marker = "<|im_start|>assistant\n"
    if gap_lines <= 0 or marker not in case.prompt:
        return case
    fillers = []
    for _ in range(gap_lines):
        fillers.append(
            "Irrelevant note: "
            + " ".join(rand_word(rng) for _ in range(10))
            + "."
        )
    prompt = case.prompt.replace(marker, "\n" + "\n".join(fillers) + "\n" + marker, 1)
    return EvalCase(case.task, prompt, case.answer)


@torch.no_grad()
def generate(
    model: LUMALM,
    tokenizer: LUMATokenizer,
    prompt: str,
    max_new: int,
    device: str,
    *,
    ablation: str = "normal",
) -> str:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids[-512:]], dtype=torch.long, device=device)
    out = model(x, return_slots=True, ablation=ablation)
    slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
    stops = stop_ids(tokenizer)
    generated: list[int] = []
    for _ in range(max_new):
        logits = out.logits[0, -1]
        for special_id in {tokenizer.pad_id, tokenizer.bos_id} - {tokenizer.eos_id}:
            if 0 <= int(special_id) < logits.numel():
                logits[int(special_id)] = -torch.inf
        next_id = int(torch.argmax(logits).item())
        ids.append(next_id)
        if next_id in stops:
            break
        generated.append(next_id)
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        out = model(x, slots_in=slots, return_slots=True, ablation=ablation)
        slots = LUMALM.detach_slots(out.slots) if out.slots is not None else None
    return prompt + tokenizer.decode(generated)


def extract_answer(text: str) -> str:
    marker = "<|im_start|>assistant"
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    if IM_END in text:
        text = text.split(IM_END, 1)[0]
    return text.strip().splitlines()[0].strip()


def evaluate(
    model: LUMALM,
    tokenizer: LUMATokenizer,
    *,
    cases: int,
    max_new: int,
    gap_lines: int,
    seed: int,
    device: str,
    ablation: str,
) -> dict:
    rng = random.Random(seed)
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    samples = []
    for idx in range(cases):
        case = add_memory_gap(rng.choice(CASE_BUILDERS)(rng), rng, gap_lines)
        raw = generate(model, tokenizer, case.prompt, max_new, device, ablation=ablation)
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
    return {"ablation": ablation, "overall": overall_correct / overall_total, "by_task": by_task, "samples": samples}


def main() -> None:
    args = parse_args()
    payload = torch.load(Path(args.ckpt), map_location=args.device, weights_only=True)
    raw_cfg = payload["config"]
    cfg = LUMAConfig(**raw_cfg)
    tokenizer = build_tokenizer(cfg.tokenizer_backend, cfg.qwen_tokenizer_path, cfg.bytepatch_vocab_path)
    assert_tokenizer_contract(raw_cfg, tokenizer)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    model = LUMALM(cfg).to(device=args.device, dtype=dtype)
    model.load_state_dict(payload["model"])
    model.eval()
    ablations = (
        ["normal", "no_slots", "random_slot_keys", "no_copy", "no_slots_no_copy", "no_local_attention"]
        if args.compare_ablations
        else [args.ablation]
    )
    results = [
        evaluate(
            model,
            tokenizer,
            cases=args.cases,
            max_new=args.max_new,
            gap_lines=args.gap_lines,
            seed=args.seed,
            device=args.device,
            ablation=ablation,
        )
        for ablation in ablations
    ]
    output = {"ckpt": args.ckpt, "results": results}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
