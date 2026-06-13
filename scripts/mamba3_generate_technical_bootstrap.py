#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path


def digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def governed(text: str, source: str, domain: str, quality: float) -> dict:
    return {
        "text": text.strip(),
        "source": source,
        "license": "project-generated",
        "language": "en",
        "domain": domain,
        "quality_score": quality,
        "toxicity_score": 0.0,
        "pii_score": 0.0,
        "dedup_hash": digest(text),
        "benchmark_contamination_flag": False,
        "teacher_model": "none",
        "generation_date": date.today().isoformat(),
    }


def code_record(rng: random.Random, i: int) -> dict:
    a = rng.randint(2, 19)
    b = rng.randint(3, 29)
    c = rng.randint(1, 11)
    name = rng.choice(["score", "total", "index", "weight", "offset", "count"])
    fn = rng.choice(["normalize", "combine", "score_item", "update_total", "filter_values"])
    code = f"""Python code note {i}.

The function `{fn}` should keep each step explicit and testable.

```python
def {fn}({name}: int) -> int:
    doubled = {name} * {a}
    shifted = doubled + {b}
    return shifted - {c}


assert {fn}(10) == {10 * a + b - c}
```

When tracing this code, first bind the input variable, then evaluate the multiplication, then the addition, then the subtraction. The final return value is deterministic and should not depend on hidden state."""
    return governed(code, "deterministic_technical_bootstrap", "code", 0.76)


def math_record(rng: random.Random, i: int) -> dict:
    x = rng.randint(2, 12)
    y = rng.randint(2, 12)
    z = rng.randint(1, 9)
    total = x * y + z
    text = f"""Math reasoning note {i}.

Consider the expression `{x} * {y} + {z}`. Multiplication has higher precedence than addition, so the product is computed first. The product is `{x * y}`. Adding `{z}` gives `{total}`.

A reliable answer should show the operation order, compute one intermediate value at a time, and finish with the exact integer result `{total}`."""
    return governed(text, "deterministic_technical_bootstrap", "math", 0.78)


def science_record(rng: random.Random, i: int) -> dict:
    topics = [
        (
            "evidence",
            "A scientific claim is stronger when independent observations point to the same explanation.",
            "A careful scientist separates what was measured from what was inferred.",
        ),
        (
            "energy",
            "Energy is conserved in an isolated system, but it can change form.",
            "Tracking inputs, outputs, and losses keeps the explanation concrete.",
        ),
        (
            "experiments",
            "A controlled experiment changes one factor while keeping other important factors stable.",
            "That design makes it easier to connect an observed effect to a cause.",
        ),
        (
            "models",
            "A model is useful when it predicts observations that were not used to build it.",
            "Wrong predictions are not just failures; they tell researchers where the model is incomplete.",
        ),
    ]
    topic, claim, detail = rng.choice(topics)
    text = f"""Science explanation note {i}: {topic}.

{claim} {detail} Good scientific writing states the claim, names the evidence, explains the mechanism, and avoids pretending that uncertainty is zero."""
    return governed(text, "deterministic_technical_bootstrap", "science", 0.76)


def structured_record(rng: random.Random, i: int) -> dict:
    city = rng.choice(["Boston", "Denver", "Seattle", "Austin", "Chicago"])
    temp = rng.randint(8, 31)
    humidity = rng.randint(20, 85)
    text = f"""Structured data note {i}.

JSON:
```json
{{"city": "{city}", "temperature_c": {temp}, "humidity_percent": {humidity}}}
```

YAML:
```yaml
city: {city}
temperature_c: {temp}
humidity_percent: {humidity}
```

The field `city` is `{city}`. The field `temperature_c` is `{temp}`. The field `humidity_percent` is `{humidity}`. Exact field extraction requires copying the requested value without inventing a new key."""
    return governed(text, "deterministic_technical_bootstrap", "code", 0.77)


GENERATORS = [code_record, math_record, science_record, structured_record]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate governed deterministic technical bootstrap shards.")
    parser.add_argument("--out", type=Path, default=Path("data/technical_bootstrap_v1.jsonl"))
    parser.add_argument("--records", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i in range(args.records * 3):
            rec = GENERATORS[i % len(GENERATORS)](rng, i)
            if rec["dedup_hash"] in seen:
                continue
            seen.add(rec["dedup_hash"])
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.records:
                break
    print(json.dumps({"out": str(args.out), "records": written}, indent=2))


if __name__ == "__main__":
    main()
