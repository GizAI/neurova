from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RLVRItem:
    prompt: str
    answer: str
    verifier: str
    target: str | list[str]


def iter_rlvr(path: Path) -> Iterable[RLVRItem]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield RLVRItem(
                prompt=str(obj["prompt"]),
                answer=str(obj["answer"]),
                verifier=str(obj["verifier"]),
                target=obj["target"],
            )


def verify_item(item: RLVRItem) -> bool:
    answer = item.answer.strip()
    if item.verifier == "exact":
        return answer == str(item.target).strip()
    if item.verifier == "casefold_exact":
        return answer.casefold() == str(item.target).strip().casefold()
    if item.verifier == "contains_all":
        targets = item.target if isinstance(item.target, list) else [str(item.target)]
        return all(str(target).casefold() in answer.casefold() for target in targets)
    if item.verifier == "python_assert":
        return _verify_python_assert(answer, str(item.target))
    raise ValueError(f"unknown verifier: {item.verifier}")


def _verify_python_assert(answer: str, assertion: str) -> bool:
    code = answer + "\n" + assertion + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        result = subprocess.run(
            ["python", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    return result.returncode == 0


def verify_file(path: Path) -> dict[str, int | float]:
    total = 0
    passed = 0
    for item in iter_rlvr(path):
        total += 1
        passed += int(verify_item(item))
    return {
        "total": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
    }
