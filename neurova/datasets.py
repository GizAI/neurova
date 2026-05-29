from __future__ import annotations
import json, random
from pathlib import Path
from typing import Iterable, List

ENTITIES = ["kibo", "luna", "orion", "alice", "bob", "charlie", "junho", "철수", "영희", "민수", "준호"]
TYPES = ["robot", "machine", "animal", "bird", "mammal", "rover", "ceo", "active", "mineral"]
ACTIONS = ["rain", "heat", "press switch", "market shock", "supply shock"]
EFFECTS = ["wet ground", "steam", "light on", "volatility", "higher prices"]
YEARS = ["2024", "2025", "2026", "2027", "2028"]


def _add(rows: List[dict], text: str, ir_type: str, slots: dict) -> None:
    rows.append({"text": text, "ir_type": ir_type, "slots": slots})


def generate_nl_ir_examples(n: int = 500, seed: int = 7) -> List[dict]:
    """Generate a no-LLM structured NL→IR corpus.

    The generator is IR-first: every surface utterance is rendered from a typed
    target structure. This is intentionally not a language model corpus; it is a
    supervised semantic-compiler seed corpus.
    """
    rng = random.Random(seed)
    rows: List[dict] = []
    templates = [
        "claim", "claim_alias", "neg", "comparison_en", "comparison_en_alias", "comparison_ko",
        "causal", "causal_alias", "causal_ko", "temporal", "temporal_alias", "exception",
    ]
    for _ in range(n):
        kind = rng.choice(templates)
        if kind == "claim":
            e, t = rng.choice(ENTITIES), rng.choice(TYPES)
            _add(rows, f"{e} is {t}", "ClaimIR", {"subject": e, "relation": "is", "object": t})
        elif kind == "claim_alias":
            e, t = rng.choice(ENTITIES), rng.choice(TYPES)
            # Still maps to relation=is; surface is broader than old regex.
            surface = rng.choice([f"{e} belongs to {t}", f"{e} counts as {t}", f"{e} is classified as {t}"])
            _add(rows, surface, "ClaimIR", {"subject": e, "relation": "is", "object": t})
        elif kind == "neg":
            e, t = rng.choice(ENTITIES), rng.choice(TYPES)
            surface = rng.choice([f"{e} is not {t}", f"{e} does not count as {t}", f"{e} is no {t}"])
            _add(rows, surface, "NegatedClaimIR", {"subject": e, "relation": "is", "object": t, "polarity": "negative"})
        elif kind == "comparison_en":
            a, b = rng.sample(ENTITIES[:7], 2)
            _add(rows, f"{a} is taller than {b}", "ComparisonIR", {"left": a, "comparator": "greater_than", "right": b})
        elif kind == "comparison_en_alias":
            a, b = rng.sample(ENTITIES[:7], 2)
            surface = rng.choice([f"{a} exceeds {b}", f"{a} is above {b}", f"{a} outranks {b}"])
            _add(rows, surface, "ComparisonIR", {"left": a, "comparator": "greater_than", "right": b})
        elif kind == "comparison_ko":
            a, b = rng.sample(["철수", "영희", "민수", "준호"], 2)
            surface = rng.choice([f"{a}는 {b}보다 크다", f"{a}는 {b}보다 높다", f"{a}는 {b}보다 우위다"])
            _add(rows, surface, "ComparisonIR", {"left": a, "comparator": "greater_than", "right": b})
        elif kind == "causal":
            a, e = rng.choice(ACTIONS), rng.choice(EFFECTS)
            _add(rows, f"{a} causes {e}", "CausalClaimIR", {"cause": a, "effect": e})
        elif kind == "causal_alias":
            a, e = rng.choice(ACTIONS), rng.choice(EFFECTS)
            surface = rng.choice([f"{a} leads to {e}", f"{e} happens because of {a}", f"{a} makes {e} happen"])
            _add(rows, surface, "CausalClaimIR", {"cause": a, "effect": e})
        elif kind == "causal_ko":
            a, e = rng.choice(["비", "열", "스위치 누름"]), rng.choice(["젖은 땅", "증기", "불 켜짐"])
            surface = rng.choice([f"{a} 때문에 {e}이 발생한다", f"{a}가 {e}의 원인이다"])
            _add(rows, surface, "CausalClaimIR", {"cause": a, "effect": e})
        elif kind == "temporal":
            e, t, year = rng.choice(ENTITIES), rng.choice(TYPES), rng.choice(YEARS)
            _add(rows, f"on {year} {e} is {t}", "TemporalClaimIR", {"subject": e, "relation": "is", "object": t, "time_expr": year})
        elif kind == "temporal_alias":
            e, t, year = rng.choice(ENTITIES), rng.choice(TYPES), rng.choice(YEARS)
            surface = rng.choice([f"in {year} {e} is {t}", f"during {year} {e} is {t}", f"{year}: {e} is {t}"])
            _add(rows, surface, "TemporalClaimIR", {"subject": e, "relation": "is", "object": t, "time_expr": year})
        elif kind == "exception":
            subj, domain, obj = rng.choice(ENTITIES), rng.choice(["bird", "robot", "machine"]), rng.choice(["fly", "move", "work"])
            surface = rng.choice([f"{subj} is exception to {domain} can {obj}", f"{subj} is an exception to {domain} can {obj}"])
            _add(rows, surface, "ExceptionIR", {"exception_subject": subj, "condition_object": domain, "conclusion_relation": "can", "conclusion_object": obj})
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def generate_v25_multitask_corpus(n: int = 6000, seed: int = 25) -> List[dict]:
    """Large no-LLM text↔IR / correction / event corpus.

    The rows are IR-first or correction-first and support structured semantic
    perception. They deliberately avoid next-token objectives.
    """
    rows = generate_nl_ir_examples(n // 2, seed=seed)
    rng = random.Random(seed + 77)
    names = ["alice", "bob", "charlie", "kibo", "orion", "seoul", "busan"]
    objs = ["book", "package", "door", "rocks", "ceo", "machine"]
    verbs = ["dominates", "outruns", "towers over", "ranks above", "sparks", "brings"]
    for _ in range(n - len(rows)):
        kind = rng.choice(["correction", "event", "belief", "goal", "temporal_interval", "exception", "question", "korean"])
        a, b = rng.sample(names, 2)
        if kind == "correction":
            v = rng.choice(verbs)
            if v in {"sparks", "brings"}:
                text = rng.choice([f'When I say "A {v} B", it means A causes B', f'"A {v} B" should be understood as A causes B'])
                _add(rows, text, "ToolCallIR", {"tool_name": "learn_construction", "target": "causal(A,B)"})
            else:
                text = rng.choice([f'When I say "A {v} B", it means A is greater than B', f'"A {v} B" is equivalent to A greater_than B'])
                _add(rows, text, "ToolCallIR", {"tool_name": "learn_construction", "target": "compare(A,greater_than,B)"})
        elif kind == "event":
            obj = rng.choice(objs[:2])
            text = rng.choice([f"{a} gave {b} a {obj}", f"{a} gave a {obj} to {b}", f"{b} received a {obj} from {a}"])
            _add(rows, text, "EventIR", {"actor": a, "action": "give", "patient": obj, "recipient": b})
        elif kind == "belief":
            text = rng.choice([f"{a} believes {b} is CEO", f"{a} thinks {b} is not CEO"])
            _add(rows, text, "BeliefIR", {"holder": a})
        elif kind == "goal":
            text = rng.choice([f"{a} intends to collect rocks", f"{a} plans on collecting rocks"])
            _add(rows, text, "GoalIR", {"agent": a, "desired_state": "collect rocks"})
        elif kind == "temporal_interval":
            y = rng.choice(YEARS[:-1]); y2 = str(int(y)+1)
            text = rng.choice([f"From {y} to {y2}, {a} served as CEO", f"{a} was CEO from {y} through {y2}", f"In {y2}, {a} was not CEO"])
            _add(rows, text, "TemporalClaimIR", {"subject": a, "object": "ceo", "time_expr": y})
        elif kind == "exception":
            text = rng.choice(["Penguins are birds; however, they usually do not fly", "Although penguins are birds, they cannot usually fly"])
            _add(rows, text, "CompositeIR", {"exception_subject": "penguin"})
        elif kind == "question":
            text = rng.choice([f"Is it fair to call {a} a machine?", f"Would {a} qualify as a machine?", f"Does {a} have the book?"])
            _add(rows, text, "QuestionIR", {})
        else:
            text = rng.choice(["영희보다 철수가 더 크다는 말이 맞아?", "철수가 영희보다 더 큰 편이다", "철수는 영희에 비해 우위에 있다", "철수는 영희보다 크지 않다"])
            _add(rows, text, "ComparisonIR", {})
    return rows

def generate_v26_developmental_corpus(n: int = 12000, seed: int = 26) -> List[dict]:
    """Developmental no-LM corpus: text↔IR, correction dialogue, event/world frames,
    coreference, elementary word problems, and memory-promotion cases.
    It is synthetic and verifier-friendly; no next-token target is produced.
    """
    rows = generate_v25_multitask_corpus(n // 2, seed=seed)
    rng = random.Random(seed + 260)
    people = ["alice", "bob", "carol", "dami", "joon", "mina", "sora"]
    objects = ["book", "pencil", "ruler", "package", "box", "notebook"]
    places = ["library", "classroom", "seoul", "busan"]
    for _ in range(n - len(rows)):
        kind = rng.choice(["llm_seed", "correction_prefix", "event_frame", "world_query", "elementary_math", "coref", "belief_question", "temporal_state", "intrinsic_goal"])
        a, b = rng.sample(people, 2)
        obj = rng.choice(objects)
        if kind == "llm_seed":
            surface = rng.choice(["A gives rise to B", "A is regarded as B", "A trails B", "A cannot be classified as B"])
            target = rng.choice(["causal(A,B)", "claim(A,is,B)", "compare(A,less_than,B)", "not_claim(A,is,B)"])
            _add(rows, f'LLM_SEED construction {surface} => {target}', "SeedCandidate", {"surface": surface, "target": target})
        elif kind == "correction_prefix":
            verb = rng.choice(["sparks", "outruns", "absorbs", "trails"])
            meaning = "A causes B" if verb in {"sparks", "absorbs"} else ("A less_than B" if verb == "trails" else "A greater_than B")
            _add(rows, rng.choice([f'Actually, "A {verb} B" means {meaning}', f'No, by "A {verb} B" I mean {meaning}', f'In this domain, "A {verb} B" means {meaning}']), "ToolCallIR", {"tool_name": "learn_construction"})
        elif kind == "event_frame":
            _add(rows, rng.choice([f"{a} bought a {obj} from {b}", f"{b} sold a {obj} to {a}", f"{a} moved the {obj} from {places[0]} to {places[1]}"]), "EventIR", {"action": "world_frame"})
        elif kind == "world_query":
            _add(rows, rng.choice([f"Does {a} have {obj}?", f"Where is the {obj}?"]), "QuestionIR", {"relation": "world_state"})
        elif kind == "elementary_math":
            x, y = rng.randint(2, 30), rng.randint(2, 9)
            _add(rows, rng.choice([f"Mina has {x} marbles and gets {y} more. How many marbles does Mina have?", f"There are {y} boxes with {x} pencils each. How many pencils are there?", f"{x*y} candies are shared equally among {y} children. How many candies does each child get?"]), "ToolCallIR", {"tool_name": "solve_arithmetic"})
        elif kind == "coref":
            _add(rows, f"{a} gave {b} a {obj}. He thanked her.", "EventIR", {"coreference": True})
        elif kind == "belief_question":
            _add(rows, f"Does {a} believe {b} is not CEO?", "QuestionIR", {"target": "BeliefIR"})
        elif kind == "temporal_state":
            _add(rows, f"{a} became principal in 2025. {a} stopped being principal in 2027.", "TemporalClaimIR", {"state_machine": True})
        else:
            _add(rows, "recent failures show belief question weakness", "LearningGoal", {"target": "repair:belief_question"})
    return rows


def generate_v27_developmental_corpus(n: int = 20000, seed: int = 27) -> list[dict]:
    """V27 corpus alias/extension: no-autoregressive text↔IR/correction/event rows.

    It reuses the verified V26 generator and adds metadata for adversarial learning.
    """
    rows = generate_v26_developmental_corpus(n, seed=seed)
    for r in rows:
        r.setdefault('version', 'v27')
        r.setdefault('training_objective', 'text_to_ir_or_correction_patch_not_next_token')
    return rows
