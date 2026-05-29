#!/usr/bin/env python3
"""Korea comprehension test for v9 Construction Learner engine."""

import sys; sys.path.insert(0, ".")
from neurova.engine import (
    NeurovaEngine, WorldModel, compile_update,
    process_doc, answer_question, _parse_q, sent_split
)

text = (
    "Korea is a peninsular region in East Asia consisting of the Korean Peninsula, "
    "Jeju Island, and smaller islands. Since the end of World War II in Asia in 1945, "
    "it has been politically divided at or near the 38th parallel between North Korea "
    "(Democratic People's Republic of Korea; DPRK) and South Korea (Republic of Korea; ROK). "
    "Both countries proclaimed independence in 1948, and the two countries fought the "
    "Korean War from 1950 to 1953. The region is bordered by China to the north and "
    "Russia to the northeast, across the Amnok (Yalu) and Duman (Tumen) rivers, and is "
    "separated from Japan to the southeast by the Korea Strait."
)

print("=" * 60)
print("Korea Comprehension Test — v9 Construction Learner")
print("=" * 60)

# === METHOD 1: Full Engine ===
print("\n[Method 1] Full Engine:")
engine = NeurovaEngine()
engine.hear(text)

queries = [
    "What is Korea?",
    "Is Korea in East Asia?",
    "Is Korea divided?",
    "Is Korea a peninsular region?",
    "What is in the southeast of Korea?",
    "How are South and North Korea divided?",
    "What borders Korea to the north?",
    "What is Korea separated from?",
    "Where is Jeju Island?",
    "Where is the Tumen River?",
]

pass_count = 0
for q in queries:
    a = engine.answer(q)
    expected_good = []
    if q == "What is Korea?": expected_good = ["peninsular", "korean peninsula", "jeju island", "east asia"]
    elif q == "Is Korea in East Asia?": expected_good = ["yes"]
    elif q == "Is Korea divided?": expected_good = ["yes"]
    elif q == "Is Korea a peninsular region?": expected_good = ["yes"]
    elif q == "What is in the southeast of Korea?": expected_good = ["japan"]
    elif q == "How are South and North Korea divided?": expected_good = ["north korea", "south korea", "38th"]
    elif q == "What borders Korea to the north?": expected_good = ["china"]
    elif q == "What is Korea separated from?": expected_good = ["japan"]
    elif q == "Where is Jeju Island?": expected_good = ["east asia"]
    elif q == "Where is the Tumen River?": expected_good = ["korea", "russia", "east asia"]
    
    al = a.lower()
    ok = any(expected in al for expected in expected_good) if expected_good else True
    status = "✓" if ok else f"✗ (expected: {expected_good})"
    if ok: pass_count += 1
    print(f"  {status} Q: {q}")
    print(f"      A: {a}")

print(f"\n[Result] {pass_count}/{len(queries)} passed")
print()

# === METHOD 2: Direct WorldModel Inspection ===
print("\n[Method 2] WorldModel State:")
model = WorldModel()
process_doc(text, model)

for name in sorted(model.entities.keys()):
    e = model.entities[name]
    bits = []
    if e.attributes: bits.append(f"attrs={dict(list(e.attributes.items())[:3])}")
    if e.properties: bits.append(f"props={e.properties}")
    if e.location: bits.append(f"loc={e.location}")
    if e.parts: bits.append(f"parts={e.parts}")
    if e.part_of: bits.append(f"of={e.part_of}")
    if e.borders: bits.append(f"border={e.borders}")
    if e.direction_relations: bits.append(f"dir={e.direction_relations}")
    if e.separated_from: bits.append(f"sep={e.separated_from}")
    if e.divided_into: bits.append(f"div={e.divided_into}")
    if e.division_criterion: bits.append(f"crit={e.division_criterion}")
    if e.located_across: bits.append(f"across={e.located_across}")
    if bits: print(f"  {name}: {', '.join(bits)}")
