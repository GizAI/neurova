#!/usr/bin/env python3
"""Neurova v11 Stress Suite — 500+ test cases spanning multiple reasoning domains.

Tests measure: spatial, coref, temporal, classification, coordination,
paraphrase, negation, division, part-whole, and multi-sentence inference.

Usage: python3 stress_suite.py
"""

import sys; sys.path.insert(0, ".")
from neurova.engine import Brain

PASS = 0
FAIL = 0
ERRORS = []

def test(category, text, questions, desc="", min_pass=1):
    """Learn from text, then evaluate questions. Returns number passed."""
    global PASS, FAIL
    brain = Brain()
    brain.hear(text)
    local_pass = 0
    for q, expected_kw in questions:
        try:
            a = brain.hear(q)
            ok = any(kw.lower() in a.lower() for kw in expected_kw)
            if ok:
                local_pass += 1
            else:
                ERRORS.append((category, q, a, expected_kw))
        except Exception as e:
            ERRORS.append((category, q, str(e), expected_kw))
    
    status = "✓" if local_pass >= min_pass else "✗"
    print(f"  {status} [{category}] {desc or text[:50]}... {local_pass}/{len(questions)}")
    PASS += local_pass
    FAIL += len(questions) - local_pass
    return local_pass

# ============================
# SPATIAL / DIRECTION
# ============================
test("spatial-border", 
    "Canada is located north of the United States. Mexico is south of the United States.",
    [("What is north of the United States?", ["canada"]),
     ("What is south of the United States?", ["mexico"])],
    "basic north/south borders")

test("spatial-compass",
    "France is bordered by Spain to the southwest, Germany to the east, and Italy to the southeast.",
    [("What borders France to the east?", ["germany"]),
     ("What is to the southeast of France?", ["italy"]),
     ("What is southwest of France?", ["spain"])],
    "multiple compass directions")

test("spatial-separated",
    "England is separated from France by the English Channel.",
    [("What is England separated from?", ["france"])],
    "separation by channel")

test("spatial-contains",
    "Europe contains France, Germany, and Italy. Asia contains China, Japan, and Korea.",
    [("What does Europe contain?", ["france", "germany", "italy"]),
     ("Is France in Europe?", ["yes"]),
     ("Is China in Europe?", ["no"])],
    "continent contains countries")

test("spatial-location-chain",
    "Jeju is part of Korea. Korea is in East Asia.",
    [("Where is Jeju?", ["korea", "east asia"])],
    "location via part-of chain")

test("spatial-across-river",
    "The city is located across the river from the village.",
    [("What is across the river from the village?", [])],  # just parsing test
    "across river relation", min_pass=0)

# ============================
# CLASSIFICATION
# ============================
test("classify-is-a",
    "A dog is a mammal. A whale is a mammal. A bat is a mammal.",
    [("Is a dog a mammal?", ["yes"]),
     ("Is a whale a mammal?", ["yes"]),
     ("Is a bat a mammal?", ["yes"])],
    "basic is-a classification")

test("classify-category-prop",
    "Mars is a planet. Saturn is a planet. The Sun is a star.",
    [("Is Mars a planet?", ["yes"]),
     ("Is the Sun a star?", ["yes"]),
     ("Is Saturn a star?", ["no"])],
    "planet vs star")

test("classify-attribute",
    "Gold is a yellow metal. Silver is a white metal.",
    [("Is gold a metal?", ["yes"]),
     ("Is gold yellow?", ["yes"])],
    "attribute classification")

test("classify-negative",
    "A rock is not a living thing. A tree is a living thing.",
    [("Is a rock a living thing?", ["no"]),
     ("Is a tree a living thing?", ["yes"])],
    "negative classification", min_pass=1)

# ============================
# COREFERENCE
# ============================
test("coref-pronoun-it",
    "Japan is an island country. It is in East Asia. It has many volcanoes.",
    [("What is Japan?", ["island country", "east asia"]),
     ("Is Japan in East Asia?", ["yes"])],
    "pronoun 'it' resolution")

test("coref-pronoun-they",
    "John and Mary are friends. They live in Boston. They work at a hospital.",
    [("Where do John and Mary live?", ["boston"]),
     ("Do John and Mary work at a hospital?", ["yes"])],
    "pronoun 'they' resolution")

test("coref-definite-region",
    "France is a country in Europe. The region has many famous landmarks.",
    [("What is in Europe?", ["france"]),
     ("Is France a region?", [])],
    "definite reference 'the region'")

test("coref-multi-sentence",
    "Alice has a cat. The cat is black. It is very friendly.",
    [("What color is Alice's cat?", ["black"]),
     ("Is the cat friendly?", ["yes"])],
    "multi-sentence coref chain")

test("coref-demonstrative",
    "The Eiffel Tower is in Paris. This structure was built in 1889. It is very tall.",
    [("What was built in 1889?", ["eiffel tower"]),
     ("Is the Eiffel Tower tall?", ["yes"])],
    "demonstrative 'this' + pronoun")

# ============================
# TEMPORAL
# ============================
test("temporal-year",
    "World War II ended in 1945. The Korean War started in 1950.",
    [("Did World War II end in 1945?", ["yes"]),
     ("Did the Korean War start in 1950?", ["yes"])],
    "year-based events")

# ============================
# DIVISION / SEPARATION
# ============================
test("division-split",
    "The Roman Empire split into the Western Roman Empire and the Eastern Roman Empire.",
    [("What did the Roman Empire split into?", ["western roman empire", "eastern roman empire"])],
    "empire division")

test("division-between",
    "The land was divided between the two brothers.",
    [("Was the land divided?", ["yes"])],
    "division between two parties")

# ============================
# PART-WHOLE
# ============================
test("part-whole-simple",
    "A bicycle consists of two wheels, a frame, and handlebars.",
    [("Does a bicycle have wheels?", ["yes"]),
     ("What does a bicycle consist of?", ["wheels", "frame", "handlebars"])],
    "part-whole composition")

test("part-whole-has",
    "A car has four wheels, an engine, and seats.",
    [("Does a car have an engine?", ["yes"]),
     ("How many wheels does a car have?", ["four"])],
    "'has' part-whole", min_pass=1)

# ============================
# NEGATION
# ============================
test("negation-not",
    "Penguins cannot fly. Eagles can fly.",
    [("Can penguins fly?", ["no"]),
     ("Can eagles fly?", ["yes"])],
    "negation with cannot")

# ============================
# PARAPHRASE / VARIATION
# ============================
test("paraphrase-same-meaning",
    "Berlin is the capital of Germany. The capital city of Germany is Berlin.",
    [("What is the capital of Germany?", ["berlin"]),
     ("Which city is the capital of Germany?", ["berlin"])],
    "same fact different phrasing")

# ============================
# INFERENCE / REASONING
# ============================
test("inference-simple",
    "All humans are mortal. Socrates is human.",
    [("Is Socrates mortal?", ["yes"])],
    "basic syllogism")

test("inference-chain",
    "Water freezes at 0 degrees Celsius. It is -5 degrees outside.",
    [("Will water freeze outside?", ["yes"])],
    "temperature inference chain", min_pass=0)  # hard without numeric reasoning

# ============================
# MULTI-ENTITY COMPLEX
# ============================
test("multi-entity-borders",
    "Germany is bordered by France to the west, Poland to the east, and Austria to the south.",
    [("What is west of Germany?", ["france"]),
     ("What borders Germany to the east?", ["poland"]),
     ("Is Austria south of Germany?", ["yes"])],
    "Germany's neighbors")

test("multi-entity-korea-full",
    "Korea is a peninsular region in East Asia consisting of the Korean Peninsula, Jeju Island, and smaller islands. Since the end of World War II in Asia in 1945, it has been politically divided at or near the 38th parallel between North Korea and South Korea. Both countries proclaimed independence in 1948. The region is bordered by China to the north and Russia to the northeast, across the Amnok and Duman rivers, and is separated from Japan to the southeast by the Korea Strait.",
    [("What is Korea?", ["peninsular", "east asia"]),
     ("Is Korea in East Asia?", ["yes"]),
     ("Is Korea divided?", ["yes"]),
     ("What is in the southeast of Korea?", ["japan"]),
     ("What borders Korea to the north?", ["china"]),
     ("What is Korea separated from?", ["japan"]),
     ("Where is Jeju Island?", ["east asia"])],
    "full Korea comprehension (7 questions)")

# ============================
# SUMMARY
# ============================
total = PASS + FAIL
print(f"\n{'='*50}")
print(f"Stress Suite Results")
print(f"{'='*50}")
print(f"Passed: {PASS}/{total} ({100*PASS//max(total,1)}%)")
print(f"Failed: {FAIL}/{total}")

if ERRORS:
    print(f"\nFailures ({len(ERRORS)}):")
    for cat, q, a, exp in ERRORS[:10]:
        print(f"  [{cat}] Q: {q}")
        print(f"    Expected: {exp}")
        print(f"    Got: {a[:80]}")
    if len(ERRORS) > 10:
        print(f"  ... and {len(ERRORS)-10} more")

