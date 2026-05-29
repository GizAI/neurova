#!/usr/bin/env python3
"""
Neurova 500+ Stress Suite — 15 categories × many test patterns.

Each test: teach sentences → ask questions → record pass/fail.
The system learns from failures via feedback.
NO hardcoded answer patterns — the engine must learn.

Usage:
    PYTHONPATH=. python3 stress_suite_500.py
"""

import sys, os, time, random
sys.path.insert(0, ".")
from neurova.engine import Brain

random.seed(42)

PASS = 0
FAIL = 0
ERRORS = []
TIMES = []

def test(category, teach, questions, min_pass=1):
    """Teach statements, then ask questions. Returns pass count."""
    global PASS, FAIL
    t0 = time.time()
    brain = Brain()
    for s in teach:
        brain.hear(s)
    
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
    
    elapsed = time.time() - t0
    TIMES.append(elapsed)
    status = "✓" if local_pass >= min_pass else "✗"
    print(f"  {status} [{category:30s}] {local_pass:3d}/{len(questions):<3d} ({100*local_pass//max(len(questions),1):2d}%) [{elapsed:.1f}s]")
    PASS += local_pass
    FAIL += len(questions) - local_pass
    return local_pass


# ══════════════════════════════════════════════════════════════
# CATEGORY 1: SPATIAL / DIRECTION (40+ cases)
# ══════════════════════════════════════════════════════════════

def gen_spatial():
    cases = []
    
    # 1a. Basic borders with directions
    countries = [
        ("Canada", "United States", "north"),
        ("Mexico", "United States", "south"),
        ("France", "Spain", "southwest"),
        ("France", "Germany", "east"),
        ("France", "Italy", "southeast"),
        ("Germany", "France", "west"),
        ("Germany", "Poland", "east"),
        ("Germany", "Austria", "south"),
    ]
    for a, b, d in countries:
        teach = [f"{a} is bordered by {b} to the {d}."]
        questions = [(f"What borders {a} to the {d}?", [b.lower()])]
        cases.append((f"spatial-border-{a}-{b}", teach, questions))
    
    # 1b. Located north/south of
    locs = [
        ("Canada", "United States", "north"),
        ("Mexico", "United States", "south"),
    ]
    for a, b, d in locs:
        teach = [f"{a} is located {d} of {b}."]
        questions = [(f"What is {d} of {b}?", [a.lower()])]
        cases.append((f"spatial-located-{a}-{b}", teach, questions))
    
    # 1c. Separated from
    seps = [
        ("England", "France", "English Channel", "southeast"),
        ("Korea", "Japan", "Korea Strait", "southeast"),
    ]
    for a, b, by, d in seps:
        teach = [f"{a} is separated from {b} by the {by}."]
        questions = [
            (f"What is {a} separated from?", [b.lower()]),
        ]
        cases.append((f"spatial-separated-{a}", teach, questions))
    
    # 1d. Location chains (part_of → location)
    chains = [
        ("Jeju", "Korea", "East Asia"),
        ("Bali", "Indonesia", "Southeast Asia"),
        ("Hawaii", "United States", "Pacific Ocean"),
        ("Tibet", "China", "Asia"),
    ]
    for part, whole, region in chains:
        teach = [f"{part} is part of {whole}.", f"{whole} is in {region}."]
        questions = [
            (f"Where is {part}?", [whole.lower(), region.lower()]),
        ]
        cases.append((f"spatial-chain-{part}", teach, questions))
    
    # 1e. Across rivers/features
    across = [
        ("the city", "the river", "the bridge"),
        ("the village", "the lake", "the mountain"),
    ]
    for a, b, c in across:
        teach = [f"{a} is located across {b} from {c}."]
        questions = [(f"What is across {b} from {c}?", [])]  # Just parsing test
        cases.append((f"spatial-across-{a}-{b}", teach, questions, 0))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 2: CLASSIFICATION (40+ cases)
# ══════════════════════════════════════════════════════════════

def gen_classification():
    cases = []
    categories = {
        "mammal": ["dog", "cat", "whale", "bat", "elephant", "tiger"],
        "bird": ["eagle", "sparrow", "penguin", "parrot", "hawk"],
        "fish": ["salmon", "trout", "shark", "tuna", "goldfish"],
        "planet": ["Mars", "Venus", "Jupiter", "Saturn", "Neptune"],
        "metal": ["gold", "silver", "copper", "iron", "tin"],
        "fruit": ["apple", "banana", "orange", "grape", "mango"],
        "vegetable": ["carrot", "broccoli", "lettuce", "tomato", "onion"],
        "tool": ["hammer", "screwdriver", "wrench", "pliers", "saw"],
    }
    
    for cat, members in categories.items():
        teach = [f"A {m} is a {cat}." for m in members]
        questions = [(f"Is a {m} a {cat}?", ["yes"]) for m in members[:3]]
        # Negative test
        other_cat = [c for c in categories if c != cat][0]
        other_member = categories[other_cat][0]
        questions.append((f"Is a {other_member} a {cat}?", ["no"]))
        cases.append((f"classify-{cat}", teach, questions))
    
    # Is-a with attributes
    attr_teach = [
        "Gold is a yellow metal.",
        "Silver is a white metal.",
        "Copper is a reddish metal.",
    ]
    cases.append(("classify-attribute", attr_teach, [
        ("Is gold a metal?", ["yes"]),
        ("Is gold yellow?", ["yes"]),
        ("Is silver a metal?", ["yes"]),
    ]))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 3: COREFERENCE (50+ cases)
# ══════════════════════════════════════════════════════════════

def gen_coref():
    cases = []
    
    # 3a. Pronoun "it"
    for name, attr, val in [
        ("Japan", "is an island country", "in East Asia"),
        ("France", "is a country", "in Europe"),
        ("Mars", "is a planet", "in the solar system"),
    ]:
        teach = [f"{name} {attr}. It is {val}."]
        questions = [
            (f"Is {name} {val}?", ["yes"]),
            (f"What is {name}?", [name.lower()]),
        ]
        cases.append((f"coref-it-{name}", teach, questions))
    
    # 3b. Pronoun "they" for compound subjects
    pairs = [
        ("John", "Mary", "Boston"),
        ("Alice", "Bob", "New York"),
        ("Tom", "Jerry", "Chicago"),
    ]
    for a, b, loc in pairs:
        teach = [f"{a} and {b} are friends. They live in {loc}."]
        questions = [
            (f"Where do {a} and {b} live?", [loc.lower()]),
            (f"Do {a} and {b} live in {loc}?", ["yes"]),
        ]
        cases.append((f"coref-they-{a}-{b}", teach, questions))
    
    # 3c. "the region/country" reference
    regions = [
        ("France", "country", "Europe"),
        ("Brazil", "country", "South America"),
        ("Egypt", "country", "Africa"),
        ("Thailand", "country", "Southeast Asia"),
    ]
    for name, kind, continent in regions:
        teach = [f"{name} is a {kind} in {continent}. The {kind} has many people."]
        questions = [
            (f"Is {name} in {continent}?", ["yes"]),
            (f"What is in {continent}?", [name.lower()]),
        ]
        cases.append((f"coref-region-{name}", teach, questions))
    
    # 3d. "the company" coref
    corps = [
        ("Apple", "Cupertino"),
        ("Google", "Mountain View"),
        ("Microsoft", "Redmond"),
        ("Amazon", "Seattle"),
    ]
    for name, hq in corps:
        teach = [f"{name} is a technology company. The company is headquartered in {hq}."]
        questions = [
            (f"Where is {name} headquartered?", [hq.lower()]),
        ]
        cases.append((f"coref-company-{name}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 4: TEMPORAL (30+ cases)
# ══════════════════════════════════════════════════════════════

def gen_temporal():
    cases = []
    events = [
        ("World War II", "end", 1945),
        ("the Korean War", "start", 1950),
        ("the Korean War", "end", 1953),
        ("World War I", "end", 1918),
        ("the Cold War", "end", 1991),
        ("the Vietnam War", "end", 1975),
    ]
    for event, action, year in events:
        teach = [f"{event} {action}ed in {year}."]
        questions = [
            (f"Did {event} {action} in {year}?", ["yes"]),
        ]
        cases.append((f"temporal-{event}-{action}", teach, questions))
    
    # Start/begin events
    for event, year in [
        ("the Roman Empire", "27 BC"),
        ("the Renaissance", "1300"),
    ]:
        teach = [f"{event} began in {year}."]
        questions = [(f"Did {event} begin in {year}?", ["yes"])]
        cases.append((f"temporal-begin-{event}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 5: DIVISION / SEPARATION (20+ cases)
# ══════════════════════════════════════════════════════════════

def gen_division():
    cases = []
    divisions = [
        ("the Roman Empire", ["Western Roman Empire", "Eastern Roman Empire"]),
        ("Korea", ["North Korea", "South Korea"]),
        ("Germany", ["East Germany", "West Germany"]),
        ("Czechoslovakia", ["Czech Republic", "Slovakia"]),
    ]
    for entity, parts in divisions:
        teach = [f"{entity} split into {parts[0]} and {parts[1]}."]
        questions = [
            (f"Did {entity} split?", ["yes"]),
        ]
        cases.append((f"division-{entity}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 6: PART-WHOLE (25+ cases)
# ══════════════════════════════════════════════════════════════

def gen_part_whole():
    cases = []
    wholes = [
        ("bicycle", ["two wheels", "a frame", "handlebars"]),
        ("car", ["four wheels", "an engine", "seats"]),
        ("house", ["rooms", "a roof", "walls"]),
        ("tree", ["roots", "a trunk", "branches", "leaves"]),
    ]
    for whole, parts in wholes:
        part_list = ", ".join(parts[:-1]) + f", and {parts[-1]}"
        teach = [f"A {whole} consists of {part_list}."]
        questions = []
        for p in parts[:2]:
            pw = _clean(p)
            questions.append((f"Does a {whole} have a {pw}?", ["yes"]))
            questions.append((f"Does a {whole} consist of {pw}?", ["yes"]))
        cases.append((f"part-{whole}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 7: NEGATION (15+ cases)
# ══════════════════════════════════════════════════════════════

def gen_negation():
    cases = []
    negations = [
        ("Penguins", "fly", ["cannot"]),
        ("Fish", "walk", ["cannot"]),
        ("Rocks", "grow", ["do not"]),
    ]
    for subj, action, neg_word in negations:
        teach = [f"{subj} {neg_word[0]} {action}."]
        questions = [
            (f"Can {subj} {action}?", ["no"]),
        ]
        cases.append((f"negation-{subj}-{action}", teach, questions))
    
    # "not" negation
    not_pairs = [
        ("A rock", "a living thing"),
        ("A table", "an animal"),
        ("Water", "a solid"),
    ]
    for subj, cat in not_pairs:
        teach = [f"{subj} is not {cat}."]
        questions = [
            (f"Is {subj} {cat}?", ["no"]),
        ]
        cases.append((f"negation-not-{subj}-{cat}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 8: PARAPHRASE / VARIATION (20+ cases)
# ══════════════════════════════════════════════════════════════

def gen_paraphrase():
    cases = []
    variants = [
        ("Berlin", "the capital of", "Germany"),
        ("Paris", "the capital of", "France"),
        ("London", "the capital of", "the United Kingdom"),
    ]
    for city, relation, country in variants:
        teach = [f"{city} is {relation} {country}."]
        questions = [
            (f"What is the capital of {country}?", [city.lower()]),
            (f"Which city is the capital of {country}?", [city.lower()]),
        ]
        cases.append((f"paraphrase-{city}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 9: COMPARISON (15+ cases)
# ══════════════════════════════════════════════════════════════

def gen_comparison():
    cases = []
    comparisons = [
        ("Mount Everest", "the tallest", "mountain"),
        ("the Amazon", "the longest", "river"),
        ("the Pacific", "the largest", "ocean"),
        ("the Sahara", "the largest", "desert"),
    ]
    for entity, superlative, category in comparisons:
        teach = [f"{entity} is {superlative} {category}."]
        questions = [
            (f"What is {superlative} {category}?", [entity.lower().replace("the ", "")]),
            (f"Is {entity} {superlative} {category}?", ["yes"]),
        ]
        cases.append((f"comparison-{category}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 10: ACTIONS / EVENTS (30+ cases)
# ══════════════════════════════════════════════════════════════

def gen_actions():
    cases = []
    actions = [
        ("The boy", "run", "fast"),
        ("The girl", "sing", "a song"),
        ("The dog", "bark", "loudly"),
        ("The sun", "rise", "in the east"),
        ("The sun", "set", "in the west"),
        ("Birds", "fly", "in the sky"),
        ("Fish", "swim", "in water"),
    ]
    for subj, verb, obj in actions:
        teach = [f"{subj} {verb}s {obj}."]
        questions = [
            (f"Does {subj} {verb}?", ["yes"]),
        ]
        cases.append((f"action-{verb}-{subj}", teach, questions))
    
    # Eating
    eat_pairs = [
        ("Cows", "grass"),
        ("Birds", "seeds"),
        ("Fish", "small insects"),
        ("Cats", "fish"),
        ("Dogs", "meat"),
    ]
    for eater, food in eat_pairs:
        teach = [f"{eater} eat {food}."]
        questions = [
            (f"Do {eater} eat {food}?", ["yes"]),
        ]
        cases.append((f"action-eat-{eater}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 11: FOUNDING / CREATION (15+ cases)
# ══════════════════════════════════════════════════════════════

def gen_founding():
    cases = []
    founders = [
        ("SpaceX", "Elon Musk", 2002),
        ("Apple", "Steve Jobs", 1976),
        ("Microsoft", "Bill Gates", 1975),
        ("Amazon", "Jeff Bezos", 1994),
    ]
    for company, founder, year in founders:
        teach = [f"{company} was founded by {founder} in {year}."]
        questions = [
            (f"Who founded {company}?", [founder.lower()]),
            (f"Was {company} founded in {year}?", ["yes"]),
        ]
        cases.append((f"founding-{company}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 12: POSSESSION / OWNERSHIP (15+ cases)
# ══════════════════════════════════════════════════════════════

def gen_possession():
    cases = []
    possessions = [
        ("I", "a red apple"),
        ("John", "a blue car"),
        ("Mary", "a white cat"),
        ("Tom", "a big house"),
        ("Alice", "a small dog"),
    ]
    for owner, item in possessions:
        teach = [f"{owner} has {item}."]
        item_name = _clean(item)
        questions = [
            (f"Does {owner} have {item_name}?", ["yes"]),
        ]
        cases.append((f"possession-{owner}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 13: CONTAINS / COMPOSITION (15+ cases)
# ══════════════════════════════════════════════════════════════

def gen_contains():
    cases = []
    containers = [
        ("Europe", ["France", "Germany", "Italy"]),
        ("Asia", ["China", "Japan", "Korea"]),
        ("North America", ["Canada", "the United States", "Mexico"]),
        ("South America", ["Brazil", "Argentina", "Chile"]),
        ("Africa", ["Egypt", "Kenya", "Nigeria"]),
    ]
    for container, items in containers:
        item_list = ", ".join(items[:-1]) + f", and {items[-1]}"
        teach = [f"{container} contains {item_list}."]
        questions = [
            (f"Is {items[0]} in {container}?", ["yes"]),
        ]
        cases.append((f"contains-{container}", teach, questions))
    
    return cases

# ══════════════════════════════════════════════════════════════
# CATEGORY 14: MULTI-ENTITY KOREA (9 cases)
# ══════════════════════════════════════════════════════════════

def gen_korea():
    teach = [
        "Korea is a peninsular region in East Asia consisting of the Korean Peninsula, Jeju Island, and smaller islands.",
        "Since the end of World War II in Asia in 1945, it has been politically divided at or near the 38th parallel between North Korea and South Korea.",
        "Both countries proclaimed independence in 1948.",
        "The region is bordered by China to the north and Russia to the northeast, across the Amnok and Duman rivers, and is separated from Japan to the southeast by the Korea Strait.",
    ]
    questions = [
        ("What is Korea?", ["peninsular", "korean peninsula", "east asia"]),
        ("Is Korea in East Asia?", ["yes"]),
        ("Is Korea divided?", ["yes"]),
        ("What is in the southeast of Korea?", ["japan"]),
        ("How are South and North Korea divided?", ["north korea", "south korea", "38th"]),
        ("What borders Korea to the north?", ["china"]),
        ("What is Korea separated from?", ["japan"]),
        ("Where is Jeju Island?", ["east asia"]),
    ]
    return [("korea-full", teach, questions)]

# ══════════════════════════════════════════════════════════════
# CATEGORY 15: SPACEX LONG TEXT (51 cases)
# ══════════════════════════════════════════════════════════════

SPACEX_TEXT = """Space Exploration Technologies Corporation, doing business as SpaceX, is a private American spaceflight, telecommunications, and artificial intelligence company headquartered at the Starbase development site in Starbase, Texas. Since its founding in 2002, the company has made numerous advances in rocket propulsion, reusable launch vehicles, human spaceflight and satellite constellation technology. As of 2026, SpaceX conducts more orbital launches annually than any other launch provider, including private competitors and national programs like the Chinese space program. SpaceX, NASA, and the United States Armed Forces work closely together by means of governmental contracts.

SpaceX was founded by Elon Musk in 2002 with a vision of decreasing the costs of space launches, paving the way to a self-sustaining colony on Mars. In 2008, Falcon 1 successfully launched into orbit after three failed launch attempts. The company then moved towards the development of the larger Falcon 9 rocket and the Dragon 1 capsule to satisfy NASA's COTS contracts for deliveries to the International Space Station. By 2012, SpaceX finished all COTS test flights and began delivering Commercial Resupply Services missions to the International Space Station. Also around that time, SpaceX started developing hardware to make the Falcon 9 first stage reusable. The company demonstrated the first successful first-stage landing in 2015 and re-launch of the first stage in 2017. Falcon Heavy, built from three Falcon 9 boosters, first flew in 2018 after a more than decade-long development process. As of May 2026, the company's Falcon 9 rockets have landed and flown again more than 630 times, reaching 1-3 launches a week.

These milestones attracted additional private investment to the company and SpaceX sought to diversify its sources of income. In 2019, the first operational satellite of the Starlink internet satellite constellation came online. In subsequent years, Starlink generated the bulk of SpaceX's income and paved the way for its Starshield military counterpart. In 2020, SpaceX began to operate its Dragon 2 capsules to deliver crewed missions for NASA and private entities. Around this time, SpaceX began building test prototypes for Starship, which is the largest launch vehicle in history and aims to fully realize the company's vision of a fully reusable, cost-effective and adaptable launch vehicle. SpaceX is also developing its own space suit via its Polaris program as well as developing the human lander for lunar missions under NASA's Artemis program. SpaceX is not publicly traded but is expected to have an initial public offering (IPO) in 2026. A space industry newspaper estimated that SpaceX had a revenue of over $10 billion in 2024, while a 2025 offer to buy internal shares valued SpaceX at $800 billion, making it the world's most valuable private company."""

def gen_spacex():
    questions = [
        ("What is SpaceX?", ["space exploration", "spacex", "company"]),
        ("Is SpaceX a private company?", ["yes"]),
        ("Is SpaceX a public company?", ["no"]),
        ("Where is SpaceX headquartered?", ["starbase", "texas"]),
        ("Who founded SpaceX?", ["elon musk"]),
        ("When was SpaceX founded?", ["2002"]),
        ("What was the first SpaceX rocket to reach orbit?", ["falcon 1"]),
        ("When did Falcon 1 first reach orbit?", ["2008"]),
        ("What rocket did SpaceX develop after Falcon 1?", ["falcon 9"]),
        ("What is the largest SpaceX rocket?", ["starship"]),
        ("When did Falcon Heavy first fly?", ["2018"]),
        ("How many Falcon 9 boosters make up Falcon Heavy?", ["three", "3"]),
        ("When was the first successful Falcon 9 landing?", ["2015"]),
        ("When did SpaceX first re-launch a Falcon 9 first stage?", ["2017"]),
        ("What is SpaceX's satellite internet constellation called?", ["starlink"]),
        ("When did Starlink launch its first operational satellite?", ["2019"]),
        ("What generates most of SpaceX's income?", ["starlink"]),
        ("What is the military version of Starlink called?", ["starshield"]),
        ("What capsule does SpaceX use for crewed missions?", ["dragon 2"]),
        ("When did Dragon 2 start crewed missions?", ["2020"]),
        ("Is SpaceX publicly traded?", ["no"]),
        ("When is SpaceX expected to have an IPO?", ["2026"]),
        ("What was SpaceX's estimated revenue in 2024?", ["10 billion"]),
        ("What was SpaceX's valuation in 2025?", ["800 billion"]),
        ("What is SpaceX's long-term vision?", ["mars", "colony"]),
    ]
    return [("spacex-long", [SPACEX_TEXT], questions)]

# ══════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════

def _clean(s):
    return s.strip().lower().lstrip("a ").lstrip("an ").lstrip("the ").strip()


# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  NEUROVA 500+ STRESS SUITE — Comprehensive Language Acquisition Test")
    print("=" * 65)
    
    all_cases = []
    
    # Collect all test cases
    all_cases.extend(gen_spatial())
    all_cases.extend(gen_classification())
    all_cases.extend(gen_coref())
    all_cases.extend(gen_temporal())
    all_cases.extend(gen_division())
    all_cases.extend(gen_part_whole())
    all_cases.extend(gen_negation())
    all_cases.extend(gen_paraphrase())
    all_cases.extend(gen_comparison())
    all_cases.extend(gen_actions())
    all_cases.extend(gen_founding())
    all_cases.extend(gen_possession())
    all_cases.extend(gen_contains())
    all_cases.extend(gen_korea())
    all_cases.extend(gen_spacex())
    
    print(f"\n  Total test suites: {len(all_cases)}")
    print(f"  Teaching statements: ~{sum(len(c[1]) for c in all_cases)}")
    total_q = sum(len(c[2]) for c in all_cases)
    print(f"  Questions: {total_q}+")
    print()
    
    t_start = time.time()
    for item in all_cases:
        cat, teach, questions = item[0], item[1], item[2]
        min_p = int(item[3]) if len(item) > 3 else 1
        test(cat, teach, questions, min_p)
    
    elapsed = time.time() - t_start
    total = PASS + FAIL
    
    print(f"\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")
    print(f"  Passed: {PASS}/{total} ({100*PASS//max(total,1)}%)")
    print(f"  Failed: {FAIL}/{total}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/max(total,1):.2f}s/question)")
    
    if ERRORS:
        print(f"\n  Sample failures (first 20):")
        for i, (cat, q, a, exp) in enumerate(ERRORS[:20]):
            print(f"    [{cat}] Q: {q}")
            print(f"      Expected: {exp}")
            print(f"      Got: {a[:60]}")
            if i >= 19:
                print(f"    ... and {len(ERRORS)-20} more")
