#!/usr/bin/env python3
"""SpaceX comprehension test — complex long-form text understanding."""

import sys; sys.path.insert(0, ".")
from neurova.engine import Brain

SPACEX_TEXT = """Space Exploration Technologies Corporation, doing business as SpaceX, is a private American spaceflight, telecommunications, and artificial intelligence company headquartered at the Starbase development site in Starbase, Texas. Since its founding in 2002, the company has made numerous advances in rocket propulsion, reusable launch vehicles, human spaceflight and satellite constellation technology. As of 2026, SpaceX conducts more orbital launches annually than any other launch provider, including private competitors and national programs like the Chinese space program. SpaceX, NASA, and the United States Armed Forces work closely together by means of governmental contracts.

SpaceX was founded by Elon Musk in 2002 with a vision of decreasing the costs of space launches, paving the way to a self-sustaining colony on Mars. In 2008, Falcon 1 successfully launched into orbit after three failed launch attempts. The company then moved towards the development of the larger Falcon 9 rocket and the Dragon 1 capsule to satisfy NASA's COTS contracts for deliveries to the International Space Station. By 2012, SpaceX finished all COTS test flights and began delivering Commercial Resupply Services missions to the International Space Station. Also around that time, SpaceX started developing hardware to make the Falcon 9 first stage reusable. The company demonstrated the first successful first-stage landing in 2015 and re-launch of the first stage in 2017. Falcon Heavy, built from three Falcon 9 boosters, first flew in 2018 after a more than decade-long development process. As of May 2026, the company's Falcon 9 rockets have landed and flown again more than 630 times, reaching 1-3 launches a week.

These milestones attracted additional private investment to the company and SpaceX sought to diversify its sources of income. In 2019, the first operational satellite of the Starlink internet satellite constellation came online. In subsequent years, Starlink generated the bulk of SpaceX's income and paved the way for its Starshield military counterpart. In 2020, SpaceX began to operate its Dragon 2 capsules to deliver crewed missions for NASA and private entities. Around this time, SpaceX began building test prototypes for Starship, which is the largest launch vehicle in history and aims to fully realize the company's vision of a fully reusable, cost-effective and adaptable launch vehicle. SpaceX is also developing its own space suit via its Polaris program as well as developing the human lander for lunar missions under NASA's Artemis program. SpaceX is not publicly traded but is expected to have an initial public offering (IPO) in 2026. A space industry newspaper estimated that SpaceX had a revenue of over $10 billion in 2024, while a 2025 offer to buy internal shares valued SpaceX at $800 billion, making it the world's most valuable private company."""

# Questions organized by category
questions = [
    # === BASIC FACTS ===
    ("What is SpaceX?", ["space exploration", "spacex", "company", "spaceflight"]),
    ("What does SpaceX stand for?", ["space exploration technologies"]),
    ("Is SpaceX a private company?", ["yes"]),
    ("Is SpaceX a public company?", ["no"]),
    ("Where is SpaceX headquartered?", ["starbase", "texas"]),
    
    # === FOUNDING & HISTORY ===
    ("Who founded SpaceX?", ["elon musk"]),
    ("When was SpaceX founded?", ["2002"]),
    ("Did SpaceX start in 2002?", ["yes"]),
    
    # === LAUNCH VEHICLES ===
    ("What was the first SpaceX rocket to reach orbit?", ["falcon 1"]),
    ("When did Falcon 1 first reach orbit?", ["2008"]),
    ("How many times did Falcon 1 fail before succeeding?", ["three", "3"]),
    ("What rocket did SpaceX develop after Falcon 1?", ["falcon 9"]),
    ("What is the largest SpaceX rocket?", ["starship"]),
    ("What is Starship?", ["largest launch vehicle", "fully reusable"]),
    ("When did Falcon Heavy first fly?", ["2018"]),
    ("How many Falcon 9 boosters make up Falcon Heavy?", ["three", "3"]),
    
    # === REUSABILITY ===
    ("When was the first successful Falcon 9 landing?", ["2015"]),
    ("When did SpaceX first re-launch a Falcon 9 first stage?", ["2017"]),
    ("How many times have Falcon 9 rockets landed and flown again?", ["630"]),
    ("How often does SpaceX launch as of 2026?", ["1-3", "launches a week"]),
    
    # === MISSIONS & CAPSULES ===
    ("What capsule did SpaceX develop for NASA cargo deliveries?", ["dragon 1"]),
    ("What capsule does SpaceX use for crewed missions?", ["dragon 2"]),
    ("When did Dragon 2 start crewed missions?", ["2020"]),
    ("What was the purpose of Dragon 1?", ["cots", "iss", "international space station"]),
    ("When did SpaceX finish COTS test flights?", ["2012"]),
    
    # === STARLINK ===
    ("What is SpaceX's satellite internet constellation called?", ["starlink"]),
    ("When did Starlink launch its first operational satellite?", ["2019"]),
    ("What generates most of SpaceX's income?", ["starlink"]),
    ("What is the military version of Starlink called?", ["starshield"]),
    
    # === NASA & GOVERNMENT ===
    ("Who does SpaceX work closely with?", ["nasa", "united states armed forces"]),
    ("What NASA program is SpaceX developing a lunar lander for?", ["artemis"]),
    ("What did SpaceX develop for NASA's COTS program?", ["falcon 9", "dragon 1"]),
    
    # === CORPORATE ===
    ("Is SpaceX publicly traded?", ["no"]),
    ("When is SpaceX expected to have an IPO?", ["2026"]),
    ("What was SpaceX's estimated revenue in 2024?", ["10 billion"]),
    ("What was SpaceX's valuation in 2025?", ["800 billion"]),
    ("What makes SpaceX the world's most valuable private company?", ["800 billion"]),
    ("Who valued SpaceX at $800 billion?", ["2025 offer", "internal shares"]),
    
    # === SPACEX'S GOALS ===
    ("What is SpaceX's long-term vision?", ["mars", "colony", "self-sustaining"]),
    ("What does SpaceX want to do on Mars?", ["colony", "self-sustaining"]),
    
    # === COMPARISON ===
    ("Does SpaceX launch more rockets than any other company?", ["yes"]),
    ("Is SpaceX more active than the Chinese space program?", ["yes"]),
    
    # === COREFERENCE TESTS ===
    ("Elon Musk founded a company. Where is it headquartered?", ["starbase", "texas"]),
    ("SpaceX develops rockets. When was it founded?", ["2002"]),
    ("The company builds launch vehicles. What is its largest rocket?", ["starship"]),
    
    # === NEGATION ===
    ("Is SpaceX a government agency?", ["no"]),
    ("Is SpaceX based in Florida?", ["no"]),
    ("Did Falcon 1 succeed on its first try?", ["no"]),
    
    # === MULTI-SENTENCE INFERENCE ===
    ("SpaceX was founded in 2002. Falcon 1 reached orbit in 2008. How many years passed?", ["six", "6"]),
    ("SpaceX builds rockets. One rocket is made from three boosters. Which rocket is that?", ["falcon heavy"]),
    ("What technology did SpaceX develop after finishing COTS flights?", ["reusable", "falcon 9 first stage"]),
]

print("=" * 60)
print("SpaceX Comprehension Test — Long-form Text Understanding")
print("=" * 60)

brain = Brain()
print("\n[Ingesting SpaceX article...]")
r = brain.hear(SPACEX_TEXT)
print(f"  Result: {r}\n")

pass_count = 0
fail_count = 0
results = []

for q, expected_kw in questions:
    a = brain.hear(q)
    al = a.lower()
    ok = any(kw.lower() in al for kw in expected_kw)
    if ok:
        pass_count += 1
    else:
        fail_count += 1
    results.append((ok, q, a, expected_kw))

print(f"\n{'='*60}")
print(f"Results: {pass_count}/{len(questions)} passed ({100*pass_count//len(questions)}%)")
print(f"{'='*60}\n")

# Show by category
cats = {
    "Basic Facts": (0, 5),
    "Founding & History": (5, 8),
    "Launch Vehicles": (8, 17),
    "Reusability": (17, 21),
    "Missions & Capsules": (21, 26),
    "Starlink": (26, 30),
    "NASA & Government": (30, 33),
    "Corporate": (33, 38),
    "Goals": (38, 40),
    "Comparison": (40, 42),
    "Coreference": (42, 45),
    "Negation": (45, 48),
    "Inference": (48, 51),
}

for cat, (start, end) in cats.items():
    subset = results[start:end]
    p = sum(1 for ok,_,_,_ in subset if ok)
    print(f"  [{cat}] {p}/{len(subset)} ({100*p//max(len(subset),1)}%)")

print(f"\n--- FAILURES ({fail_count}) ---")
for ok, q, a, exp in results:
    if not ok:
        print(f"  ✗ Q: {q}")
        print(f"    Expected: {exp}")
        print(f"    Got: {a[:100]}")
        print()
