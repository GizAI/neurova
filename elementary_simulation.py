#!/usr/bin/env python3
"""
Neurova Elementary School Curriculum Simulation.

The system learns like a child — through simple sentences, questions,
and corrections. The teacher (this script) guides the learning process
through the Korean/English elementary school curriculum stages.

Usage:
    PYTHONPATH=. python3 elementary_simulation.py [--interactive]
"""

import sys, os, time
sys.path.insert(0, ".")
from neurova.engine import Brain

# ── Curriculum stages ──

STAGE_1 = {
    "name": "Stage 1: Basic Identity & Classification (Grade 1-2)",
    "lessons": [
        {
            "teach": [
                "I am a student.",
                "You are a teacher.",
                "This is a book.",
                "That is a pencil.",
            ],
            "test": [
                ("Am I a student?", "yes"),
                ("Are you a teacher?", "yes"),
                ("What is this?", "book"),
            ],
            "corrections": {
                "Am I a student?": "I am a student.",
                "What is this?": "This is a book.",
            }
        },
        # Lesson 2: Animals
        {
            "teach": [
                "A dog is an animal.",
                "A cat is an animal.",
                "A bird is an animal.",
                "A fish is an animal.",
            ],
            "test": [
                ("Is a dog an animal?", "yes"),
                ("Is a cat an animal?", "yes"),
                ("Is a fish an animal?", "yes"),
                ("Is a rock an animal?", "no"),
            ]
        },
    ]
}

STAGE_2 = {
    "name": "Stage 2: Location & Existence (Grade 2-3)",
    "lessons": [
        {
            "teach": [
                "The book is on the table.",
                "The pencil is in the box.",
                "The cat is under the chair.",
                "The bird is in the tree.",
            ],
            "test": [
                ("Where is the book?", "table"),
                ("Is the pencil in the box?", "yes"),
                ("Is the cat on the table?", "no"),
            ]
        },
        {
            "teach": [
                "Seoul is in South Korea.",
                "Busan is in South Korea.",
                "South Korea is in Asia.",
                "Japan is in Asia.",
            ],
            "test": [
                ("Where is Seoul?", "south korea"),
                ("Is Seoul in Asia?", "yes"),
                ("Is Japan in Asia?", "yes"),
                ("Is Busan in Japan?", "no"),
            ]
        },
    ]
}

STAGE_3 = {
    "name": "Stage 3: Possession & Attributes (Grade 3-4)",
    "lessons": [
        {
            "teach": [
                "I have a red apple.",
                "The apple is sweet.",
                "The apple is round.",
                "The sky is blue.",
                "Grass is green.",
            ],
            "test": [
                ("Is the apple red?", "yes"),
                ("Is the apple sweet?", "yes"),
                ("Is the sky blue?", "yes"),
                ("Is the sky green?", "no"),
            ],
            "corrections": {
                "Is the apple red?": "The apple is red.",
                "Is the apple sweet?": "The apple is sweet.",
                "Is the sky blue?": "The sky is blue.",
            }
        },
        {
            "teach": [
                "Cows eat grass.",
                "Birds eat seeds.",
                "Fish eat small insects.",
            ],
            "test": [
                ("Do cows eat grass?", "yes"),
                ("Do birds eat grass?", "no"),
                ("What do cows eat?", "grass"),
            ],
            "corrections": {
                "Do cows eat grass?": "Cows eat grass.",
                "Do birds eat grass?": "Birds eat seeds.",
                "What do cows eat?": "Cows eat grass.",
            }
        },
    ]
}

STAGE_4 = {
    "name": "Stage 4: Action & Events (Grade 4-5)",
    "lessons": [
        {
            "teach": [
                "The boy runs fast.",
                "The girl sings a song.",
                "The dog barks loudly.",
                "The sun rises in the east.",
                "The sun sets in the west.",
            ],
            "test": [
                ("Does the boy run fast?", "yes"),
                ("What does the girl do?", "sings"),
                ("Where does the sun rise?", "east"),
            ]
        },
        {
            "teach": [
                "World War II ended in 1945.",
                "The Korean War started in 1950.",
                "The Korean War ended in 1953.",
            ],
            "test": [
                ("Did World War II end in 1945?", "yes"),
                ("Did the Korean War start in 1950?", "yes"),
                ("Did the Korean War end in 1953?", "yes"),
            ]
        },
    ]
}

STAGE_5 = {
    "name": "Stage 5: Relations & Comparisons (Grade 5-6)",
    "lessons": [
        {
            "teach": [
                "Mount Everest is the tallest mountain.",
                "The Amazon is the longest river.",
                "The Pacific is the largest ocean.",
                "Cheetahs are faster than turtles.",
                "Elephants are bigger than mice.",
            ],
            "test": [
                ("What is the tallest mountain?", "everest"),
                ("Is the Pacific the largest ocean?", "yes"),
                ("Are cheetahs faster than turtles?", "yes"),
                ("Are mice bigger than elephants?", "no"),
            ]
        },
        {
            "teach": [
                "France is in Europe.",
                "Germany is in Europe.",
                "Germany is bordered by France to the west.",
                "France is bordered by Germany to the east.",
                "Korea is separated from Japan by the sea.",
            ],
            "test": [
                ("Is France in Europe?", "yes"),
                ("What borders France to the east?", "germany"),
                ("Is Korea separated from Japan?", "yes"),
            ]
        },
    ]
}

ALL_STAGES = [STAGE_1, STAGE_2, STAGE_3, STAGE_4, STAGE_5]


def run_simulation(brain, stages=ALL_STAGES, interactive=False):
    """Run the elementary school simulation."""
    total_questions = 0
    total_correct = 0
    
    for stage in stages:
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"{'='*60}")
        
        for lesson_idx, lesson in enumerate(stage["lessons"]):
            print(f"\n  --- Lesson {lesson_idx + 1} ---")
            
            # Teach
            for sentence in lesson["teach"]:
                response = brain.hear(sentence)
                if interactive:
                    print(f'  Teach: "{sentence}"')
                    print(f'    → {response}')
            
            # Test
            for question, expected_keyword in lesson["test"]:
                answer = brain.hear(question)
                is_correct = expected_keyword.lower() in answer.lower()
                total_questions += 1
                
                if is_correct:
                    total_correct += 1
                    status = "✓"
                    if interactive:
                        print(f'  {status} Q: {question}')
                        print(f'    Got: {answer}')
                else:
                    status = "✗"
                    print(f'  {status} Q: {question}')
                    print(f'    Got: {answer}')
                    print(f'    Expected: {expected_keyword}')
                    if interactive:
                        correction = input("    Correction (or Enter to skip): ").strip()
                        if correction:
                            response = brain.feedback(question, correction)
                            print(f'    {response}')
                    else:
                        # Auto-learn from failures by feeding the expected answer
                        correction_sentence = lesson.get("corrections", {}).get(question, "")
                        if correction_sentence:
                            response = brain.feedback(question, correction_sentence)
                            print(f'    Auto-learn: {response}')
                        else:
                            # Try to infer a correction from the expected keyword
                            # Just re-store the fact
                            if question.startswith("Is ") or question.startswith("Are ") or question.startswith("Am "):
                                rest = question[3:].strip().rstrip("?")
                                # "Is X Y?" → "X is Y"
                                correction_sentence = f"{rest}"
                                response = brain.feedback(question, correction_sentence)
                                print(f'    Auto-learn: {response}')
                            elif question.startswith("Do ") or question.startswith("Does "):
                                rest = question[3:].strip().rstrip("?")
                                # "cows eat grass?" -> "cows eat grass."
                                correction_sentence = f"{rest}."
                                response = brain.feedback(question, correction_sentence)
                                print(f'    Auto-learn: {response}')
                            elif question.startswith("Did "):
                                rest = question[3:].strip().rstrip("?")
                                # "world war ii end in 1945?" -> needs proper tense
                                parts = rest.split(None, 2)
                                if len(parts) >= 2:
                                    subj = parts[0]
                                    verb = parts[1]
                                    rest_of = " ".join(parts[2:]) if len(parts) > 2 else ""
                                    # Convert past-tense question to past-tense statement
                                    correction_sentence = f"{subj} {verb}ed {rest_of}." if not rest_of.endswith("ed") else f"{subj} {verb} {rest_of}."
                                else:
                                    correction_sentence = f"{rest}."
                                response = brain.feedback(question, correction_sentence)
                                print(f'    Auto-learn: {response}')
                            elif question.startswith("What "):
                                # "What is X?" → expected_keyword tells us X's description
                                if "is" in question:
                                    parts = question.lower().replace("what is ", "").replace("what are ", "").rstrip("?").strip()
                                    correction_sentence = f"{parts} is {expected_keyword}"
                                    response = brain.feedback(question, correction_sentence)
                                    print(f'    Auto-learn: {response}')
                            else:
                                # Generic: just tell the system
                                response = brain.feedback(question, expected_keyword)
                                print(f'    Auto-learn: {response}')
        
        # Show stage progress
        stage_questions = sum(len(l["test"]) for l in stage["lessons"])
        stage_correct = sum(
            1 for l in stage["lessons"]
            for q, e in l["test"]
            if e.lower() in brain.hear(q).lower()
        )
        print(f"\n  Stage progress: {stage_correct}/{stage_questions}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"  SIMULATION COMPLETE")
    print(f"  Total: {total_correct}/{total_questions} ({100*total_correct//max(total_questions,1)}%)")
    print(f"  Entities learned: {len(brain.model.entities)}")
    print(f"  Constructions: {len(brain.cmem.constructions)}")
    print(f"  Episodes: {len(brain.epmem.episodes)}")
    print(f"{'='*60}")
    return total_correct, total_questions


if __name__ == "__main__":
    interactive = "--interactive" in sys.argv
    brain = Brain()
    run_simulation(brain, interactive=interactive)
