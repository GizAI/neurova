#!/usr/bin/env python3
"""
Neurova V40 CLI — Language Acquisition Engine (Clean v12).

Multi-sentence learning, coreference, self-learning from feedback.
Up/Down arrows for history. Type 'exit' to quit.
Prefix with 'learn: ' to teach the system from a wrong answer.
"""

import os, sys, readline, atexit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurova.engine import NeurovaEngine

def main():
    histfile = os.path.join(os.path.expanduser("~"), ".neurova_v40_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, histfile)

    print("=" * 58)
    print("  Neurova V12 — Language Acquisition Engine")
    print("  No hardcoded grammar rules. Learns from interaction.")
    print("  Up/Down arrows. 'exit' to quit. 'learn: <fact>' to teach.")
    print("=" * 58)
    print()

    engine = NeurovaEngine()
    print("[System] Ready.\n")

    while True:
        try:
            inp = input(">>> ").strip()
        except EOFError:
            break
        if not inp:
            continue
        if inp.lower() in ("exit", "quit"):
            break
        if inp.lower() == "sleep":
            report = engine.brain.sleep_cycle()
            print(f"[Sleep] {report}\n")
            continue
        if inp.lower() == "status":
            s = engine.brain.get_status()
            print(f"[Status] {s}\n")
            continue
        if inp.lower() == "reset":
            engine.reset()
            print("[System] Reset complete.\n")
            continue

        # Feedback mode: user teaches the system
        if inp.lower().startswith("learn: "):
            correction = inp[7:].strip()
            response = engine.brain.feedback("(training)", correction)
            print(f"[Learn] {response}\n")
            continue

        response = engine.hear(inp)
        print(f"[V40] {response}\n")


if __name__ == "__main__":
    main()
