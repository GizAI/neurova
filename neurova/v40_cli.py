#!/usr/bin/env python3
"""
Neurova V40 CLI — Construction Learner Architecture (v9).

Multi-sentence learning, coreference, GPU embeddings.
Up/Down arrows for history.
"""

import os
import sys
import readline
import atexit

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
    print("  Neurova V40 — Cognitive Engine (v9 Construction Learner)")
    print("  Multi-sentence / Coreference / Self-Learning")
    print("  Up/Down arrows for history. 'exit' to quit.")
    print("=" * 58)
    print()

    engine = NeurovaEngine()
    print("[System] Ready.")

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
            report = engine.sleep_cycle()
            print(f"[Sleep] Consolidation: {report}\n")
            continue
        if inp.lower() == "status":
            s = engine.get_status()
            print(f"[Status] {s}\n")
            continue
        if inp.lower() == "reset":
            engine.reset()
            print("[System] Reset complete.\n")
            continue

        response = engine.hear(inp)
        print(f"[V40] {response}\n")


if __name__ == "__main__":
    main()
