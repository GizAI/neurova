#!/usr/bin/env python3
"""
Neurova V40 CLI — Language Acquisition Engine (Clean).
Zero hardcoded grammar rules. Learns from conversation.
Up/Down arrows for history. Type 'exit' to quit.

Commands:
  model       — show world model (entities & relations)
  episodes    — show recent episodes
  status      — show counts
  sleep       — consolidate memory
  reset       — clear everything
  learn: <X>  — teach the system a fact
  exit        — quit
"""

import os, sys, json, readline, atexit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from neurova.v40_language_acquisition import NeurovaEngine
except ImportError:
    from neurova.engine import NeurovaEngine


def show_model(engine):
    world = engine.brain.model
    if not world.entities:
        print("[Model] (empty)\n")
        return
    print(f"[Model] {len(world.entities)} entities\n")
    for name, entity in sorted(world.entities.items()):
        print(f"  ─ {name}")
        if entity.attributes:
            for k, v in entity.attributes.items():
                print(f"    {k}: {v}")
        if entity.location:
            print(f"    location: {entity.location}")
        if entity.time:
            print(f"    time: {entity.time}")
        if entity.properties:
            print(f"    properties: {', '.join(sorted(entity.properties)[:5])}")
        if entity.relations:
            for rtype, entries in entity.relations.items():
                for tgt, props in entries:
                    extra = ""
                    if props:
                        extra = " (" + ", ".join(f"{k}={v}" for k, v in props.items()) + ")"
                    print(f"    {rtype}: {tgt}{extra}")
        print()


def show_episodes(engine):
    eps = engine.brain.epmem.episodes
    if not eps:
        print("[Episodes] (empty)\n")
        return
    print(f"[Episodes] {len(eps)} total (last 20):\n")
    for ep in eps[-20:]:
        chk = "✓" if ep.success else "✗"
        qmark = "?" if ep.is_question else " "
        inp = ep.input[:80]
        ans = ep.answer[:60] if ep.answer else "(stored)"
        print(f"  {chk}{qmark} {inp} → {ans}")
    print()


def main():
    histfile = os.path.join(os.path.expanduser("~"), ".neurova_v40_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, histfile)

    print("=" * 58)
    print("  Neurova V40 — Language Acquisition Engine")
    print("  No hardcoded grammar rules. Learns from interaction.")
    print("  Commands: model | episodes | status | sleep | reset | learn: <fact>")
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
        if inp.lower() == "model":
            show_model(engine)
            continue
        if inp.lower() == "episodes":
            show_episodes(engine)
            continue
        if inp.lower() == "sleep":
            report = engine.brain.sleep_cycle()
            print(f"[Sleep] {report}\n")
            continue
        if inp.lower() == "status":
            s = engine.get_status()
            print(f"[Status] {s}\n")
            continue
        if inp.lower() == "reset":
            engine.reset()
            print("[System] Reset complete.\n")
            continue
        if inp.lower().startswith("learn: "):
            correction = inp[7:].strip()
            response = engine.brain.feedback("(training)", correction)
            print(f"[Learn] {response}\n")
            continue

        response = engine.hear(inp)
        print(f"[V40] {response}\n")


if __name__ == "__main__":
    main()
