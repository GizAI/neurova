#!/usr/bin/env python3
"""CLI for Neurova TTT-Qwen Engine (Qwen3.5-4B + TTT)."""
from __future__ import annotations
import os, sys, readline, atexit
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurova.ttt_qwen_engine import NeurovaTTTEngine, VLLM_URL, VLLM_MODEL

def main():
    hist = os.path.expanduser("~/.neurova_ttt_qwen_history")
    try: readline.read_history_file(hist)
    except FileNotFoundError: pass
    readline.set_history_length(2000)
    atexit.register(readline.write_history_file, hist)

    print("=" * 68)
    print(f"  Neurova TTT-Qwen — {VLLM_MODEL} + Test-Time Training")
    print(f"  Endpoint: {VLLM_URL}")
    print("  Commands:")
    print("    correct: <Q> => <A>   Learn a correction (episodic)")
    print("    learn: <Q> => <A>     Alias for correct")
    print("    lora                  Train ephemeral LoRA adapter")
    print("    verify: <Q>           Self-verify last answer")
    print("    status                Show engine status")
    print("    distill               Export training data")
    print("=" * 68)

    engine = NeurovaTTTEngine()
    print(f"\n[System] Ready.\n")

    while True:
        try:
            inp = input(">>> ").strip()
        except EOFError:
            print(); break
        if not inp: continue
        if inp.lower() in ("exit", "quit", ":q"): break
        try:
            ans = engine.hear(inp)
            print(f"[TTT-Qwen] {ans}\n")
        except KeyboardInterrupt:
            print("\n[Interrupted]\n")
        except Exception as exc:
            print(f"[Error] {exc}\n")

if __name__ == "__main__":
    main()
