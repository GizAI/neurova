#!/usr/bin/env python3
"""Neurova In-Place TTT CLI v3 — Qwen3.5-4B + 진짜 TTT 스트리밍.
 
Usage:
  python -m neurova.neurova_cli
  NEUROVA_MAX_CONTEXT=262144 NEUROVA_MAX_TOKENS=8192 python -m neurova.neurova_cli
"""
import os, sys, readline, atexit, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from neurova.neurova_inplace_ttt import NeurovaInPlaceTTT, MODEL_NAME, TTT_LAYERS
from typing import Generator


def print_banner():
    ctx = os.environ.get("NEUROVA_MAX_CONTEXT", "262144")
    tokens = os.environ.get("NEUROVA_MAX_TOKENS", "8192")
    passes = os.environ.get("NEUROVA_TTT_PASSES", "5")
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print(f"║  Neurova In-Place TTT v3 — {MODEL_NAME:<28s}║")
    print(f"║  Context: {int(ctx):>6,} | Max tokens: {int(tokens):>5,} | TTT: {TTT_LAYERS}  ║")
    print(f"║  Passes: {passes} | 절대 언어규칙 하드코딩 금지 | 언어를 배울 그릇       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Commands:                                              ║")
    print("║    ttt: on/off     Load/unload model                    ║")
    print("║    <text>          Generate response (streaming)        ║")
    print("║    learn: Q => A   TTT learn + episodic memory          ║")
    print("║    /verify         Check TTT weight changes              ║")
    print("║    thinking:on/off Toggle reasoning display             ║")
    print("║    passes: N       Set TTT multi-pass count (1-20)       ║")
    print("║    context: N      Set context size                     ║")
    print("║    status          Show engine status                   ║")
    print("║    /clear          Reset history + TTT momentum          ║")
    print("║    exit/quit       Exit                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def main():
    # Readline history
    hist = os.path.expanduser("~/.neurova_cli_history")
    try:
        readline.read_history_file(hist)
    except FileNotFoundError:
        pass
    readline.set_history_length(5000)
    atexit.register(readline.write_history_file, hist)

    print_banner()
    engine = NeurovaInPlaceTTT()
    print("[System] Ready. Type 'ttt: on' to load the model.\n")

    while True:
        try:
            inp = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not inp:
            continue
        if inp.lower() in ("exit", "quit", ":q", ":exit"):
            break

        try:
            result = engine.hear(inp)
            
            # Handle streaming (generator returned)
            if isinstance(result, Generator):
                collected = ""
                for token in result:
                    print(token, end="", flush=True)
                    collected += token
                # Generator exhausted — get return value
                print()  # final newline
            elif result:
                print(f"[Neurova] {result}")
            
            print()  # blank line after response
            
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as exc:
            print(f"[Error] {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
