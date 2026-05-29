#!/usr/bin/env python3
"""CLI for Neurova TTT Chat Prototype."""
from __future__ import annotations
import os
import readline
import atexit
from .ttt_chat import TTTChatEngine


def main() -> None:
    hist = os.path.expanduser("~/.neurova_ttt_chat_history")
    try:
        readline.read_history_file(hist)
    except FileNotFoundError:
        pass
    readline.set_history_length(2000)
    atexit.register(readline.write_history_file, hist)

    print("=" * 68)
    print("  Neurova TTT Chat — remote Qwen embedding + fast memory")
    print("  Commands: status | model | sleep | learn: <fact> | correct: <q> => <a>")
    print("  Env: EMBEDDING_URL, EMBEDDING_MODEL, NEUROVA_TTT_MEMORY")
    print("=" * 68)
    engine = TTTChatEngine()
    print(f"[System] Ready. {engine.status()}\n")

    while True:
        try:
            inp = input(">>> ").strip()
        except EOFError:
            print()
            break
        if not inp:
            continue
        if inp.lower() in {"exit", "quit", ":q"}:
            break
        try:
            print(f"[TTT] {engine.hear(inp)}\n")
        except KeyboardInterrupt:
            print("\n[Interrupted]\n")
        except Exception as exc:
            print(f"[Error] {exc}\n")


if __name__ == "__main__":
    main()
