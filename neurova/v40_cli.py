import os
import sys
import readline
import atexit
from neurova.architecture.v40_engine_clean import V40CleanEngine

def main():
    histfile = os.path.join(os.path.expanduser("~"), ".neurova_v40_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    atexit.register(readline.write_history_file, histfile)

    print("=" * 55)
    print(" Neurova V40 Clean Engine CLI")
    print(" Up/down arrows for history. 'exit' to quit.")
    print("=" * 55)
    print()

    engine = V40CleanEngine()
    print("[ready]")

    while True:
        try:
            inp = input(">>> ").strip()
        except EOFError:
            break
        if not inp:
            continue
        if inp.lower() in ("exit", "quit"):
            break

        response = engine.hear(inp)
        print(f"[V40] {response}\n")


if __name__ == "__main__":
    main()
