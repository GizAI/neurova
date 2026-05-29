"""BrainOS interactive CLI REPL."""
import sys
import argparse
from pathlib import Path
from .agent import FinalCognitiveOS


def main():
    parser = argparse.ArgumentParser(description="BrainOS interactive CLI")
    parser.add_argument("--root", default=None, help="State persistence directory (default: ./neurova_state)")
    parser.add_argument("--seed", action="store_true", help="Auto-seed with builtin child-language seed")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test and exit")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path.cwd() / "neurova_state"

    os = FinalCognitiveOS(root=root, auto_seed=args.seed)

    if args.smoke:
        import json
        result = os.run_smoke()
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2))
        return

    print("BrainOS interactive CLI. Type 'quit' or 'exit' to stop.")
    print("Just talk naturally — statements are remembered, questions are answered.")
    print()

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            break

        result = os.observe(text)
        print(f"  [{result.ir_type}] {result.response}")
        print()


if __name__ == "__main__":
    main()
