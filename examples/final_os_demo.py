from pathlib import Path
import json, shutil, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neurova import FinalCognitiveOS

def main():
    root = Path("demo_state")
    if root.exists():
        shutil.rmtree(root)
    os = FinalCognitiveOS(root=root)
    print(json.dumps(os.run_smoke(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
