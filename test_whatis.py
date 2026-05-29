import sys
from neurova.agent import FinalCognitiveOS
from pathlib import Path

def run():
    print("Initializing...")
    os = FinalCognitiveOS(root=Path.cwd() / "neurova_test_state", auto_seed=False)
    
    # Test 1: I am CEO of Giz Inc
    r1 = os.observe("I am CEO of Giz Inc.")
    print(f"Q: I am CEO of Giz Inc.\nA: {r1.response}\n")

    # Test 2: What is Giz?
    r2 = os.observe("What is Giz?")
    print(f"Q: What is Giz?\nA: {r2.response}\n")

    
if __name__ == "__main__":
    run()
