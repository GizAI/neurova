import sys
from neurova.agent import FinalCognitiveOS
from pathlib import Path

def run():
    print("Initializing...")
    os = FinalCognitiveOS(root=Path.cwd() / "neurova_test_state", auto_seed=False)
    
    # Test 1: Who are you?
    r1 = os.observe("Who are you?")
    print(f"Q: Who are you?\nA: {r1.response}\n")

    # Test 2: I am Kyngtae
    r2 = os.observe("I am Kyngtae.")
    print(f"Q: I am Kyngtae.\nA: {r2.response}\n")

    # Test 3: Who am I?
    r3 = os.observe("Who am I?")
    print(f"Q: Who am I?\nA: {r3.response}\n")
    
if __name__ == "__main__":
    run()
