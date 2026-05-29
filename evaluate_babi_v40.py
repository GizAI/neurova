import os
import sys
import time
from pathlib import Path
from neurova.architecture.v40_engine import V40Engine

def run_evaluation():
    lines = [
        "John travelled to the hallway.",
        "Mary journeyed to the bathroom.",
        "Where is John? \thallway",
        "Daniel went back to the bathroom.",
        "John moved to the bedroom.",
        "Where is Mary? \tbathroom",
    ]

    print("Initializing V40 Engine for bAbI Eval...")
    agent = V40Engine()
    agent.start()

    correct = 0
    total = 0
    
    for content in lines:
        if "\t" in content:
            q, ans = content.split("\t")
            total += 1
            response = agent.speak_to(q, timeout=4.0)
            
            is_correct = ans.lower() in response.lower()
            if is_correct:
                correct += 1
                
            print(f"Q: {q}")
            print(f"Expected: {ans} | Got: {response}")
            print(f"Result: {'PASS' if is_correct else 'FAIL'}\n")
        else:
            print(f"Injecting: {content}")
            res = agent.speak_to(content, timeout=4.0)
            print(f"Internal: {res}")
            
    print("=" * 40)
    print(f"bAbI Task 1 Evaluation Results:")
    print(f"Accuracy: {correct}/{total} ({correct/max(1,total)*100:.1f}%)")
    print("=" * 40)
    
    agent.stop()

if __name__ == "__main__":
    run_evaluation()
