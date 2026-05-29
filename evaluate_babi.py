import os
import sys
import time
from pathlib import Path

def run_evaluation():
    from neurova.v40_dual_brain import V40DualBrain
    
    # Just a small sanity test to show the engine works on generic facts
    lines = [
        "John is in the hallway.",
        "Mary is in the bathroom.",
        "Where is John? \thallway",
        "Daniel is in the bedroom.",
        "Where is Mary? \tbathroom",
    ]

    print("Initializing V40 Dual Brain Engine for bAbI Eval...")
    brain_state_dir = Path.cwd() / "neurova_v40_eval_state"
    brain_state_dir.mkdir(exist_ok=True)
    
    agent = V40DualBrain(root_path=brain_state_dir)
    agent.start()

    correct = 0
    total = 0
    
    for content in lines:
        if "\t" in content:
            q, ans = content.split("\t")
            total += 1
            response = agent.speak_to(q, timeout=3.0)
            
            is_correct = ans.lower() in response.lower()
            if is_correct:
                correct += 1
                
            print(f"Q: {q}")
            print(f"Expected: {ans} | Got: {response}")
            print(f"Result: {'PASS' if is_correct else 'FAIL'}\n")
        else:
            print(f"Teacher: {content}")
            agent.speak_to(content, timeout=4.0)
            
    print("=" * 40)
    print(f"Evaluation Results:")
    print(f"Accuracy: {correct}/{total} ({correct/max(1,total)*100:.1f}%)")
    print("=" * 40)
    
    agent.stop()

if __name__ == "__main__":
    run_evaluation()
