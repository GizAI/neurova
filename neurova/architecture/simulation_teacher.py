import time
from neurova.architecture.v40_engine import V40Engine

def run_simulation():
    print("\n=======================================================")
    print(" V40 Active Inference Engine - Primary School Simulation")
    print("=======================================================\n")
    
    agent = V40Engine()
    agent.start()
    
    dialogue = [
        "A plant is a living thing.",
        "An animal is a living thing.",
        "A rock is not a living thing.",
        "A living thing grows.",
        "A living thing needs water.",
        "A sunflower is a plant.",
        "A dog is an animal.",
        "Does a sunflower grow?",
        "What does a dog need?",
        "Does a rock grow?"
    ]
    
    for text in dialogue:
        print(f"[Teacher] {text}")
        response = agent.speak_to(text, timeout=10.0) # increased timeout for full dialogue
        print(f"[V40 AI]  {response}\n")
        time.sleep(1.0)
        
    agent.stop()
    print("=======================================================")
    print(" Simulation Completed.")
    print("=======================================================\n")

if __name__ == "__main__":
    run_simulation()
