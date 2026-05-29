from .agent import FinalCognitiveOS
class ProductionEvaluator:
    def __init__(self, agent: FinalCognitiveOS): self.agent=agent
    def run(self): return self.agent.run_smoke()
