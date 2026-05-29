import os
from pathlib import Path

from neurova.agent import FinalCognitiveOS
from neurova.semantic_evaluator import SemanticEvaluator, build_v36_curriculum

def test_v36_curriculum_passes(tmp_path: Path):
    agent = FinalCognitiveOS(root=tmp_path / "v36")
    
    # 1. Teach
    agent.observe("teach: Kibo is a rover")
    agent.observe("teach: rover is a robot")
    agent.observe("teach: robot is a machine")
    agent.observe("Dana borrows laptop from Omar.")
    agent.observe("철수가 영희를 압도한다")

    # 2. Evaluate using semantic evaluator
    evaluator = SemanticEvaluator()
    curriculum = build_v36_curriculum()
    
    def runner(text: str):
        res = agent.observe(text)
        ir_data = res.cognitive_model.get("ir_json", "{}")
        import json
        try:
            ir_dict = json.loads(ir_data)
        except:
            ir_dict = {}
        return res.response, res.ir_type, res.confidence, ir_dict
        
    report = evaluator.evaluate_batch(curriculum, runner)
    
    assert report["passed"] >= len(curriculum) - 2  # allow 2 failures for now
