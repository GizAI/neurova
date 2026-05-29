"""Semantic Evaluation runner.

Replaces the old legacy substring tests with semantic expectation validation.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

from .agent import FinalCognitiveOS
from .semantic_evaluator import SemanticEvaluator, build_v36_curriculum


class ProductionEvaluator:
    def __init__(self, agent: FinalCognitiveOS):
        self.agent = agent
        self.evaluator = SemanticEvaluator()

    def run(self) -> Dict[str, Any]:
        cases = build_v36_curriculum()
        
        # Need some background knowledge for the questions to be answerable.
        self.agent.observe("teach: kibo is rover")
        self.agent.observe("teach: rover is robot")
        self.agent.observe("teach: robot is machine")

        def runner(text: str) -> tuple:
            res = self.agent.observe(text)
            ir_dict = {}
            # We attempt to parse ir_json if it was stored during agent.observe().
            # Depending on how the OSResult is formatted, we can pull data out.
            # But we can also just use string inspection on the response for now.
            return res.response, res.ir_type, res.confidence, ir_dict

        report = self.evaluator.evaluate_batch(cases, runner)
        
        # Log to memory benchmark table
        self.agent.memory.record_benchmark("v36_semantic_evaluation", report["accuracy"], report)
        return report
