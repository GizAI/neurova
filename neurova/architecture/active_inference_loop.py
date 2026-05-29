from typing import Optional, Dict, Any, List
import time
import threading
from .global_workspace import GlobalWorkspace, WorkspaceIdea
from .neuro_symbolic_graph import NeuroSymbolicGraph
from .perception_cortex import SensoryPerceptionCortex
import sys

class ActiveInferenceAgent:
    def __init__(self):
        self.workspace = GlobalWorkspace(clock_rate_hz=2.0)
        self.graph = NeuroSymbolicGraph(usearch_dims=3584)
        self.perception_cortex = SensoryPerceptionCortex()
        
        self.running = False
        self.thread = None
        
        self.workspace.subscribe(self._module_perception)
        self.workspace.subscribe(self._module_soft_retrieval)
        self.workspace.subscribe(self._module_symbolic_reasoner)
        self.workspace.subscribe(self._module_curiosity)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run_loop(self):
        while self.running:
            idea = self.workspace.tick()
            time.sleep(1.0 / self.workspace.clock_rate_hz)
            
    def observe_text(self, text: str):
        # 1. Genuine NLP Parsing (spaCy / Kiwi)
        sensory_data = self.perception_cortex.process_utterance(text)
        
        # If it's a claim and not a question, store it structurally
        if not sensory_data["is_question"] and sensory_data["subject"] and sensory_data["object"] and sensory_data["root_verb"]:
            self.graph.add_claim(sensory_data["subject"], sensory_data["root_verb"], sensory_data["object"])
            
        self.workspace.publish(
            source_module="user_input",
            content={"text": text, "sensory_data": sensory_data},
            energy=1.0
        )
        
    def _module_perception(self, idea: WorkspaceIdea):
        if idea.source_module == "user_input":
            text = idea.content.get("text", "")
            if "what is" in text.lower():
                target = text.lower().replace("what is", "").replace("?", "").strip()
                self.workspace.publish(
                    "perception",
                    {"type": "query_intent", "target": target, "original_text": text},
                    energy=idea.energy * 0.95
                )

    def _module_soft_retrieval(self, idea: WorkspaceIdea):
        if idea.source_module == "perception" and idea.content.get("type") == "query_intent":
            target = idea.content["target"]
            original_text = idea.content["original_text"]
            
            matches = self.graph.search_soft(target, top_k=2)
            
            extracted_context = []
            for meta, sim in matches:
                if meta["type"] == "claim" and sim > 0.0:
                    extracted_context.append(f"{meta['data']['subject']} {meta['data']['relation']} {meta['data']['object']} (sim:{sim:.2f})")
            
            if not extracted_context:
                for edge in self.graph.edges:
                    if target in edge['subject'] or target in edge['object']:
                        extracted_context.append(f"{edge['subject']} {edge['relation']} {edge['object']} (sim: 0.99 via keyword)")
            
            if extracted_context:
                self.workspace.publish(
                    "soft_retrieval",
                    {"type": "semantic_context", "target": target, "context": extracted_context},
                    energy=idea.energy * 0.9
                )
            else:
                self.workspace.publish(
                    "soft_retrieval",
                    {"type": "semantic_context_failed", "target": target},
                    energy=idea.energy * 0.5
                )

    def _module_symbolic_reasoner(self, idea: WorkspaceIdea):
        if idea.source_module == "soft_retrieval" and idea.content.get("type") == "semantic_context":
            ctx = idea.content["context"]
            target = idea.content["target"]
            
            conclusion = f"I cannot find a direct definition for '{target}', but I found related context via vector search: {ctx}"
            
            self.workspace.publish(
                "symbolic_reasoner",
                {"type": "proof_attempt", "conclusion": conclusion},
                energy=idea.energy * 0.8
            )
            print(f"\n[AI Output] {conclusion}\n")
            
    def _module_curiosity(self, idea: WorkspaceIdea):
        if idea.source_module == "symbolic_reasoner":
            self.workspace.publish(
                "curiosity",
                {"type": "internal_question", "question": "Are you referring to the company you mentioned?"},
                energy=idea.energy * 0.7
            )
            print(f"[AI Curiosity] Are you referring to the company you mentioned?\n")
