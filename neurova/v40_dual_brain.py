import time
import threading
from typing import Dict, Any
from pathlib import Path

from .agent import FinalCognitiveOS
from .architecture.global_workspace import GlobalWorkspace, WorkspaceIdea
from .architecture.neuro_symbolic_graph import NeuroSymbolicGraph
from .architecture.perception_cortex import SensoryPerceptionCortex

class V40DualBrain:
    """The Ultimate Neuro-Symbolic Integration (System 1 + System 2)."""
    def __init__(self, root_path: Path):
        self.workspace = GlobalWorkspace(clock_rate_hz=20.0)
        
        # System 2: Left Brain (Strict Logic, Grammar, Exact DB)
        self.left_brain = FinalCognitiveOS(root=root_path, auto_seed=True)
        
        # System 1: Right Brain (Fast Association, GPU Embeddings, USearch)
        self.right_brain = NeuroSymbolicGraph(usearch_dims=3584)
        
        # Senses: NLP Perception
        self.perception = SensoryPerceptionCortex()
        
        self.running = False
        self.thread = None
        self._current_response = None
        self._response_event = threading.Event()
        
        # Bind the neural pathways
        self.workspace.subscribe(self._sensory_input)
        self.workspace.subscribe(self._system2_logic)
        self.workspace.subscribe(self._system1_association)
        self.workspace.subscribe(self._synthesis_and_speech)
        self.workspace.subscribe(self._curiosity_daemon)
        
        self._sync_memories()

    def _sync_memories(self):
        """Warm up the Right Brain with all existing Left Brain knowledge."""
        rows = self.left_brain.memory.conn.execute("SELECT subject, relation, object FROM claims").fetchall()
        for r in rows:
            self.right_brain.add_claim(r["subject"], r["relation"], r["object"])

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.left_brain.close()

    def _run_loop(self):
        while self.running:
            self.workspace.tick()
            time.sleep(1.0 / self.workspace.clock_rate_hz)

    def speak_to(self, text: str, timeout: float = 5.0) -> str:
        self._current_response = None
        self._response_event.clear()
        
        self.workspace.publish(
            source_module="ear",
            content={"text": text},
            energy=1.0 # High attention spike
        )
        
        if self._response_event.wait(timeout):
            return self._current_response
        return "(Lost in thought...)"

    # --- Neural Pathways ---

    def _sensory_input(self, idea: WorkspaceIdea):
        if idea.source_module == "ear":
            text = idea.content["text"]
            sensory = self.perception.process_utterance(text)
            
            if sensory is None:
                return
            
            # Send to both brains simultaneously
            self.workspace.publish(
                "sensory",
                {"text": text, "sensory": sensory},
                energy=idea.energy * 0.95
            )

    def _system2_logic(self, idea: WorkspaceIdea):
        """Left Brain: Attempts strict parsing, schema learning, and logical proof."""
        if idea.source_module == "sensory":
            text = idea.content["text"]
            sensory = idea.content["sensory"]
            
            # Execute V36 Core Pipeline
            result = self.left_brain.observe(text)
            
            # Sync new facts to Right Brain
            if not sensory["is_question"] and sensory["subject"] and sensory["root_verb"]:
                obj = sensory["object"] or ""
                self.right_brain.add_claim(sensory["subject"], sensory["root_verb"], obj)

            # If Left Brain proves it or strictly refutes it, it wins immediately.
            if "I cannot prove" not in result.response and "I don't have enough" not in result.response:
                self.workspace.publish(
                    "system2",
                    {"type": "strict_answer", "response": result.response},
                    energy=idea.energy * 0.9  # High confidence
                )
            else:
                # Left Brain failed (Rule mismatch). Trigger Right Brain rescue.
                self.workspace.publish(
                    "system2",
                    {"type": "logic_failed", "text": text, "sensory": sensory},
                    energy=idea.energy * 0.8
                )

    def _system1_association(self, idea: WorkspaceIdea):
        """Right Brain: Steps in when Left Brain fails. Uses GPU Vector Search."""
        if idea.source_module == "system2" and idea.content["type"] == "logic_failed":
            sensory = idea.content["sensory"]
            target = sensory["subject"] or sensory["object"] or sensory["raw_text"].replace("?", "").strip()
            if target:
                target = target.replace("a ", "").replace("an ", "").replace("the ", "")

            matches = self.right_brain.search_soft(target, top_k=5)
            ctx = []
            for meta, sim in matches:
                if meta["type"] == "claim" and sim > 0.4:
                    ctx.append({"subj": meta['data']['subject'], "rel": meta['data']['relation'], "obj": meta['data']['object']})
            
            if not ctx:
                for edge in self.right_brain.edges:
                    if target in edge['subject'] or target in edge['object']:
                        ctx.append({"subj": edge['subject'], "rel": edge['relation'], "obj": edge['object']})
            
            if ctx:
                deductions = []
                for c in ctx:
                    deductions.append(f"{c['subj']} {c['rel']} {c['obj']}")
                ans = "Logically I can't prove it, but intuitively I associate it with: " + ", ".join(set(deductions))
                self.workspace.publish(
                    "system1",
                    {"type": "associative_answer", "response": ans},
                    energy=idea.energy * 0.85
                )
            else:
                self.workspace.publish(
                    "system1",
                    {"type": "total_failure"},
                    energy=idea.energy * 0.5
                )

    def _synthesis_and_speech(self, idea: WorkspaceIdea):
        if idea.source_module in ("system2", "system1"):
            if idea.content.get("type") in ("strict_answer", "associative_answer"):
                self._current_response = idea.content["response"]
                self._response_event.set()
        elif idea.source_module == "system1" and idea.content.get("type") == "total_failure":
            self._current_response = "I have no logical proof nor any associative memory for that."
            self._response_event.set()

    def _curiosity_daemon(self, idea: WorkspaceIdea):
        """Meta-cognition: Generates questions when learning new associations."""
        if idea.source_module == "sensory":
            sensory = idea.content["sensory"]
            if not sensory["is_question"] and sensory["subject"]:
                # The agent gets curious about the subject just learned
                pass # Can be expanded to push questions to the UI asynchronously
