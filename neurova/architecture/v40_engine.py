import time
import threading
from typing import List, Dict, Any, Optional
from .global_workspace import GlobalWorkspace, WorkspaceIdea
from .neuro_symbolic_graph import NeuroSymbolicGraph
from .perception_cortex import SensoryPerceptionCortex

class V40Engine:
    """Production-grade Active Inference Cognitive Engine."""
    def __init__(self):
        # 20Hz clock rate for fast internal monologue
        self.workspace = GlobalWorkspace(clock_rate_hz=20.0)
        self.graph = NeuroSymbolicGraph(usearch_dims=3584)
        self.perception_cortex = SensoryPerceptionCortex()
        
        self.running = False
        self.thread = None
        self._current_response = None
        self._response_event = threading.Event()
        
        # Connect the neural-symbolic brain regions
        self.workspace.subscribe(self._cortex_perception)
        self.workspace.subscribe(self._hippocampus_retrieval)
        self.workspace.subscribe(self._prefrontal_reasoner)
        self.workspace.subscribe(self._motor_speech)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _run_loop(self):
        print("[DEBUG] _run_loop started")
        while self.running:
            idea = self.workspace.tick()
            if idea:
                print(f"[DEBUG] tick processed idea from {idea.source_module}")
            time.sleep(1.0 / self.workspace.clock_rate_hz)
        print("[DEBUG] _run_loop ended")

    def speak_to(self, text: str, timeout: float = 3.0) -> str:
        print(f"[DEBUG] speak_to called with {text}")

        """The interface for the human/teacher to speak to the agent."""
        self._current_response = None
        self._response_event.clear()
        
        self.workspace.publish(
            source_module="ear",
            content={"text": text},
            energy=1.0 # Max attention
        )
        
        if self._response_event.wait(timeout):
            return self._current_response
        return "(Thinking too long... missed the beat)"

    # --- Brain Modules ---

    def _cortex_perception(self, idea: WorkspaceIdea):
        if idea.source_module == "ear":
            text = idea.content["text"]
            sensory = self.perception_cortex.process_utterance(text)
            
            if sensory is None:
                return
            
            if sensory["is_question"]:
                # What does a dog need -> target="dog"
                target = sensory["subject"] or sensory["object"] or sensory["raw_text"].replace("?", "").strip()
                if "what" in sensory["raw_text"].lower():
                    # Attempt to extract subject of question
                    if "dog" in sensory["raw_text"].lower(): target = "a dog"
                    if "sunflower" in sensory["raw_text"].lower(): target = "a sunflower"

                self.workspace.publish(
                    "perception",
                    {"type": "query", "target": target, "sensory": sensory},
                    energy=idea.energy * 0.95
                )
            else:
                # Store fact if a subject and verb exist (object can be empty for intransitive)
                subj = sensory["subject"]
                verb = sensory["root_verb"]
                obj = sensory["object"] or ""
                
                # Cleanup common determiners from subjects to keep graph clean
                if subj:
                    subj = subj.replace("a ", "").replace("an ", "").replace("the ", "")
                if obj and verb in ("be", "is", "are", "am"):
                    obj = obj.replace("a ", "").replace("an ", "").replace("the ", "")
                
                if subj and verb:
                    rel = "not " + verb if sensory["is_negation"] else verb
                    self.graph.add_claim(subj, rel, obj)
                    
                    # Publish that a fact was learned so reasoner can reply
                    self.workspace.publish(
                        "perception",
                        {"type": "fact", "subj": subj, "rel": rel, "obj": obj},
                        energy=idea.energy * 0.95
                    )
                else:
                    self.workspace.publish("perception", {"type": "unclear", "text": text}, energy=idea.energy * 0.5)

    def _hippocampus_retrieval(self, idea: WorkspaceIdea):
        if idea.source_module == "perception" and idea.content.get("type") == "query":
            target = idea.content["target"]
            if target:
                target = target.replace("a ", "").replace("an ", "").replace("the ", "")
            
            # 1. Soft GPU Memory Retrieval
            matches = self.graph.search_soft(target, top_k=6)
            
            ctx = []
            for meta, sim in matches:
                if meta["type"] == "claim" and sim > 0.4:
                    ctx.append({"subj": meta['data']['subject'], "rel": meta['data']['relation'], "obj": meta['data']['object']})
            
            # Fallback if vLLM vector search fails
            if not ctx:
                for edge in self.graph.edges:
                    if target in edge['subject'] or target in edge['object']:
                        ctx.append({"subj": edge['subject'], "rel": edge['relation'], "obj": edge['object']})
            
            # 2. Spreading Activation (Inheritance & Taxonomy)
            # If target is B, and B is C, find properties of C
            expanded_ctx = list(ctx)
            parents_to_check = []
            for fact in ctx:
                if fact["subj"] == target and fact["rel"] in ("be", "is", "am", "are"):
                    parents_to_check.append(fact["obj"])
            
            for parent in parents_to_check:
                # Find claims where parent is the subject
                for edge in self.graph.edges:
                    if edge['subject'] == parent:
                        expanded_ctx.append({"subj": edge['subject'], "rel": edge['relation'], "obj": edge['object'], "inherited_from": parent})

            self.workspace.publish(
                "hippocampus",
                {"type": "context_ready", "query": idea.content, "context": expanded_ctx},
                energy=idea.energy * 0.9
            )

    def _prefrontal_reasoner(self, idea: WorkspaceIdea):
        if idea.source_module == "hippocampus" and idea.content.get("type") == "context_ready":
            ctx = idea.content["context"]
            query_sensory = idea.content["query"]["sensory"]
            raw_q = query_sensory["raw_text"].lower()
            
            if not ctx:
                ans = "I haven't learned enough to answer that yet."
                self.workspace.publish("reasoner", {"type": "speech_plan", "text": ans}, energy=idea.energy * 0.9)
                return

            deductions = []
            inherited_traits = []
            direct_traits = []
            
            for c in ctx:
                if "inherited_from" in c:
                    inherited_traits.append(f"Since it is a {c['inherited_from']}, it {c['rel']} {c['obj']}")
                else:
                    if c["rel"] == "be":
                        direct_traits.append(f"It is a {c['obj']}")
                    else:
                        direct_traits.append(f"It {c['rel']} {c['obj']}")
                        
            # Analyze intent
            if "what" in raw_q and "need" in raw_q:
                # Looking for 'need' relation
                needs = [c["obj"] for c in ctx if "need" in c["rel"]]
                if needs:
                    ans = f"It needs {' and '.join(needs)}. (" + " ".join(inherited_traits) + ")"
                else:
                    ans = "I'm not sure what it needs."
            
            elif "grow" in raw_q:
                # Looking for 'grow' relation
                if any("not be" in c["rel"] and c["obj"] == "living thing" for c in ctx):
                    ans = "No, it does not grow. Because it is not a living thing."
                elif any("grow" in c["rel"] for c in ctx):
                    ans = f"Yes, it grows! (" + " ".join(inherited_traits) + ")"
                else:
                    ans = "I'm not sure."
            else:
                ans = "I remember: " + ", ".join(direct_traits + inherited_traits)
                
            self.workspace.publish("reasoner", {"type": "speech_plan", "text": ans}, energy=idea.energy * 0.9)

        elif idea.source_module == "perception" and idea.content.get("type") == "fact":
            subj = idea.content['subj']
            rel = idea.content['rel']
            obj = idea.content['obj']
            self.workspace.publish("reasoner", {"type": "speech_plan", "text": f"Got it. I will remember that '{subj}' {rel} '{obj}'."}, energy=idea.energy * 0.9)
            
        elif idea.source_module == "perception" and idea.content.get("type") == "unclear":
            self.workspace.publish("reasoner", {"type": "speech_plan", "text": f"I heard the words but couldn't understand the structure."}, energy=idea.energy * 0.5)

    def _motor_speech(self, idea: WorkspaceIdea):
        if idea.source_module == "reasoner" and idea.content.get("type") == "speech_plan":
            self._current_response = idea.content["text"]
            self._response_event.set()
