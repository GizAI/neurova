"""
EnhancedBrain — Neurova with GPU embedding + learning loop.

Wraps Brain with:
- EmbeddingMemory for all statements, failures, and queries
- USearch vector index for fast semantic retrieval
- Automatic construction creation from failure clusters
- Sleep consolidation with embedding-based pattern discovery

Usage:
    from neurova.enhanced_brain import EnhancedBrain
    brain = EnhancedBrain()
    brain.hear("Korea is a peninsular region.")
    brain.feedback("What is Korea?", "Korea is in East Asia.")
    brain.sleep_cycle()  # Consolidate and learn
"""

import os, time, uuid, json
from typing import Optional

from neurova.engine import Brain, ConstructionMemory, Construction, compile_text
from neurova.embedding_memory import EmbeddingMemory, MemItem


class EnhancedBrain(Brain):
    """Brain with GPU embedding and self-learning loop."""
    
    def __init__(self, embedding_url: str = "http://ml-dmc8:8081/v1/embeddings",
                 embedding_model: str = "Qwen/Qwen3-Embedding-4B"):
        super().__init__()
        self.emem = EmbeddingMemory(url=embedding_url, model=embedding_model)
        self._init_embedding = False
    
    def hear(self, text: str) -> str:
        """Override hear to also store in EmbeddingMemory."""
        result = super().hear(text)
        
        # Store in embedding memory
        is_q = text.strip().endswith("?")
        category = "query" if is_q else "statement"
        
        # For statements, store the construction used
        if not is_q:
            # Find which construction parsed this
            for c in self.cmem.constructions.values():
                if c.last_used > time.time() - 0.1:  # Just used
                    self.emem.store(text, {
                        "construction": c.id,
                        "event_type": c.event_type,
                        "answer": result
                    }, category)
                    break
            else:
                self.emem.store(text, {"answer": result}, category)
        else:
            self.emem.store(text, {"answer": result}, category)
        
        return result
    
    def feedback(self, question: str, correct_answer: str) -> str:
        """Enhanced feedback: learn from wrong answer using embeddings."""
        wrong = self.qplanner.answer(question)
        old_entity_count = len(self.model.entities)
        
        # Store the failure
        self.emem.store(question, {
            "type": "failure",
            "wrong_answer": wrong,
            "correction": correct_answer
        }, "failure")
        
        # Try to parse correction
        n = compile_text(correct_answer, self.model, self.coref, self.cmem, self.epmem)
        
        # If parsing failed, try embedding-based construction matching
        if n == 0:
            # Search for similar past statements that WERE parsed
            similar = self.emem.search(correct_answer, top_k=3, category="statement")
            
            if similar:
                best_score, best_item = similar[0]
                ctype = best_item.metadata.get("construction", "")
                if best_score > 0.65 and ctype:
                    # Create a new construction inspired by the similar one
                    template_c = self.cmem.constructions.get(ctype)
                    if template_c:
                        # Get the ROOT verb from correction
                        import spacy
                        try:
                            doc = spacy.load("en_core_web_sm")(correct_answer)
                            for tok in doc:
                                if tok.dep_ == "ROOT":
                                    root_lemma = tok.lemma_.lower()
                                    cid = f"eb_{root_lemma}"
                                    if cid not in self.cmem.constructions:
                                        new_c = Construction(
                                            id=cid,
                                            event_type=template_c.event_type,
                                            trigger_lemmas=[root_lemma],
                                            trigger_deps=["ROOT"],
                                            role_mapping=dict(template_c.role_mapping),
                                            prep_signals=dict(template_c.prep_signals),
                                            confidence=0.4
                                        )
                                        self.cmem.constructions[cid] = new_c
                                        # Retry
                                        n = compile_text(correct_answer, self.model, 
                                                       self.coref, self.cmem, self.epmem)
                                    break
                        except:
                            pass
            else:
                # Try to create generic construction from embedding neighbors
                if self._try_generic_parse(correct_answer):
                    n = compile_text(correct_answer, self.model, self.coref, 
                                   self.cmem, self.epmem)
        
        # Store the correction
        self.emem.store(correct_answer, {
            "type": "correction",
            "for_question": question,
            "parsed": n > 0
        }, "statement")
        
        self.epmem.record(question, "", None, "", n > 0)
        new_entities = len(self.model.entities) - old_entity_count
        
        if n > 0:
            return f"Learned: {correct_answer} ({new_entities} new facts, via embeddings)"
        
        # Last resort: store as attribute
        import spacy
        try:
            doc = spacy.load("en_core_web_sm")(correct_answer)
            for tok in doc:
                if tok.dep_ == "ROOT":
                    root_lemma = tok.lemma_.lower()
                    cid = f"gen_{root_lemma}"
                    if cid not in self.cmem.constructions:
                        new_c = Construction(
                            id=cid, event_type="STATEMENT",
                            trigger_lemmas=[root_lemma],
                            trigger_deps=["ROOT"],
                            role_mapping={"nsubj": "entity", "dobj": "target"},
                            prep_signals={"in": "location", "on": "location", "at": "location"},
                            confidence=0.3
                        )
                        self.cmem.constructions[cid] = new_c
                        n2 = compile_text(correct_answer, self.model, self.coref,
                                        self.cmem, self.epmem)
                        if n2 > 0:
                            return f"Learned: {correct_answer} (via generic parse)"
                    break
        except:
            pass
        
        return f"Noted: {correct_answer}"
    
    def _try_generic_parse(self, text: str) -> bool:
        """Try to find enough embedding similarity to create a parse."""
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for tok in doc:
                if tok.dep_ == "ROOT":
                    root_lemma = tok.lemma_.lower()
                    if root_lemma in ("be", "is", "are", "am", "was", "were"):
                        return True  # Already handled by classify
                    # Check if similar verb has been seen
                    similar = self.emem.search(root_lemma, top_k=1)
                    if similar and similar[0][0] > 0.6:
                        return True
        except:
            pass
        return False
    
    def sleep_cycle(self) -> str:
        """Consolidation: cluster failures, suggest new constructions."""
        # Base consolidation
        base_report = super().sleep_cycle()
        
        # Cluster failures
        clusters = self.emem.cluster_failures(min_cluster=3)
        
        new_constructions = 0
        for cluster in clusters:
            # Find common pattern in cluster
            texts = [item.text for item in cluster]
            # Create a new construction from common elements
            for text in texts:
                import spacy
                try:
                    nlp = spacy.load("en_core_web_sm")
                    doc = nlp(text)
                    for tok in doc:
                        if tok.dep_ == "ROOT":
                            root_lemma = tok.lemma_.lower()
                            cid = f"sleep_{root_lemma}"
                            if cid not in self.cmem.constructions:
                                new_c = Construction(
                                    id=cid, event_type="STATEMENT",
                                    trigger_lemmas=[root_lemma],
                                    role_mapping={"nsubj": "entity", "dobj": "target"},
                                    prep_signals={"in": "location"},
                                    confidence=0.25
                                )
                                self.cmem.constructions[cid] = new_c
                                new_constructions += 1
                            break
                except:
                    continue
        
        emem_stats = f"{self.emem.count('statement')} statements, {self.emem.count('failure')} failures"
        return f"{base_report} | Embedding: {emem_stats} | New constructions: {new_constructions} | Clusters: {len(clusters)}"
    
    def get_status(self) -> str:
        base = super().get_status()
        emb_count = self.emem.count()
        return f"{base} | EmbeddingMemory: {emb_count} items"
