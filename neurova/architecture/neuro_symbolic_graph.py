import numpy as np
from typing import Dict, Any, List
from .usearch_index import HybridSemanticIndex
from .ml_dmc8_embedding import VLLMEmbeddingClient
import uuid

class NeuroSymbolicGraph:
    """The hybrid storage combining SQLite strictness with USearch vector space."""
    def __init__(self, usearch_dims: int = 3584):
        self.vector_index = HybridSemanticIndex(dims=usearch_dims)
        self.embedding_client = VLLMEmbeddingClient()
        self.nodes = {}
        self.edges = []
        self._next_id = 0
        
    def _get_id(self) -> int:
        self._next_id += 1
        return self._next_id
        
    def add_claim(self, subject: str, relation: str, object_str: str) -> None:
        """Stores a claim in both symbolic (nodes/edges) and semantic (usearch) space."""
        # 1. Symbolic storage
        if subject not in self.nodes:
            self.nodes[subject] = {"id": self._get_id(), "type": "entity", "text": subject}
        if object_str not in self.nodes:
            self.nodes[object_str] = {"id": self._get_id(), "type": "entity", "text": object_str}
            
        edge = {"subject": subject, "relation": relation, "object": object_str}
        self.edges.append(edge)
        
        # 2. Semantic storage (Embed the full phrase to capture meaning)
        phrase = f"{subject} {relation} {object_str}"
        vec = self.embedding_client.embed(phrase)[0]
        
        self.vector_index.add(
            self._get_id(), 
            vec, 
            {"type": "claim", "data": edge, "text": phrase}
        )
        
        # also embed entities separately for soft-matching
        for entity in [subject, object_str]:
            vec_ent = self.embedding_client.embed(entity)[0]
            self.vector_index.add(
                self._get_id(),
                vec_ent,
                {"type": "entity", "data": {"name": entity}, "text": entity}
            )

    def search_soft(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Uses Qwen3 Embeddings + USearch to find contextually relevant graph components."""
        vec = self.embedding_client.embed(text)[0]
        results = self.vector_index.search(vec, top_k=top_k)
        return results
