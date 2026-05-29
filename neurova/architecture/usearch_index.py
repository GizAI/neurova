import numpy as np
try:
    import usearch.index
    USEARCH_AVAILABLE = True
except ImportError:
    USEARCH_AVAILABLE = False
from typing import List, Dict, Any, Tuple

class HybridSemanticIndex:
    """Uses USearch for blazing fast vector retrieval while keeping metadata."""
    def __init__(self, dims: int = 3584): # Qwen3-Embedding-4B uses 3584 dims
        self.dims = dims
        self.metadata = {}
        self.fallback_vectors = {}
        if USEARCH_AVAILABLE:
            self.index = usearch.index.Index(ndim=dims, metric='cos', dtype='f32')
        else:
            self.index = None
            self.fallback_vectors = {}

    def add(self, key_id: int, vector: np.ndarray, meta: Dict[str, Any]):
        self.metadata[key_id] = meta
        if self.index:
            self.index.add(key_id, vector.astype(np.float32))
        else:
            self.fallback_vectors[key_id] = vector

    def search(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.index:
            matches = self.index.search(query.astype(np.float32), top_k)
            results = []
            for k, dist in zip(matches.keys, matches.distances):
                sim = 1.0 - float(dist)
                results.append((self.metadata[int(k)], sim))
            return results
        else:
            results = []
            q_norm = np.linalg.norm(query)
            if q_norm == 0:
                return []
            for k, v in self.fallback_vectors.items():
                v_norm = np.linalg.norm(v)
                if v_norm == 0:
                    continue
                sim = np.dot(query, v) / (q_norm * v_norm)
                results.append((self.metadata[k], float(sim)))
            return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
