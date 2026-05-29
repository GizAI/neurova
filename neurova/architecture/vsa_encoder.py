from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import hashlib

class VectorSymbolicArchitecture:
    """Implements Vector Symbolic Architecture (VSA) operations.
    Uses Holographic Reduced Representations (HRR) using circular convolution.
    Can be accelerated by GPU (PyTorch/CuPy) but falls back to NumPy.
    """
    def __init__(self, dims: int = 1024):
        self.dims = dims
        self.memory: Dict[str, np.ndarray] = {}
        
    def _random_vector(self) -> np.ndarray:
        # Gaussian distributed vectors, normalized
        v = np.random.randn(self.dims)
        return v / np.linalg.norm(v)
        
    def get_or_create(self, symbol: str) -> np.ndarray:
        if symbol not in self.memory:
            # Deterministic seeding based on symbol for reproducibility across runs
            seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
            np.random.seed(seed)
            self.memory[symbol] = self._random_vector()
            np.random.seed() # reset
        return self.memory[symbol]
        
    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Binding operation using circular convolution (Holographic Reduced Representation)."""
        return np.fft.irfft(np.fft.rfft(v1) * np.fft.rfft(v2), n=self.dims)
        
    def unbind(self, bound: np.ndarray, v1: np.ndarray) -> np.ndarray:
        """Unbinding (approximate inverse of bind) using involution."""
        # Involution: reverse the vector except the first element
        v1_inv = np.zeros_like(v1)
        v1_inv[0] = v1[0]
        v1_inv[1:] = v1[1:][::-1]
        return self.bind(bound, v1_inv)
        
    def bundle(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Superposition (bundling) operation."""
        v = v1 + v2
        return v / np.linalg.norm(v)

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def find_nearest(self, query_v: np.ndarray, top_k: int = 5) -> List[tuple[str, float]]:
        results = []
        for sym, v in self.memory.items():
            sim = self.cosine_similarity(query_v, v)
            results.append((sym, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def encode_claim(self, subject: str, relation: str, obj: str) -> np.ndarray:
        """Encodes a full structural claim into a single point in VSA space.
        V_claim = (V_subj ⊗ V_role_subj) ⊕ (V_rel ⊗ V_role_rel) ⊕ (V_obj ⊗ V_role_obj)
        """
        v_subj = self.get_or_create(subject)
        v_rel = self.get_or_create(relation)
        v_obj = self.get_or_create(obj)
        
        r_subj = self.get_or_create("__ROLE_SUBJECT__")
        r_rel = self.get_or_create("__ROLE_RELATION__")
        r_obj = self.get_or_create("__ROLE_OBJECT__")
        
        b1 = self.bind(v_subj, r_subj)
        b2 = self.bind(v_rel, r_rel)
        b3 = self.bind(v_obj, r_obj)
        
        return self.bundle(self.bundle(b1, b2), b3)
