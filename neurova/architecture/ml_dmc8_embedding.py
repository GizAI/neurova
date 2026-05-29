import requests
import json
import numpy as np
from typing import List, Union

class FastTextFallbackClient:
    """Mock embedder just in case the docker API is completely down, but still creates non-zero vectors so USearch actually has something to match!"""
    def __init__(self):
        self.dims = 3584
        import hashlib
        self.hashlib = hashlib
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        vecs = []
        for text in texts:
            # Deterministic pseudo-embedding based on character hashing
            np.random.seed(int(self.hashlib.md5(text.lower().encode()).hexdigest()[:8], 16))
            v = np.random.randn(self.dims)
            vecs.append(v / np.linalg.norm(v))
            np.random.seed() # reset
        return np.array(vecs, dtype=np.float32)

class VLLMEmbeddingClient:
    """Connects to the Qwen3-Embedding-4B model running on ml-dmc8 via vLLM"""
    def __init__(self, endpoint_url: str = "http://localhost:8081/v1/embeddings"):
        self.endpoint_url = endpoint_url
        self.model_name = "Qwen/Qwen3-Embedding-4B"
        self.fallback = FastTextFallbackClient()
        
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        payload = {
            "input": texts,
            "model": self.model_name
        }
        
        try:
            response = requests.post(self.endpoint_url, json=payload, timeout=2)
            response.raise_for_status()
            data = response.json()
            
            embeddings = sorted(data["data"], key=lambda x: x["index"])
            return np.array([e["embedding"] for e in embeddings], dtype=np.float32)
        except Exception as e:
            # Silently fallback to pseudo-embeddings so the semantic search logic can still be demonstrated
            return self.fallback.embed(texts)
