"""GPU Embedding Service — connects to qwen3-embedding on ml-dmc8:8081.

Usage:
    from neurova.embeddings import embed, EmbeddingService
    vec = embed("some text")  # returns 2560-dim vector
    service = EmbeddingService()
    sim = service.similarity("query", "candidate")
"""

import json, os, sys
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://ml-dmc8:8081/v1/embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBEDDING_DIM = 2560

def embed(text: str, url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL) -> Optional[List[float]]:
    """Get embedding vector for text. Returns None on failure."""
    if not text.strip():
        return None
    try:
        data = json.dumps({"input": text, "model": model}).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        return result["data"][0]["embedding"]
    except Exception as e:
        return None

def embed_batch(texts: List[str], url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL) -> List[Optional[List[float]]]:
    """Get embeddings for multiple texts. Returns list of vectors or None for failures."""
    if not texts:
        return []
    try:
        data = json.dumps({"input": texts, "model": model}).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        vectors = [None] * len(texts)
        for item in result["data"]:
            vectors[item["index"]] = item["embedding"]
        return vectors
    except Exception as e:
        return [None] * len(texts)

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class EmbeddingService:
    """GPU embedding service with caching."""
    
    def __init__(self, url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL):
        self.url = url
        self.model = model
        self._cache = {}
    
    def encode(self, text: str) -> Optional[List[float]]:
        if text in self._cache:
            return self._cache[text]
        vec = embed(text, self.url, self.model)
        if vec:
            self._cache[text] = vec
        return vec
    
    def similarity(self, a: str, b: str) -> float:
        va = self.encode(a)
        vb = self.encode(b)
        if va is None or vb is None:
            return 0.0
        return cosine_similarity(va, vb)
    
    def most_similar(self, query: str, candidates: List[str], top_k: int = 5):
        """Find top-k most similar candidates to query."""
        vq = self.encode(query)
        if vq is None:
            return []
        scored = []
        for c in candidates:
            vc = self.encode(c)
            if vc:
                scored.append((cosine_similarity(vq, vc), c))
        scored.sort(key=lambda x: -x[0])
        return [(s, c) for s, c in scored[:top_k]]
