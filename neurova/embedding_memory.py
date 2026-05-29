"""
EmbeddingMemory — GPU-accelerated semantic memory using qwen3-embedding + USearch.

This is the semantic cortex of Neurova:
- Embeds all statements, constructions, and failures
- USearch index for fast nearest-neighbor retrieval
- Failure clustering for automatic schema discovery
- Semantic construction matching for unknown patterns

Usage:
    em = EmbeddingMemory(url="http://ml-dmc8:8081/v1/embeddings")
    em.store("Korea is a peninsular region.", {"type": "statement"})
    similar = em.search("What is Korea?")
"""

import json, os, uuid, pickle, time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# USearch import
try:
    from usearch.index import Index
    HAS_USEARCH = True
except ImportError:
    HAS_USEARCH = False
    Index = None

EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://ml-dmc8:8081/v1/embeddings")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
EMBEDDING_DIM = 2560


# ── Embedding client ──
_embedding_cache = {}

def embed(text: str, url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Get embedding vector. Caches results."""
    if not text.strip():
        return None
    cache_key = f"{url}:{text[:200]}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    
    try:
        import urllib.request
        data = json.dumps({"input": text, "model": model}).encode()
        req = urllib.request.Request(url, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
        vec = np.array(result["data"][0]["embedding"], dtype=np.float32)
        _embedding_cache[cache_key] = vec
        return vec
    except Exception as e:
        return None

def embed_batch(texts: List[str], url: str = EMBEDDING_URL, 
                model: str = EMBEDDING_MODEL) -> List[Optional[np.ndarray]]:
    """Get embeddings for multiple texts."""
    if not texts:
        return []
    try:
        import urllib.request
        data = json.dumps({"input": texts, "model": model}).encode()
        req = urllib.request.Request(url, data=data,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
        vectors = [None] * len(texts)
        for item in result["data"]:
            vectors[item["index"]] = np.array(item["embedding"], dtype=np.float32)
        return vectors
    except Exception as e:
        return [None] * len(texts)


# ── Memory Item ──
@dataclass
class MemItem:
    id: str = ""
    text: str = ""
    category: str = ""  # "statement", "construction", "failure", "query"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


class EmbeddingMemory:
    """
    GPU-accelerated semantic memory with USearch index.
    
    Stores all texts with their embeddings and provides:
    - Semantic search: find most similar texts
    - Failure clustering: group similar failures
    - Construction matching: find construction whose examples match input
    """
    
    def __init__(self, url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL,
                 dim: int = EMBEDDING_DIM):
        self.url = url
        self.model = model
        self.dim = dim
        self.items: List[MemItem] = []
        self.index: Optional[Index] = None
        if HAS_USEARCH:
            self.index = Index(ndim=dim, metric='cos')
        self._next_id = 0
    
    def _ensure_index(self):
        if self.index is None and HAS_USEARCH:
            self.index = Index(ndim=self.dim, metric='cos')
    
    def store(self, text: str, metadata: Dict = None, category: str = "statement") -> Optional[str]:
        """Store text with embedding. Returns item id or None on failure."""
        vec = embed(text, self.url, self.model)
        if vec is None:
            return None
        
        item_id = _uid()
        item = MemItem(
            id=item_id,
            text=text,
            category=category,
            metadata=metadata or {},
            embedding=vec
        )
        self._ensure_index()
        if self.index is not None:
            idx = self._next_id
            self.index.add(idx, vec)
            item.metadata["_index_id"] = idx
            self._next_id += 1
        
        self.items.append(item)
        return item_id
    
    def search(self, query: str, top_k: int = 10, 
               category: Optional[str] = None) -> List[Tuple[float, MemItem]]:
        """Search for most similar items. Returns list of (score, item)."""
        qvec = embed(query, self.url, self.model)
        if qvec is None or self.index is None:
            return []
        
        # Search index
        ids, dists, _ = self.index.search(qvec, top_k * 3)
        
        # Map back to items and filter
        results = []
        id_to_item = {}
        for item in self.items:
            idx = item.metadata.get("_index_id")
            if idx is not None:
                id_to_item[idx] = item
        
        for idx, dist in zip(ids, dists):
            item = id_to_item.get(idx)
            if item is None:
                continue
            if category and item.category != category:
                continue
            # USearch cosine distance: 0=same, 1=orthogonal, 2=opposite
            score = 1.0 - (dist / 2.0)  # Convert to similarity
            results.append((score, item))
        
        results.sort(key=lambda x: -x[0])
        return results[:top_k]
    
    def cluster_failures(self, min_cluster: int = 3) -> List[List[MemItem]]:
        """Group similar failures into clusters for schema discovery."""
        failures = [item for item in self.items if item.category == "failure"]
        if len(failures) < min_cluster:
            return []
        
        clusters = []
        used = set()
        
        for i, f1 in enumerate(failures):
            if i in used:
                continue
            cluster = [f1]
            used.add(i)
            
            for j, f2 in enumerate(failures):
                if j in used:
                    continue
                if f1.embedding is not None and f2.embedding is not None:
                    sim = np.dot(f1.embedding, f2.embedding) / (
                        np.linalg.norm(f1.embedding) * np.linalg.norm(f2.embedding)
                    )
                    if sim > 0.85:  # Very similar
                        cluster.append(f2)
                        used.add(j)
            
            if len(cluster) >= min_cluster:
                clusters.append(cluster)
        
        return clusters
    
    def find_construction_for(self, text: str) -> Optional[Tuple[str, float]]:
        """Find the most similar statement in memory, return its construction type."""
        results = self.search(text, top_k=3, category="statement")
        if not results:
            return None
        best_score, best_item = results[0]
        if best_score < 0.7:
            return None
        ctype = best_item.metadata.get("construction", "")
        return (ctype, best_score) if ctype else None
    
    def items_since(self, timestamp: float) -> List[MemItem]:
        """Get all items stored after the given timestamp."""
        return [item for item in self.items if item.timestamp > timestamp]
    
    def count(self, category: Optional[str] = None) -> int:
        if category:
            return sum(1 for item in self.items if item.category == category)
        return len(self.items)


def _uid():
    return uuid.uuid4().hex[:12]


# ── Vector utility ──
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class EmbeddingService:
    """Lightweight wrapper for embedding + search, no persistence."""
    
    def __init__(self, url: str = EMBEDDING_URL, model: str = EMBEDDING_MODEL):
        self.url = url
        self.model = model
        self.cache = {}
        self.memory = EmbeddingMemory(url, model)
    
    def encode(self, text: str) -> Optional[np.ndarray]:
        if text in self.cache:
            return self.cache[text]
        vec = embed(text, self.url, self.model)
        if vec is not None:
            self.cache[text] = vec
        return vec
    
    def store(self, text: str, meta: dict = None, category: str = "statement"):
        return self.memory.store(text, meta, category)
    
    def search(self, query: str, top_k: int = 10, category: str = None):
        return self.memory.search(query, top_k, category)
    
    def similarity(self, a: str, b: str) -> float:
        va = self.encode(a)
        vb = self.encode(b)
        if va is None or vb is None:
            return 0.0
        return cosine_sim(va, vb)
