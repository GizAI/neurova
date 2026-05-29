from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Protocol
from .ir import ResearchTaskIR

@dataclass
class SourceDoc:
    id: str
    title: str
    text: str
    path: str = ""

class SourceConnector(Protocol):
    def search(self, query: str, limit: int = 5) -> List[SourceDoc]: ...

class LocalTextSourceConnector:
    def __init__(self, root: Path): self.root=Path(root)
    def search(self, query: str, limit: int=5) -> List[SourceDoc]:
        out=[]; q=query.lower()
        if not self.root.exists(): return out
        terms=[t for t in q.split() if len(t)>3]
        for p in self.root.rglob("*.txt"):
            text=p.read_text(encoding="utf-8", errors="replace")
            low=text.lower()
            if q in low or any(t in low for t in terms):
                out.append(SourceDoc(str(p), p.name, text[:4000], str(p)))
            if len(out)>=limit: break
        return out

class ResearchEngine:
    def __init__(self, connectors=None): self.connectors=connectors or []
    def answer(self, task: ResearchTaskIR) -> str:
        docs=[]
        for c in self.connectors: docs.extend(c.search(task.question))
        if not docs:
            return "No relevant sources found. I will not fabricate research results. Provide sources or enable web/source retrieval."
        lines=["Research synthesis from connected sources:"]
        for d in docs[:5]:
            lines.append(f"- {d.title}: {' '.join(d.text.split()[:40])}")
        return "\n".join(lines)
