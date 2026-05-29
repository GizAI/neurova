from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SemanticTestTimeAdapter:
    """Session-local semantic adaptation. It never mutates core parser weights directly."""
    aliases: Dict[str, str] = field(default_factory=dict)
    corrections: Dict[str, str] = field(default_factory=dict)

    def observe_alias(self, surface: str, canonical: str) -> None:
        self.aliases[surface.lower().strip()] = canonical.lower().strip()

    def adapt_text(self, text: str) -> str:
        out = text
        for surface, canonical in sorted(self.aliases.items(), key=lambda kv: len(kv[0]), reverse=True):
            out = out.replace(surface, canonical)
        return out

    def propose_patch(self, failed_text: str, target_hint: str) -> dict:
        key = failed_text.strip().lower()
        self.corrections[key] = target_hint
        return {"kind": "semantic_adapter_patch", "text": failed_text, "hint": target_hint, "status": "session_local"}
