from __future__ import annotations
from dataclasses import dataclass
import re
from typing import List

@dataclass
class PhraseSegment:
    text: str
    connector: str = ""
    index: int = 0

class SurfaceSegmenter:
    """No-LLM surface segmenter for clause-level semantic fragments.

    It deliberately does not attempt open-domain parsing. It cuts text at high-value
    discourse boundaries so small phrase parsers can compose meaning fragments.
    """
    CONNECTOR_RE = re.compile(r"\s*(,?\s+(?:and|but|while|although|then|because)\s+|그리고|하지만|그러나|때문에|,|;|\.\s+)\s*", re.I)

    def segment(self, text: str) -> List[PhraseSegment]:
        raw = text.strip()
        if not raw:
            return []
        parts: List[PhraseSegment] = []
        start = 0
        index = 0
        last_connector = ""
        for m in self.CONNECTOR_RE.finditer(raw):
            chunk = raw[start:m.start()].strip(" ,.;")
            if chunk:
                parts.append(PhraseSegment(chunk, last_connector, index)); index += 1
            last_connector = m.group(0).strip()
            start = m.end()
        tail = raw[start:].strip(" ,.;")
        if tail:
            parts.append(PhraseSegment(tail, last_connector, index))
        return parts or [PhraseSegment(raw, "", 0)]
