from __future__ import annotations

"""V36 language-acquisition audit.

This audit is intentionally not a smoke test for individual regexes.  It checks
whether the system can build and query a situation/world model from a short text
stream with coreference, spatial frames, temporal state, event-driven fluent
state, and embedding-backed recall.  The goal is to verify that the source of
truth is SituationFrame/ObjectState rather than flat triples.
"""

from dataclasses import asdict
from typing import Any, Dict, List
from .acquisition_world import LanguageAcquisitionSubstrate
from .agent import FinalCognitiveOS


class V36LanguageAcquisitionAudit:
    def __init__(self):
        self.substrate = LanguageAcquisitionSubstrate()

    def run(self) -> Dict[str, Any]:
        tests: List[Dict[str, Any]] = []
        # 1. Geographical situation model: coreference, spatial separation,
        # parallel relations, and temporal state must not collapse into triples.
        geo_text = (
            "Korea is a peninsula in East Asia. "
            "It is separated from Japan to the southeast by the Korea Strait. "
            "China to the north and Russia to the northeast. "
            "The region became independent in 1948."
        )
        frames = self.substrate.observe(geo_text)
        tests.append(self._check("geo_frames_created", len(frames) >= 4, len(frames)))
        tests.append(self._check("separator_query", self.substrate.ask("What separates Korea from Japan?") == "Korea Strait", self.substrate.ask("What separates Korea from Japan?")))
        tests.append(self._check("direction_query", self.substrate.ask("What direction is Japan relative to Korea?") == "southeast", self.substrate.ask("What direction is Japan relative to Korea?")))
        tests.append(self._check("parallel_north_query", self.substrate.ask("Which country lies to the north of Korea?") == "china", self.substrate.ask("Which country lies to the north of Korea?")))
        tests.append(self._check("temporal_state_not_location", self.substrate.ask("When did Korea become independent?") == "1948", self.substrate.ask("When did Korea become independent?")))

        report = self.substrate.state_report()
        triples = report["triples"]
        bad_year_location = any(t[0] == "korea" and t[1] in {"location", "located_at"} and t[2] == "1948" for t in triples)
        tests.append(self._check("year_not_misread_as_location", not bad_year_location, triples))
        frame_types = [f["frame_type"] for f in report["frames"]]
        tests.append(self._check("event_frame_source_of_truth", "spatial_separation" in frame_types and "temporal_state" in frame_types, frame_types))

        # 2. Object-centric state-space update: current state should change after a move.
        state = LanguageAcquisitionSubstrate()
        state.observe("John is in the kitchen. John went to the garden.")
        tests.append(self._check("state_space_current_location", state.ask("Where is John?") == "garden", state.ask("Where is John?")))
        state.observe("Mira picked up the cup. She put it on the shelf.")
        tests.append(self._check("coref_event_state_update", state.ask("Where is the cup?") == "shelf", state.ask("Where is the cup?")))

        # 3. Embedding-backed recall should retrieve related event frame even when query wording differs.
        hits = self.substrate.world.embeddings.search("separator between Korea and Japan", top_k=3)
        tests.append(self._check("embedding_association_recall", any(h[2].get("frame_type") == "spatial_separation" for h in hits), [(h[2], h[3]) for h in hits]))

        # 4. Integration with FinalCognitiveOS: situation updates/queries should go through observe().
        os = FinalCognitiveOS(root=None)
        os.observe("Korea is a peninsula in East Asia.")
        os.observe("It is separated from Japan to the southeast by the Korea Strait.")
        ans = os.observe("What separates Korea from Japan?").response
        tests.append(self._check("agent_integration_situation_query", "Korea Strait" in ans, ans))

        passed = sum(1 for t in tests if t["passed"])
        return {
            "version": "v36_language_acquisition_substrate",
            "passed": passed,
            "total": len(tests),
            "accuracy": passed / len(tests),
            "tests": tests,
            "state_report": report,
            "honesty_note": "This proves a controlled situation-model language-acquisition substrate, not human-level open-domain intelligence.",
        }

    def _check(self, name: str, passed: bool, observed: Any) -> Dict[str, Any]:
        return {"name": name, "passed": bool(passed), "observed": observed}
