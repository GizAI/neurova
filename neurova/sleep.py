from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import json
from typing import Dict, List
from .memory import EvidenceGraphMemory


@dataclass
class SleepReplayReport:
    trajectories: int
    strategies: Dict[str, int]
    promotion_candidates: int
    actions_logged: int


class SleepReplayConsolidator:
    """Offline consolidation pass: failures become strategies/tests, not hallucinated knowledge."""
    def __init__(self, memory: EvidenceGraphMemory):
        self.memory = memory

    def run(self) -> SleepReplayReport:
        rows = self.memory.conn.execute("SELECT * FROM trajectories ORDER BY created_at DESC LIMIT 200").fetchall()
        failures = Counter(r["failure_type"] or r["status"] for r in rows if r["status"] != "success" or r["failure_type"])
        promoted = 0
        for failure_type, count in failures.items():
            lesson = f"sleep strategy: add regression and parser/memory skill for {failure_type} seen {count} times"
            self.memory.add_promotion_candidate("sleep_strategy", {"failure_type": failure_type, "count": count, "lesson": lesson}, status="candidate", score=min(1.0, 0.2 * count))
            self.memory.add_learned_strategy(failure_type, lesson, count)
            self.memory.log_action("SLEEP_STRATEGY", failure_type, {"count": count, "lesson": lesson}, min(1.0, 0.2 * count))
            promoted += 1
        # Consolidation marker even when no failures exist.
        self.memory.log_action("SLEEP_REPLAY", "global", {"trajectories": len(rows), "failure_types": dict(failures)}, 0.1 * len(rows))
        report = SleepReplayReport(len(rows), dict(failures), promoted, 1 + promoted)
        self.memory.record_sleep_report(report.__dict__)
        return report


# ===========================================================================
# V36 Evolution: Full Sleep Consolidation Cycle
# ===========================================================================

class SleepConsolidationCycle:
    """Full nightly-style consolidation cycle.

    Performs:
    1. Replay: review recent episodes.
    2. Merge: consolidate repeated patterns.
    3. Forget: demote/expire weak schemas.
    4. Promote: upgrade strong experimental schemas.
    5. Update: refresh semantic encoder with verified data.
    6. Hygiene: clean stale sourced claims.
    """

    def __init__(self, memory: EvidenceGraphMemory, schema_substrate=None, predictive_loop=None, internet_world=None):
        self.memory = memory
        self.schema_substrate = schema_substrate
        self.predictive_loop = predictive_loop
        self.internet_world = internet_world
        self.legacy_consolidator = SleepReplayConsolidator(memory)

    def run_full_cycle(self) -> dict:
        report = {"phases": {}}

        # Phase 1: Legacy replay (trajectories -> strategies).
        legacy = self.legacy_consolidator.run()
        report["phases"]["replay"] = {
            "trajectories": legacy.trajectories,
            "strategies": legacy.strategies,
        }

        # Phase 2: Predictive loop consolidation (error clusters -> skills).
        if self.predictive_loop:
            pred_consol = self.predictive_loop.consolidate()
            report["phases"]["predictive_consolidation"] = pred_consol

        # Phase 3: Schema consolidation (promote/demote/compile failures).
        if self.schema_substrate:
            schema_consol = self.schema_substrate.consolidate()
            report["phases"]["schema_consolidation"] = schema_consol

        # Phase 4: Internet world hygiene (expire stale claims).
        if self.internet_world:
            expired = self.internet_world.expire_stale()
            report["phases"]["internet_hygiene"] = {"expired_claims": expired}

        # Phase 5: Memory stats snapshot.
        report["memory_stats"] = self.memory.stats()

        # Log the full cycle.
        self.memory.record_sleep_report(report)
        return report
