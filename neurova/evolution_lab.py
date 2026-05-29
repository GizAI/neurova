"""Evolution Lab -- BrainOS's engine for continuous self-improvement.

Runs periodically (every N episodes or on explicit sleep/consolidation):
1. Collect failure episodes.
2. Cluster failures by embedding similarity.
3. Estimate error type per cluster.
4. Request schema candidates from LLM Teacher Society.
5. Request counterexamples from LLM Critic.
6. Verify candidates with BrainOS's own runtime.
7. Promote/demote schemas through regression gate.
8. Update semantic encoder with verified data.

The LLM is NEVER the answer generator -- it is the schema proposer.
BrainOS verifier has final say on what enters memory.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import time

from .schema_learning import (
    SchemaLearningSubstrate,
    SchemaCandidate,
    SchemaTest,
    FailureToSchemaCompiler,
    SchemaDemotionEngine,
)
from .semantic_encoder import DeepSemanticEncoder, SemanticMemoryIndex


@dataclass
class EvolutionReport:
    """Result of a single evolution cycle."""
    timestamp: float = field(default_factory=time.time)
    failure_clusters: int = 0
    candidates_proposed: int = 0
    candidates_verified: int = 0
    candidates_promoted: int = 0
    candidates_rejected: int = 0
    schemas_demoted: int = 0
    encoder_updates: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class EvolutionLab:
    """Orchestrates the full failure-to-schema evolution cycle.

    Usage:
        lab = EvolutionLab(substrate, encoder)
        report = lab.run_cycle(
            runtime_fn=lambda text: brain.observe(text).response,
            teacher_fn=optional_llm_teacher,
        )
    """

    def __init__(
        self,
        substrate: SchemaLearningSubstrate,
        encoder: Optional[DeepSemanticEncoder] = None,
    ):
        self.substrate = substrate
        self.encoder = encoder or DeepSemanticEncoder()
        self.index = SemanticMemoryIndex(self.encoder)
        self.compiler = FailureToSchemaCompiler(substrate)
        self.demotion = SchemaDemotionEngine(substrate.memory)
        self._history: List[EvolutionReport] = []

    def run_cycle(
        self,
        runtime_fn: Optional[Callable[[str], str]] = None,
        teacher_fn: Optional[Callable[[str, List[str]], Optional[Dict[str, Any]]]] = None,
        critic_fn: Optional[Callable[[Dict[str, Any]], List[str]]] = None,
        min_cluster_size: int = 2,
    ) -> EvolutionReport:
        """Execute one full evolution cycle.

        Args:
            runtime_fn: BrainOS observe function for verification.
            teacher_fn: LLM teacher that proposes schema candidates from error cluster.
            critic_fn: LLM critic that generates counterexamples for candidates.
        """
        report = EvolutionReport()

        # 1. Compile recurring failures into schema candidates.
        compile_result = self.compiler.compile_from_errors(min_cluster_size)
        report.failure_clusters = compile_result["error_clusters"]
        report.candidates_proposed = len(compile_result["candidates_generated"])

        # 2. If teacher_fn is available, ask for refined candidates.
        if teacher_fn:
            for cand_info in compile_result["candidates_generated"]:
                schema_row = self.substrate.memory.conn.execute(
                    "SELECT schema_json FROM schema_candidates WHERE id=?",
                    (cand_info["schema_id"],),
                ).fetchone()
                if not schema_row:
                    continue
                schema = json.loads(schema_row["schema_json"])
                sample_texts = schema.get("sample_texts", [])
                if not sample_texts:
                    continue

                teacher_proposal = teacher_fn(cand_info["error_type"], sample_texts)
                if teacher_proposal:
                    # Teacher proposed a refined schema; update the candidate.
                    self.substrate.memory.conn.execute(
                        "UPDATE schema_candidates SET schema_json=?, confidence=? WHERE id=?",
                        (json.dumps(teacher_proposal, ensure_ascii=False), 0.55, cand_info["schema_id"]),
                    )
                    self.substrate.memory.conn.commit()

        # 3. If critic_fn is available, generate counterexamples.
        if critic_fn:
            for cand_info in compile_result["candidates_generated"]:
                schema_row = self.substrate.memory.conn.execute(
                    "SELECT schema_json FROM schema_candidates WHERE id=?",
                    (cand_info["schema_id"],),
                ).fetchone()
                if not schema_row:
                    continue
                schema = json.loads(schema_row["schema_json"])
                counterexamples = critic_fn(schema)
                if counterexamples:
                    tests = [
                        SchemaTest(ce, {"should_not_match": True}, is_counterexample=True)
                        for ce in counterexamples
                    ]
                    self.substrate.memory.add_schema_tests(cand_info["schema_id"], tests)

        # 4. Verify candidates with runtime if available.
        if runtime_fn:
            for cand_info in compile_result["candidates_generated"]:
                tests = self.substrate.memory.conn.execute(
                    "SELECT * FROM schema_tests WHERE schema_id=?",
                    (cand_info["schema_id"],),
                ).fetchall()
                passed_all = True
                for test in tests:
                    try:
                        result = runtime_fn(test["input_text"])
                        is_counter = bool(test["is_counterexample"])
                        matched = bool(result and "fallback" not in result.lower())
                        if is_counter and matched:
                            passed_all = False
                        elif not is_counter and not matched:
                            passed_all = False
                    except Exception:
                        passed_all = False
                if passed_all:
                    report.candidates_verified += 1

        # 5. Re-run consolidation to promote/reject.
        consol = self.substrate.consolidate()
        report.candidates_promoted = len(consol.get("promoted", []))
        report.schemas_demoted = len(consol.get("demotions", []))
        report.candidates_rejected = report.candidates_proposed - report.candidates_promoted

        # 6. Update semantic index with verified schemas for retrieval.
        verified_schemas = self.substrate.memory.schemas(include_experimental=False)
        for sid, stype, schema, conf, status in verified_schemas:
            forms = schema.get("forms", []) or [schema.get("form", "")]
            text = " ".join(forms)
            if text.strip():
                self.index.add(sid, "schema", text, {"type": stype, "confidence": conf})
                report.encoder_updates += 1

        report.details = {
            "compile_result": compile_result,
            "consolidation": consol,
        }
        self._history.append(report)
        return report

    @property
    def history(self) -> List[EvolutionReport]:
        return self._history
