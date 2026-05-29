from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class MeaningOperation:
    name: str
    description: str
    input_atoms: tuple[str, ...]
    output_atom: str


class MeaningAtomCalculus:
    """Explicit operations over BrainOS meaning atoms.

    This is the non-LLM reasoning substrate: language is reduced to atoms, and
    cognition uses auditable operators such as compose, negate, time-scope,
    causal-chain, comparison-chain, exception-block, support/refute, and derive.
    """
    def __init__(self):
        self.operations: Dict[str, MeaningOperation] = {
            "compose": MeaningOperation("compose", "Combine atom fragments into a composite IR graph.", ("Relation", "Relation"), "ProofStep"),
            "negate": MeaningOperation("negate", "Attach negative polarity to a proposition.", ("Relation",), "Negation"),
            "time_scope": MeaningOperation("time_scope", "Restrict a proposition to a validity interval.", ("Relation", "TemporalValidity"), "TemporalValidity"),
            "cause_chain": MeaningOperation("cause_chain", "Infer indirect causation over causal edges.", ("Causation", "Causation"), "Causation"),
            "compare_chain": MeaningOperation("compare_chain", "Infer transitive ordered comparisons.", ("Comparison", "Comparison"), "Comparison"),
            "exception_block": MeaningOperation("exception_block", "Block a rule conclusion for an exception case.", ("Exception", "Condition"), "Uncertainty"),
            "support": MeaningOperation("support", "Connect evidence to a claim/proof.", ("Evidence", "Relation"), "ProofStep"),
            "refute": MeaningOperation("refute", "Connect evidence or opposite polarity to refutation.", ("Evidence", "Contradiction"), "ProofStep"),
            "derive": MeaningOperation("derive", "Create a proof step from premises and a rule.", ("ProofStep", "Condition"), "ProofStep"),
            "generalize": MeaningOperation("generalize", "Promote repeated claim patterns into a rule candidate.", ("Relation",), "Quantification"),
            "specialize": MeaningOperation("specialize", "Instantiate a rule for a concrete subject.", ("Quantification", "Entity"), "Relation"),
        }

    def has(self, op: str) -> bool:
        return op in self.operations

    def validate(self) -> bool:
        return all(o.name == k and o.output_atom for k, o in self.operations.items())

    def apply(self, op: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if op not in self.operations:
            return {"ok": False, "error": f"unknown operation: {op}"}
        return {"ok": True, "operation": op, "result_atom": self.operations[op].output_atom, "payload": payload}
