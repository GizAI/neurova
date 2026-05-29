from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class MeaningAtom:
    name: str
    description: str
    required_slots: tuple[str, ...]

MEANING_ATOMS: Dict[str, MeaningAtom] = {
    "Entity": MeaningAtom("Entity", "Named object, person, concept, or state participant.", ("name",)),
    "Attribute": MeaningAtom("Attribute", "Property or class assigned to an entity.", ("entity", "attribute")),
    "Relation": MeaningAtom("Relation", "Typed relation between two entities or concepts.", ("subject", "relation", "object")),
    "Comparison": MeaningAtom("Comparison", "Ordered or equality relation between two entities.", ("left", "comparator", "right")),
    "Causation": MeaningAtom("Causation", "Cause-effect relation or causal transition.", ("cause", "effect")),
    "TemporalValidity": MeaningAtom("TemporalValidity", "Time-scoped truth of a claim.", ("claim", "time")),
    "Negation": MeaningAtom("Negation", "Negative polarity for a proposition.", ("claim",)),
    "Quantification": MeaningAtom("Quantification", "Universal/existential rule or scope.", ("quantifier", "domain", "predicate")),
    "Exception": MeaningAtom("Exception", "Case that blocks an otherwise applicable rule.", ("subject", "rule")),
    "Condition": MeaningAtom("Condition", "Antecedent condition for a rule or action.", ("condition", "conclusion")),
    "Evidence": MeaningAtom("Evidence", "Grounding quote/source supporting or refuting a claim.", ("source", "quote")),
    "Intention": MeaningAtom("Intention", "Goal-bearing instruction or desired outcome.", ("actor", "goal")),
    "Action": MeaningAtom("Action", "Executable or observed operation in a world state.", ("actor", "action")),
    "Outcome": MeaningAtom("Outcome", "Resulting state after action or cause.", ("action", "outcome")),
    "Uncertainty": MeaningAtom("Uncertainty", "Confidence/ambiguity status for an interpretation.", ("target", "confidence")),
    "Contradiction": MeaningAtom("Contradiction", "Incompatible propositions or claim versions.", ("claim_a", "claim_b")),
    "ProofStep": MeaningAtom("ProofStep", "Verifiable reasoning transition from premises to conclusion.", ("premises", "conclusion")),
}

IR_TO_ATOMS: Dict[str, List[str]] = {
    "ClaimIR": ["Entity", "Relation"],
    "NegatedClaimIR": ["Entity", "Relation", "Negation"],
    "TemporalClaimIR": ["Entity", "Relation", "TemporalValidity"],
    "CausalClaimIR": ["Causation"],
    "ComparisonIR": ["Comparison"],
    "RuleIR": ["Condition", "Relation"],
    "QuantifiedRuleIR": ["Condition", "Quantification"],
    "ExceptionIR": ["Exception"],
    "ContradictionIR": ["Contradiction"],
    "QuestionIR": ["Intention"],
    "ProgramSpecIR": ["Intention", "Action", "Outcome"],
    "ProofIR": ["ProofStep", "Evidence", "Uncertainty"],
}

class MeaningAtomTable:
    """Auditable inventory of the semantic atoms BrainOS is allowed to think with.

    This is intentionally small and explicit: sample efficiency comes from mapping
    many surface utterances onto a compact atom inventory rather than learning an
    unconstrained surface-text distribution.
    """
    def __init__(self):
        self.atoms = MEANING_ATOMS
        self.ir_to_atoms = IR_TO_ATOMS

    def atoms_for_ir(self, ir_type: str) -> List[str]:
        return list(self.ir_to_atoms.get(ir_type, []))

    def validate_inventory(self) -> bool:
        return all(atom in self.atoms for atoms in self.ir_to_atoms.values() for atom in atoms)
