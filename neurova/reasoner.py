from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
from .ir import ClaimIR, CausalClaimIR, ComparisonIR, ProofIR, ProofStepIR, QuestionIR, TemporalClaimIR
from .memory import EvidenceGraphMemory


class ActiveMemoryReasoner:
    """IR-native reasoner with explicit statuses.

    It handles taxonomy/rules, negation, temporal lookup, comparison transitivity,
    causal chain search, and exception-blocked rules. It is still a small symbolic
    reasoner, not a general natural-language reasoner.
    """
    def __init__(self, memory: EvidenceGraphMemory, cognitive_model=None):
        self.memory = memory
        self.model = cognitive_model

    def answer(self, q: QuestionIR) -> ProofIR:
        target = q.target
        if isinstance(target, TemporalClaimIR):
            return self._temporal(target)
        if isinstance(target, CausalClaimIR):
            return self._causal(target)
        if isinstance(target, ComparisonIR):
            return self._comparison(target)
        if isinstance(target, ClaimIR):
            if target.object == "?" and target.relation == "is":
                return self._what(target.subject)
            return self._claim(target)
        return ProofIR(query="unknown", success=False, status="unknown", active_memory_trace=["no_target"])

    def _what(self, subject: str) -> ProofIR:
        claims = self.memory.claims_about(subject)
        fluents = getattr(self.memory, "fluents_about", lambda _s: [])(subject)
        steps = [ProofStepIR(conclusion=c.text(), rule_applied="memory_lookup", verifier_status="verified") for c in claims]
        for f in fluents:
            steps.append(ProofStepIR(conclusion=f"{f['subject']} {f['relation']} {f['value']}", rule_applied="fluent_current_state", verifier_status="verified"))
        return ProofIR(query=f"what is {subject}", success=bool(steps), status="proved" if steps else "unknown", steps=steps, active_memory_trace=["retrieve_broad", f"zoom_in=claims_about:{len(claims)}", f"fluents={len(fluents)}", "verifier=lookup"])

    def _claim(self, target: ClaimIR) -> ProofIR:
        q = f"{target.subject} {target.relation} {target.object}"
        trace = ["retrieve_broad"]
        broad = self.memory.retrieve_broad(q)
        trace += [f"broad={len(broad)}", "zoom_in=claim_versions", "zoom_out=rules+graph"]
        # Current fluent store is checked before historical claim list.
        if target.object == "?":
            fl = getattr(self.memory, "get_fluent", lambda *_: None)(target.subject, target.relation)
            if fl:
                step = ProofStepIR(conclusion=f"{target.subject} {target.relation} {fl['value']}", rule_applied="current_fluent_lookup", verifier_status="verified")
                return ProofIR(query=q, success=True, status="proved", steps=[step], active_memory_trace=trace + ["fluent=current", "verifier=verified"])
        fl = getattr(self.memory, "get_fluent", lambda *_: None)(target.subject, target.relation)
        if fl and fl["value"] == target.object.lower():
            step = ProofStepIR(conclusion=f"{target.subject} {target.relation} {target.object}", rule_applied="current_fluent_lookup", verifier_status="verified")
            return ProofIR(query=q, success=True, status="proved", steps=[step], active_memory_trace=trace + ["fluent=current", "verifier=verified"])
        if target.relation == "is":
            fl_state = getattr(self.memory, "get_fluent", lambda *_: None)(target.subject, "state")
            if fl_state and fl_state["value"] == target.object.lower():
                step = ProofStepIR(conclusion=f"{target.subject} is {target.object}", rule_applied="state_fluent_alias", verifier_status="verified")
                return ProofIR(query=q, success=True, status="proved", steps=[step], active_memory_trace=trace + ["fluent=state_alias", "verifier=verified"])
        key = target.normalized_key()
        if self.memory.contradictions_for_key(key):
            step = ProofStepIR(conclusion=q, premises=["positive and negative claim versions exist"], rule_applied="contradiction_resolver", verifier_status="inconsistent")
            return ProofIR(query=q, success=False, status="inconsistent", steps=[step], active_memory_trace=trace + ["verifier=inconsistent"])
        neg = self.memory.find_claim(target.subject, target.relation, target.object, "negative")
        if neg:
            step = ProofStepIR(conclusion=q, premises=[f"negative claim exists: {q}"], rule_applied="direct_refutation", verifier_status="verified")
            return ProofIR(query=q, success=False, status="refuted", steps=[step], active_memory_trace=trace + ["verifier=refuted"])
        result = self._prove_relation(target.subject.lower(), target.relation.lower(), target.object.lower(), depth=0, seen=set())
        if result[0] == "proved":
            steps = [ProofStepIR(conclusion=a, premises=b, rule_applied=c, verifier_status="verified") for a, b, c in result[1]]
            return ProofIR(query=q, success=True, status="proved", steps=steps, active_memory_trace=trace + ["verifier=verified"])
        if result[0] == "blocked_by_exception":
            steps = [ProofStepIR(conclusion=a, premises=b, rule_applied=c, verifier_status="blocked") for a, b, c in result[1]]
            return ProofIR(query=q, success=False, status="blocked_by_exception", steps=steps, active_memory_trace=trace + ["verifier=blocked_by_exception"])
        return ProofIR(query=q, success=False, status="unknown", active_memory_trace=trace + ["verifier=unknown"])

    def _prove_relation(self, subj: str, rel: str, obj: str, depth: int, seen: Set[Tuple[str, str, str]]):
        if depth > 6:
            return "unknown", []
        key = (subj, rel, obj)
        if key in seen:
            return "unknown", []
        seen.add(key)
        pos = self.memory.find_claim(subj, rel, obj, "positive")
        if pos:
            return "proved", [(f"{subj} {rel} {obj}", [], "known")]
        neg = self.memory.find_claim(subj, rel, obj, "negative")
        if neg:
            return "refuted", [(f"{subj} {rel} {obj}", ["negative claim exists"], "direct_refutation")]
        direct_exc = getattr(self.memory, "direct_exception_blocks", lambda *_: None)(subj, rel, obj)
        if direct_exc:
            return "blocked_by_exception", [(f"{subj} {rel} {obj}", [f"exception: {direct_exc['exception_subject']}"], "direct_exception_block")]

        # Taxonomy transitivity for is-relations.
        if rel == "is":
            rows = self.memory.claim_versions(relation="is", polarity="positive")
            edges: Dict[str, Set[str]] = {}
            for r in rows:
                edges.setdefault(r["subject"], set()).add(r["object"])
            path = self._bfs_edges(edges, subj, obj, "is")
            if path:
                steps = []
                for i, (a, b) in enumerate(path):
                    steps.append((f"{a} is {b}", [] if i == 0 else [f"{path[i-1][0]} is {path[i-1][1]}"], "taxonomy_transitivity" if i else "known"))
                return "proved", steps

        # Rule-based derivation for any relation.
        for rule in self.memory.rules():
            if rule["conclusion_relation"] != rel or rule["conclusion_object"] != obj:
                continue
            pre_status, pre_steps = self._prove_relation(subj, rule["condition_relation"], rule["condition_object"], depth + 1, seen.copy())
            if pre_status == "proved":
                exc = self.memory.exception_blocks(subj, rule)
                if exc:
                    blocked = pre_steps + [(f"{subj} {rel} {obj}", [f"exception: {exc['exception_subject']}"], "exception_blocks_rule")]
                    return "blocked_by_exception", blocked
                return "proved", pre_steps + [(f"{subj} {rel} {obj}", [f"{subj} {rule['condition_relation']} {rule['condition_object']}"], f"rule {rule['signature']}")]
        return "unknown", []

    @staticmethod
    def _bfs_edges(edges: Dict[str, Set[str]], src: str, dst: str, label: str):
        q = [(src, [])]
        seen = {src}
        while q:
            n, path = q.pop(0)
            for nx in edges.get(n, set()):
                new_path = path + [(n, nx)]
                if nx == dst:
                    return new_path
                if nx not in seen:
                    seen.add(nx)
                    q.append((nx, new_path))
        return None

    def _temporal(self, target: TemporalClaimIR) -> ProofIR:
        t = target.valid_during or target.time_expr or target.valid_from
        trace = ["temporal_retrieve", f"time={t}"]
        if target.subject == "?":
            rows = [r for r in self.memory.claim_versions(relation=target.relation, polarity="positive", at=t) if r["object"] == target.object.lower()]
            clean_rows = []
            conflicts = []
            for r in rows:
                neg = self.memory.find_claim(r["subject"], target.relation, target.object, "negative", at=t)
                if neg:
                    conflicts.append((r, neg))
                else:
                    clean_rows.append(r)
            if conflicts:
                step = ProofStepIR(conclusion=f"who {target.relation} {target.object} at {t}", premises=[f"{c[0]['subject']} is both positive and negative at {t}" for c in conflicts], rule_applied="temporal_contradiction_resolver", verifier_status="inconsistent")
                return ProofIR(query=f"who {target.relation} {target.object} at {t}", success=False, status="inconsistent", steps=[step], active_memory_trace=trace + [f"conflicts={len(conflicts)}", "verifier=inconsistent"])
            if clean_rows:
                steps = [ProofStepIR(conclusion=f"{r['subject']} {target.relation} {target.object} @ {r['valid_from'] or t}", rule_applied="temporal_validity", verifier_status="verified") for r in clean_rows]
                return ProofIR(query=f"who {target.relation} {target.object} at {t}", success=True, status="proved", steps=steps, active_memory_trace=trace + [f"matches={len(clean_rows)}", "verifier=verified"])
            return ProofIR(query=f"who {target.relation} {target.object} at {t}", success=False, status="unknown", active_memory_trace=trace + ["matches=0"])
        if target.object == "?":
            rows = [r for r in self.memory.claim_versions(relation=target.relation, polarity="positive", at=t) if r["subject"] == target.subject.lower()]
            if rows:
                steps = [ProofStepIR(conclusion=f"{target.subject} {target.relation} {r['object']} @ {r['valid_from']}", rule_applied="temporal_validity", verifier_status="verified") for r in rows]
                return ProofIR(query=f"what {target.relation} {target.subject} at {t}", success=True, status="proved", steps=steps, active_memory_trace=trace + [f"matches={len(rows)}", "verifier=verified"])
        neg = self.memory.find_claim(target.subject, target.relation, target.object, "negative", at=t)
        row = self.memory.find_claim(target.subject, target.relation, target.object, "positive", at=t)
        if row and neg:
            step = ProofStepIR(conclusion=f"{target.subject} {target.relation} {target.object} @ {t}", premises=["positive and negative temporal claims overlap"], rule_applied="temporal_contradiction_resolver", verifier_status="inconsistent")
            return ProofIR(query=target.text(), success=False, status="inconsistent", steps=[step], active_memory_trace=trace + ["verifier=inconsistent"])
        if neg:
            step = ProofStepIR(conclusion=f"{target.subject} {target.relation} {target.object} @ {t}", premises=["negative temporal claim exists"], rule_applied="temporal_refutation", verifier_status="verified")
            return ProofIR(query=target.text(), success=False, status="refuted", steps=[step], active_memory_trace=trace + ["verifier=refuted"])
        if row:
            step = ProofStepIR(conclusion=f"{target.subject} {target.relation} {target.object} @ {row['valid_from'] or t}", rule_applied="temporal_validity", verifier_status="verified")
            return ProofIR(query=target.text(), success=True, status="proved", steps=[step], active_memory_trace=trace + ["verifier=verified"])
        return ProofIR(query=target.text(), success=False, status="unknown", active_memory_trace=trace + ["verifier=unknown"])

    def _causal(self, target: CausalClaimIR) -> ProofIR:
        graph: Dict[str, Set[str]] = {}
        for r in self.memory.all_transitions():
            graph.setdefault(r["cause"], set()).add(r["effect"])
        if self.model:
            self.model.choose_proof_operator("causal", {})
            self.model.predict_world_transition(target.cause, list(graph.get(target.cause.lower(), set())))
        if target.polarity == "negative":
            q = f"{target.cause} does not cause {target.effect}"
            trace = ["causal_negation_check"]
            path = self._bfs_edges(graph, target.cause.lower(), target.effect.lower(), "causes")
            if path:
                return ProofIR(query=q, success=False, status="refuted", steps=[ProofStepIR(conclusion=q, premises=[f"positive causal chain exists"], rule_applied="causal_refutation", verifier_status="verified")], active_memory_trace=trace + ["positive_path_exists"])
            neg = self.memory.find_claim(target.cause.lower(), "causes", target.effect.lower(), "negative")
            if neg:
                return ProofIR(query=q, success=True, status="proved", steps=[ProofStepIR(conclusion=q, premises=[f"stored negation: {target.cause} does not cause {target.effect}"], rule_applied="negated_causal_memory", verifier_status="verified")], active_memory_trace=trace + ["negation_stored"])
            return ProofIR(query=q, success=False, status="unknown", active_memory_trace=trace + ["verifier=unknown"])
        if target.effect == "?":
            effects = sorted(graph.get(target.cause.lower(), set()))
            steps = [ProofStepIR(conclusion=f"{target.cause} causes {e}", rule_applied="causal_memory", verifier_status="verified") for e in effects]
            return ProofIR(query=f"what happens after {target.cause}", success=bool(effects), status="proved" if effects else "unknown", steps=steps, active_memory_trace=["causal_retrieve", f"direct_effects={len(effects)}", "verifier=verified"])
        path = self._bfs_edges(graph, target.cause.lower(), target.effect.lower(), "causes")
        if path:
            steps = [ProofStepIR(conclusion=f"{a} causes {b}", premises=[] if i == 0 else [f"{path[i-1][0]} causes {path[i-1][1]}"], rule_applied="causal_chain" if i else "causal_memory", verifier_status="verified") for i, (a, b) in enumerate(path)]
            return ProofIR(query=target.text(), success=True, status="proved", steps=steps, active_memory_trace=["causal_graph_search", f"path_len={len(path)}", "verifier=verified"])
        return ProofIR(query=target.text(), success=False, status="unknown", active_memory_trace=["causal_graph_search", "verifier=unknown"])

    def _comparison(self, target: ComparisonIR) -> ProofIR:
        graph: Dict[str, Set[str]] = {}
        for r in self.memory.all_comparisons():
            comp = r["comparator"]
            left, right = r["left_value"], r["right_value"]
            graph.setdefault(f"{comp}:{left}", set()).add(right)
            # Store inverse relation so less_than can use greater_than facts and vice versa.
            inv = "less_than" if comp == "greater_than" else "greater_than" if comp == "less_than" else None
            if inv:
                graph.setdefault(f"{inv}:{right}", set()).add(left)
        if self.model:
            self.model.choose_proof_operator("comparison", {})
        def bfs(left: str, comp: str, right: str):
            q = [(left, [])]
            seen = {left}
            while q:
                n, path = q.pop(0)
                for nx in graph.get(f"{comp}:{n}", set()):
                    edge = (n, comp, nx)
                    if nx == right:
                        return path + [edge]
                    if nx not in seen:
                        seen.add(nx)
                        q.append((nx, path + [edge]))
            return None
        path = bfs(target.left.lower(), target.comparator, target.right.lower())
        if path:
            steps = [ProofStepIR(conclusion=f"{a} {comp} {b}", premises=[] if i == 0 else [f"{path[i-1][0]} {path[i-1][1]} {path[i-1][2]}"], rule_applied="comparison_transitivity" if i else "comparison_memory", verifier_status="verified") for i, (a, comp, b) in enumerate(path)]
            return ProofIR(query=target.text(), success=True, status="proved", steps=steps, active_memory_trace=["comparison_graph_search", f"path_len={len(path)}", "verifier=verified"])
        return ProofIR(query=target.text(), success=False, status="unknown", active_memory_trace=["comparison_graph_search", "verifier=unknown"])
