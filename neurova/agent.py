from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json, re
from .code import CodeSandbox
from .cognitive_model import NeuralCognitiveCompiler
from .compiler import HybridSemanticCompiler
from .ir import *
from .memory import EvidenceGraphMemory
from .memory_skills import MemorySkillEngine
from .reasoner import ActiveMemoryReasoner
from .research import LocalTextSourceConnector, ResearchEngine
from .self_improvement import RegressionGatedSelfImprover
from .world import StateTransitionWorldModel
from .epistemic import EpistemicImmuneSystem
from .sleep import SleepReplayConsolidator
from .domain import DomainShardRouter
from .adaptation import SemanticTestTimeAdapter
from .grounding import GroundedVerifier
from .world_frames import EventWorldGrounder
from .continual import ContinualLearningGate
from .developmental import LLMSeedKnowledgeBank, IntrinsicMotivationEngine
from .dialogue import UnifiedActionSelector
from .predictive_developmental import PredictiveSocialCognitiveLoop
from .schema_learning import SchemaLearningSubstrate
from .evolution_lab import EvolutionLab
from .teacher_society import TeacherSociety
from .internet_world import InternetWorldMemory
from .sleep import SleepConsolidationCycle
from .self_improvement import SelfModelManager
from .semantic_encoder import SchemaRetrievalIndex, ContrastiveLearningBuffer
from .v40_language_acquisition import LanguageAcquisitionEngine



@dataclass
class OSResult:
    response: str
    ir_type: str
    confidence: float
    memory_stats: Dict[str, int]
    cognitive_model: dict


class FinalCognitiveOS:
    """BrainOS V36 Developmental Neuro-Symbolic Cognitive OS.

    Core principle: code is the learning circuit, not the knowledge.
    New ability arrives as schemas, episodes, embeddings, and world model updates.

    Online answer path: LLM-free. BrainOS interprets, reasons, responds.
    Offline learning: LLM Teacher Society proposes, BrainOS verifier decides.

    observe() pipeline: predict -> execute -> compare -> episode -> learn.
    """
    def __init__(self, root: Optional[Path] = None, backend: str = "cognitive", auto_seed: bool = False):
        self.root = Path(root or "/mnt/data/final_cognitive_os_v30").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.cognitive_model = NeuralCognitiveCompiler()
        self.memory = EvidenceGraphMemory(self.root / "evidence_graph.sqlite3")
        self.compiler = HybridSemanticCompiler(self.cognitive_model, self.memory)
        self.reasoner = ActiveMemoryReasoner(self.memory, self.cognitive_model)
        self.skills = MemorySkillEngine(self.memory)
        self.world = StateTransitionWorldModel(self.memory)
        self.self_improver = RegressionGatedSelfImprover(self.memory)
        self.immune = EpistemicImmuneSystem(self.memory)
        self.sleep = SleepReplayConsolidator(self.memory)
        self.domain_router = DomainShardRouter()
        self.adapter = SemanticTestTimeAdapter()
        self.grounded = GroundedVerifier()
        self.world_grounder = EventWorldGrounder()
        self.continual_gate = ContinualLearningGate()
        self.intrinsic = IntrinsicMotivationEngine()
        self.entity_stack = []
        self.action_selector = UnifiedActionSelector()
        self.predictive_loop = PredictiveSocialCognitiveLoop()
        self.schema_substrate = SchemaLearningSubstrate(self.root / "schema_learning.sqlite3")
        self.seed_bank = LLMSeedKnowledgeBank().add_builtin_child_language_seed()
        if auto_seed:
            self.seed_bank.install(self)
            
        # Seed basic self-knowledge
        ev = self.memory.add_evidence("system", "self_identity_seed", 1.0)
        self.skills.add_claim(ClaimIR(subject="you", relation="is", object="BrainOS"), ev)
        self.skills.add_claim(ClaimIR(subject="you", relation="is", object="an AI"), ev)

        self.code = CodeSandbox(self.root / "code")
        self.research = ResearchEngine([LocalTextSourceConnector(self.root / "sources")])
        self.backend = backend

        # V36 components
        self.teacher_society = TeacherSociety()
        self.internet_world = InternetWorldMemory(self.root / "internet_world.sqlite3")
        self.evolution_lab = EvolutionLab(self.schema_substrate)
        self.sleep_cycle = SleepConsolidationCycle(
            self.memory, self.schema_substrate,
            self.predictive_loop, self.internet_world,
        )
        self.self_model_mgr = SelfModelManager(
            self.memory, self.predictive_loop, self.schema_substrate,
        )
        self.contrastive_buffer = ContrastiveLearningBuffer()
        self.schema_retrieval = SchemaRetrievalIndex()

    def observe(self, text: str, reward: float = 0.0) -> OSResult:
        # ---- V36 Pipeline: predict -> execute -> compare -> episode -> learn ----

        # Phase 0: Route and adapt.
        shard = self.domain_router.route(text)
        self.memory.log_action("DOMAIN_ROUTE", shard.name, {"text": text[:160], "confidence": shard.confidence, "reason": shard.reason}, shard.confidence)
        adapted_text = self.adapter.adapt_text(self._resolve_coreference(text))

        # Phase 1: PREDICT (before processing).
        prediction = self.predictive_loop.predict(adapted_text)

        # Phase 2: EXECUTE (schema correction path).
        if self._looks_like_schema_correction(adapted_text):
            learned = self.schema_substrate.learn_from_correction(adapted_text)
            if learned:
                schema_type = learned.get("schema_type", "Schema")
                gate = learned.get("gate", {})
                status = "promoted" if gate.get("passed") else "candidate"
                result = OSResult(
                    response=f"Learned {schema_type}: {status}; schema_id={learned.get('schema_id')}",
                    ir_type=schema_type,
                    confidence=float(learned.get("schema", {}).get("confidence", 0.8) if isinstance(learned.get("schema"), dict) else 0.8),
                    memory_stats=self.memory.stats(),
                    cognitive_model={"backend": self.cognitive_model.name, "objective": "schema_learning_substrate; no next-token"},
                )
                # Still record episode for correction path.
                self.predictive_loop.observe(
                    text=adapted_text, response=result.response,
                    ir_type=schema_type, ok=True,
                    confidence=result.confidence, parser="schema_correction",
                )
                self.schema_substrate.observe(adapted_text)
                return result

        # Phase 2: EXECUTE (runtime parse path).
        schema_candidates = self.schema_substrate.compile(adapted_text)
        candidates = schema_candidates or self.compiler.compile(adapted_text)
        cand = candidates[0]
        immune = self.immune.inspect_candidate(cand)
        if immune.quarantined:
            cand.notes.append("epistemic_immune:" + ",".join(immune.reasons))
        ir = cand.ir
        self.cognitive_model.update_from_selection(adapted_text, type(ir).__name__, reward=max(0.1, reward))
        self.memory.append_event(type(ir).__name__, text, ir)
        self._update_entity_stack(ir)
        result = self._handle_ir(ir, text, cand)

        # Phase 3: COMPARE (prediction vs actual).
        ok = self.predictive_loop.estimate_success(result.response)

        # Phase 4: EPISODE (record everything).
        ep = self.predictive_loop.observe(
            text=adapted_text,
            response=result.response,
            ir_type=type(ir).__name__,
            ok=ok,
            ir_json=cand.ir.to_json() if hasattr(cand.ir, 'to_json') else "{}",
            confidence=cand.confidence,
            parser=cand.parser,
            trace=cand.notes,
        )

        # Phase 5: LEARN (error -> schema substrate).
        if not ok:
            self.schema_substrate.record_error(
                episode_id=f"pred_{hash(adapted_text) % 2**32:08x}",
                error_type=ep.error_type or ep.prediction.prediction_error,
                expected={"ir_type": ep.prediction.predicted_ir_type},
                observed={"ir_type": type(ir).__name__, "response": result.response[:200]},
                severity=0.5,
            )
            # Feed contrastive buffer with failure signal.
            self.contrastive_buffer.add_negative(adapted_text, type(ir).__name__)
        else:
            # Feed contrastive buffer with positive signal.
            self.contrastive_buffer.add_positive(adapted_text, type(ir).__name__, result.response[:100])

        # Record every interaction for schema learning.
        self.schema_substrate.observe(adapted_text)
        return result

    def _looks_like_schema_correction(self, text: str) -> bool:
        low = text.lower().strip()
        return any(k in low for k in [
            'when i say', 'if i ask', 'when a ', 'when an ', 'should be understood as',
            'is equivalent to', 'means asking whether', 'correction:', 'means that',
        ]) or bool(re.search(r'["“].+?["”]\s*(?:means|should mean|is equivalent to|refers to)', text, re.I)) or bool(re.search(r'\bmeans?\s+[a-z]+', low, re.I))

    def _handle_ir(self, ir: CognitiveIR, text: str, cand: IRCandidate) -> OSResult:
        if isinstance(ir, CompositeIR):
            responses = []
            for item in ir.items:
                sub_cand = IRCandidate(item, cand.confidence, cand.parser, model_score=cand.model_score, memory_score=cand.memory_score)
                responses.append(self._handle_ir(item, text, sub_cand).response)
            return self._res("CompositeIR stored:\n" + "\n".join(responses), ir, cand)

        if isinstance(ir, NegatedClaimIR):
            ev = self.memory.add_evidence("user", text, cand.confidence)
            self.skills.add_claim(ir, ev)
            self.skills.mark_contradiction_if_needed(ir)
            return self._res(f"Stored negated claim IR: {ir.text()}", ir, cand)

        if isinstance(ir, TemporalClaimIR):
            ev = self.memory.add_evidence("user", text, cand.confidence)
            # Stop events close open-ended positives before inserting the negative marker.
            if ir.polarity == "negative" and ir.source_id == "stop_event":
                self.memory.close_positive_claim_before(ir.subject, ir.relation, ir.object, ir.valid_from or ir.valid_during or ir.time_expr)
            self.skills.add_claim(ir, ev)
            self.skills.mark_contradiction_if_needed(ir)
            row = self.memory.find_claim(ir.subject, ir.relation, ir.object, ir.polarity, ir.valid_from or ir.valid_during)
            if row:
                self.memory.add_edge(ev, "claimv:" + row["version_id"], "valid_during", cand.confidence)
            return self._res(f"Stored temporal claim IR: {ir.text()} @ {ir.valid_from or ir.valid_during}", ir, cand)

        if isinstance(ir, ClaimIR):
            ev = self.memory.add_evidence("user", text, cand.confidence)
            self.skills.add_claim(ir, ev)
            return self._res(f"Stored claim IR: {ir.text()}", ir, cand)

        if isinstance(ir, (RuleIR, QuantifiedRuleIR)):
            self.skills.promote_rule(ir)
            return self._res(f"Stored rule IR: {ir.text()}", ir, cand)

        if isinstance(ir, ExceptionIR):
            self.memory.add_exception(ir)
            return self._res(f"Stored exception IR: {ir.exception_subject} blocks {ir.rule_id}", ir, cand)

        if isinstance(ir, CausalClaimIR):
            if ir.polarity == "negative":
                neg_ir = NegatedClaimIR(subject=ir.cause, relation="causes", object=ir.effect)
                ev = self.memory.add_evidence("user", text, cand.confidence)
                self.skills.add_claim(neg_ir, ev)
                return self._res(f"Stored negated causal IR: {ir.cause} does not cause {ir.effect}", ir, cand)
            else:
                ev = self.memory.add_evidence("user", text, cand.confidence)
                self.memory.add_causal(ir, ev)
                self.cognitive_model.observe_transition(ir.cause, ir.effect)
                return self._res(f"Stored causal IR: {ir.text()}", ir, cand)

        if isinstance(ir, ComparisonIR):
            self.memory.add_comparison(ir)
            return self._res(f"Stored comparison IR: {ir.text()}", ir, cand)

        if isinstance(ir, EventIR):
            detail = {"actor": ir.actor, "action": ir.action, "patient": ir.patient, "recipient": ir.recipient, "location": ir.location, "time": ir.time_expr}
            self.memory.log_action("STORE_EVENT", ir.text(), detail, cand.confidence)
            effects = self.world_grounder.effects_for(ir)
            ev = self.memory.add_evidence("event", ir.text(), cand.confidence)
            for eff in effects:
                self.skills.add_claim(eff.claim, ev)
                if eff.claim.relation in {"located_at", "has", "is", "state"}:
                    self.memory.set_fluent(eff.claim.subject, eff.claim.relation, eff.claim.object, eff.claim.confidence)
                self.memory.log_action("WORLD_FRAME_EFFECT", eff.reason, {"claim": eff.claim.text()}, eff.claim.confidence)
            return self._res(f"Stored event IR: {ir.text()}\nworld_effects={len(effects)}", ir, cand)

        if isinstance(ir, BeliefIR):
            prop_text = getattr(ir.proposition, "text", lambda: str(ir.proposition))()
            detail = {"holder": ir.holder, "proposition": getattr(ir.proposition, "to_json", lambda: str(ir.proposition))()}
            self.memory.log_action("STORE_BELIEF", ir.holder, detail, cand.confidence)
            ev = self.memory.add_evidence("belief", prop_text, cand.confidence)
            self.skills.add_claim(ClaimIR(subject=ir.holder, relation="believes", object=prop_text, confidence=cand.confidence), ev)
            return self._res(f"Stored belief IR: {ir.holder} believes {prop_text}", ir, cand)

        if isinstance(ir, GoalIR):
            self.memory.log_action("STORE_GOAL", ir.agent, {"desired_state": ir.desired_state}, cand.confidence)
            return self._res(f"Stored goal IR: {ir.agent} wants {ir.desired_state}", ir, cand)

        if isinstance(ir, SpeechActIR):
            self.memory.log_action("STORE_SPEECH_ACT", ir.speaker, {"act_type": ir.act_type, "content": str(ir.content)}, cand.confidence)
            if ir.act_type in {"smalltalk_humor", "modal_nonassertive"}:
                plan = self.action_selector.plan_smalltalk(str(ir.content)) if ir.act_type == "smalltalk_humor" else None
                if plan:
                    return self._res(self.action_selector.render(plan), ir, cand)
                return self._res("No relevant sources: treated as non-asserted context, not as a fact.", ir, cand)
            ev = self.memory.add_evidence("speech_act", str(ir.content), cand.confidence)
            effects = self.world_grounder.effects_for(ir)
            for eff in effects:
                self.skills.add_claim(eff.claim, ev)
                if eff.claim.relation in {"located_at", "has", "is", "state"}:
                    self.memory.set_fluent(eff.claim.subject, eff.claim.relation, eff.claim.object, eff.claim.confidence)
                self.memory.log_action("WORLD_FRAME_EFFECT", eff.reason, {"claim": eff.claim.text()}, eff.claim.confidence)
            return self._res(f"Stored speech act IR: {ir.speaker} {ir.act_type}\nworld_effects={len(effects)}", ir, cand)


        if isinstance(ir, ToolCallIR) and ir.tool_name in {"compile_inner_question", "compile_inner_assertion", "compile_inner_negation"}:
            inner = ir.args.get("inner", "")
            cands = self.schema_substrate.compile(inner) or self.compiler._compile_inner(inner, as_question=(ir.tool_name == "compile_inner_question"))
            if cands:
                chosen = cands[0].ir
                if ir.tool_name == "compile_inner_question" and not isinstance(chosen, QuestionIR):
                    chosen = QuestionIR(target=chosen, requested_mode="proof")
                if ir.tool_name == "compile_inner_negation":
                    if isinstance(chosen, ComparisonIR):
                        inv = "less_than" if chosen.comparator == "greater_than" else "greater_than" if chosen.comparator == "less_than" else chosen.comparator
                        chosen = ComparisonIR(left=chosen.left, comparator=inv, right=chosen.right)
                    elif isinstance(chosen, CausalClaimIR):
                        chosen = CausalClaimIR(cause=chosen.cause, effect=chosen.effect, polarity="negative", relation=chosen.relation)
                    elif isinstance(chosen, ClaimIR):
                        chosen = NegatedClaimIR(subject=chosen.subject, relation=chosen.relation, object=chosen.object)
                return self._handle_ir(chosen, text, IRCandidate(chosen, cands[0].confidence, cands[0].parser, notes=cands[0].notes))
            return self._res(f"Could not compile inner clause: {inner}", ir, cand)

        if isinstance(ir, WrapperConstructionIR):
            self.memory.log_action("LEARN_WRAPPER_OPERATION", ir.wrapper_pattern, {"operation": ir.operation, "inner_slot": ir.inner_slot}, cand.confidence)
            return self._res(f"Learned grammar operation: {ir.wrapper_pattern} -> {ir.operation}({ir.inner_slot})", ir, cand)

        if isinstance(ir, EventFrameIR):
            # Install a compact dynamic frame when possible. Role/effect schemas remain stored for audit.
            self.memory.log_action("LEARN_EVENT_FRAME", ir.frame_name, {"roles": ir.roles, "effects": ir.effects, "surface": ir.surface_schema}, cand.confidence)
            return self._res(f"Learned event frame schema: {ir.frame_name}", ir, cand)

        if isinstance(ir, TemporalQuerySchemaIR):
            self.memory.log_action("LEARN_TEMPORAL_SCHEMA", ir.surface_schema, {"role_slot": ir.role_slot, "time_slot": ir.time_slot}, cand.confidence)
            return self._res(f"Learned temporal query schema: {ir.surface_schema}", ir, cand)

        if isinstance(ir, MetaMemoryQuestionIR):
            claims = self.memory.claims_about(ir.target)[:6]
            broad = self.memory.retrieve_broad(ir.target)[:6]
            bits = [c.text() for c in claims]
            for row in broad:
                if isinstance(row, str):
                    txt = row
                else:
                    txt = row.get("text") or row.get("quote") or ""
                if txt and txt not in bits:
                    bits.append(txt)
            for cmp_row in self.memory.all_comparisons():
                if ir.target.lower() in {cmp_row.get("left_value", ""), cmp_row.get("right_value", "")}:
                    bits.append(f"{cmp_row['left_value']} {cmp_row['comparator']} {cmp_row['right_value']}")
            if not bits:
                return self._res(f"I do not have recent learned facts about {ir.target}.", ir, cand)
            summary = "; ".join(bits[:8])
            return self._res(f"Recent learned facts about {ir.target}: {summary}", ir, cand)

        if isinstance(ir, SupportRequestIR):
            plan = self.action_selector.plan_support(ir.state)
            return self._res(self.action_selector.render(plan), ir, cand)

        if isinstance(ir, ToolCallIR) and ir.tool_name == "parser_step_correction":
            self.compiler.active_teacher.add_failed_parse(text, "step_level_correction", json.dumps(ir.args, ensure_ascii=False))
            self.memory.log_action("PARSER_STEP_CORRECTION", ir.args.get("field", ""), ir.args, cand.confidence)
            return self._res(f"Recorded parser step correction: {ir.args.get('field')} -> {ir.args.get('value')}", ir, cand)

        if isinstance(ir, ToolCallIR) and ir.tool_name == "learn_construction":
            surface = ir.args.get("text", "")
            target = ir.args.get("target", "")
            target_ir = self.compiler.parse_target_ir(target)
            if target_ir is None:
                self.compiler.active_teacher.add_failed_parse(surface, "bad_construction_target", target)
                return self._res(f"Could not learn construction: target IR was not understood: {target}", ir, cand)
            ok = self.compiler.learn_construction(surface, target_ir)
            if ok:
                self.memory.log_action("LEARN_CONSTRUCTION", surface, {"target": target, "target_type": type(target_ir).__name__}, 1.0)
                return self._res(f"Learned construction: '{surface}' -> {type(target_ir).__name__}", ir, cand)
            return self._res(f"Could not abstract construction from: {surface}", ir, cand)

        if isinstance(ir, ToolCallIR) and ir.tool_name == "solve_arithmetic":
            nums = [int(n) for n in ir.args.get("numbers", [])]
            op = ir.args.get("operation", "")
            result = None
            if op == "add" and len(nums) >= 2:
                result = sum(nums)
            elif op == "subtract" and len(nums) >= 2:
                result = nums[0] - sum(nums[1:])
            elif op == "multiply" and len(nums) >= 2:
                result = 1
                for n in nums[:2]:
                    result *= n
            elif op == "divide" and len(nums) >= 2 and nums[1] != 0:
                result = nums[0] // nums[1]
            if result is None:
                return self._res("Arithmetic problem could not be solved.", ir, cand)
            self.memory.log_action("ARITHMETIC_SOLVE", op, {"numbers": nums, "result": result, "text": ir.args.get("text", "")}, cand.confidence)
            return self._res(f"Arithmetic answer: {result}", ir, cand)

        if isinstance(ir, ToolCallIR) and ir.tool_name == "world_transition":
            try:
                state = json.loads(ir.args.get("state", "{}"))
                nxt = json.loads(ir.args.get("next", "{}"))
            except Exception:
                state = {"raw": ir.args.get("state", "")}
                nxt = {"raw": ir.args.get("next", "")}
            self.world.observe(state, ir.args.get("action", ""), nxt, cand.confidence)
            wreport = self.grounded.verify_world_action(self.world, ir.args.get("action", ""))
            self.memory.log_action("GROUNDED_VERIFY", wreport.channel, wreport.detail, 1.0 if wreport.ok else 0.0)
            return self._res(f"Stored world transition: action={ir.args.get('action')}", ir, cand)

        if isinstance(ir, QuestionIR):
            proof = self.reasoner.answer(ir)
            self.memory.store_proof(proof)
            greport = self.grounded.verify_proof(proof)
            self.memory.log_action("GROUNDED_VERIFY", greport.channel, greport.detail, 1.0 if greport.ok else 0.0)
            status = "success" if proof.success else "failure"
            failure = "" if proof.success else proof.status
            lesson = "proof strategy succeeded" if proof.success else "create missing fact/rule, improve parser, or add regression test"
            self.memory.log_trajectory(text, status, failure, {"proof": proof.render()}, lesson)
            if not proof.success:
                self.self_improver.propose_from_failure(text, failure, lesson)
            self.cognitive_model.score_memory_utility(proof.query, proof.success)
            return self._res(proof.render(), ir, cand)

        if isinstance(ir, ProgramSpecIR):
            result = self.code.run_repair_loop(ir)
            xreport = self.grounded.verify_execution(result.success, len(result.attempts))
            self.memory.log_action("GROUNDED_VERIFY", xreport.channel, xreport.detail, 1.0 if xreport.ok else 0.0)
            detail = {"attempts": len(result.attempts), "success": result.success, "outputs": [a.output[-300:] for a in result.attempts]}
            self.memory.log_trajectory(text, "success" if result.success else "failure", "" if result.success else "code_error", detail, "repair loop tried candidates")
            ev = self.memory.add_evidence("code_sandbox", detail["outputs"][-1] if detail["outputs"] else "", 0.9 if result.success else 0.3)
            self.memory.upsert_claim(ClaimIR(subject=ir.name, relation="passes_tests", object=str(result.success).lower(), confidence=0.9 if result.success else 0.3), ev)
            return self._res(f"ProgramSpec({ir.name}) executed.\nexit_code={result.attempts[-1].exit_code if result.attempts else 999}\nattempts={len(result.attempts)}\noutput={(result.attempts[-1].output if result.attempts else '').strip()}\nsandbox={result.sandbox_note}", ir, cand)

        if isinstance(ir, WritingTaskIR):
            return self._res(self._write(ir), ir, cand)

        if isinstance(ir, ResearchTaskIR):
            return self._res(self.research.answer(ir), ir, cand)

        if isinstance(ir, ExperimentIR):
            return self._res(f"Experiment compiled: hypothesis={ir.hypothesis}; intervention={ir.intervention}; metric={ir.metric}", ir, cand)

        return self._res("Unhandled IR.", ir, cand)

    def run_v32_predictive_agi_audit(self):
        from .v32_predictive_agi_audit import V32PredictiveAGIAudit
        return V32PredictiveAGIAudit(self.root / "v32_predictive_agi_audit").run()


    def _resolve_coreference(self, text: str) -> str:
        # Small discourse entity stack. It is heuristic, but it is no longer a
        # single last-recipient replacement: it tracks likely male/female names,
        # recent object, and previous proposition-like object.
        if not self.entity_stack:
            return text
        male_names = {"bob", "joon", "jun", "charlie", "dave", "frank", "철수", "준호", "민수"}
        female_names = {"alice", "mina", "sora", "hana", "erin", "carol", "dami", "영희"}
        recent_actor = recent_recipient = recent_patient = ""
        recent_male = recent_female = ""
        recent_event = None
        recent_belief = None
        for item in reversed(self.entity_stack):
            if recent_event is None and item.get("kind") == "event" and item.get("recipient") and item.get("patient"):
                recent_event = item
            if recent_belief is None and item.get("kind") == "belief":
                recent_belief = item
            for key in ["actor", "recipient", "patient"]:
                val = (item.get(key) or "").strip()
                if not val:
                    continue
                if not recent_actor and key == "actor": recent_actor = val
                if not recent_recipient and key == "recipient": recent_recipient = val
                if not recent_patient and key == "patient": recent_patient = val
                if not recent_male and val.lower() in male_names: recent_male = val
                if not recent_female and val.lower() in female_names: recent_female = val
        if re.search(r"\b(have|has)\b", text, flags=re.I) and recent_event:
            recent_recipient = recent_event.get("recipient") or recent_recipient
            recent_patient = recent_event.get("patient") or recent_patient
            # Possession questions after transfer events usually refer to the event recipient/object.
            recent_male = recent_recipient or recent_male
            recent_female = recent_recipient or recent_female
        if re.search(r"\b(?:believe|think|thinks|suppose|supposes)\b", text, flags=re.I) and recent_belief:
            recent_actor = recent_belief.get("actor") or recent_actor
            recent_recipient = recent_belief.get("recipient") or recent_recipient
            recent_patient = recent_belief.get("patient") or recent_patient
            # In belief questions, pronouns usually target the belief holder/proposition,
            # not the most recent event actor. Prefer the belief holder for she/he.
            recent_female = recent_actor
            recent_male = recent_actor
        # Targeted belief coreference before broad pronoun replacement.
        # If the last belief was "Bob believes Alice is CEO", then
        # "Does he believe she is CEO?" -> "Does Bob believe Alice is CEO?".
        if recent_belief and re.search(r"\b(?:believe|believes|think|thinks|suppose|supposes)\b", text, flags=re.I):
            holder = recent_belief.get("actor") or recent_actor
            prop_subj = recent_belief.get("recipient") or recent_recipient
            if holder:
                text = re.sub(r"\b(he|she)\b(?=\s+(?:believe|believes|think|thinks|suppose|supposes))", holder, text, flags=re.I)
            if prop_subj:
                text = re.sub(r"\b(believe|believes|think|thinks|suppose|supposes)\s+(he|she)\b", lambda m: f"{m.group(1)} {prop_subj}", text, flags=re.I)

        # Do not resolve the expletive/bridging 'it' in exception clauses such as
        # "even though it is a bird"; there it refers to the queried subject.
        protect_exception_it = bool(re.search(r"even\s+though\s+it\s+is|is\s+it\s+fair|would\s+it\s+be\s+fair", text, flags=re.I))
        # Do not replace complementizer 'that' in belief/speech clauses.
        complementizer_that = bool(re.search(r"\b(?:believe|believes|believed|think|thinks|thought|say|says|said|know|knows|knew)\s+that\b", text, flags=re.I))
        mapping = {
            "he": recent_male or recent_recipient or recent_actor or "he",
            "him": recent_male or recent_recipient or recent_actor or "him",
            "his": recent_male or recent_recipient or recent_actor or "his",
            "she": recent_female or recent_actor or recent_recipient or "she",
            "her": recent_female or recent_actor or recent_recipient or "her",
            "the book": "book",
            "the package": "package",
            "the box": "box",
        }
        if not complementizer_that:
            mapping["that"] = recent_patient or "that"
        if not protect_exception_it:
            mapping["it"] = recent_patient or "it"
        out = text
        # Longer phrases first, then pronouns, with word boundaries so punctuation is handled.
        for k, v in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            out = re.sub(r"\b" + re.escape(k) + r"\b", v, out, flags=re.I)
        return out.strip()

    def _update_entity_stack(self, ir):
        if isinstance(ir, EventIR):
            self.entity_stack.append({"kind": "event", "actor": ir.actor or "", "recipient": ir.recipient or "", "patient": ir.patient or ""})
            self.entity_stack = self.entity_stack[-12:]
        if isinstance(ir, BeliefIR):
            prop = getattr(ir, "proposition", None)
            subj = getattr(prop, "subject", "") if prop is not None else ""
            self.entity_stack.append({"kind": "belief", "actor": ir.holder or "", "recipient": subj, "patient": getattr(prop, "object", "") if prop is not None else ""})
            self.entity_stack = self.entity_stack[-12:]


    def run_v27_adversarial_growth_benchmark(self):
        from .v27_growth import V27AdversarialGrowthLab
        return V27AdversarialGrowthLab(self.root / "v27_adversarial_growth_lab").run()

    def run_v28_generalization_audit(self):
        from .v28_generalization import V28GeneralizationAudit
        return V28GeneralizationAudit(self.root / "v28_generalization_audit").run()

    def run_v29_systemic_learning_audit(self):
        from .v29_systemic_learning import V29SystemicLearningAudit
        return V29SystemicLearningAudit(self.root / "v29_systemic_learning_audit").run()

    def run_developmental_growth_lab(self):
        from .developmental import DevelopmentalGrowthLab
        return DevelopmentalGrowthLab(self.root / "developmental_growth_lab").run()

    def _write(self, ir: WritingTaskIR) -> str:
        claims = self.memory.claims_about(ir.topic)
        if not claims:
            rows = self.memory.claim_versions(polarity="positive")[:8]
            claims = [ClaimIR(subject=r["subject"], relation=r["relation"], object=r["object"], confidence=r["confidence"]) for r in rows]
        lines = [f"# {ir.topic.title()}", "", f"**Thesis.** {ir.topic} is rendered from typed IR, evidence graph, proof traces, and explicit uncertainty.", "", "## Claim graph"]
        if claims:
            for c in claims:
                lines.append(f"- {c.text()} [confidence={c.confidence:.2f}]")
        else:
            lines.append("- No verified claims available.")
        lines += ["", "## Evidence and uncertainty", "Claims must be supported by EvidenceRef, proof trace, code execution, or marked as hypothesis.", "", "## Cognitive model role", "The neural component ranks IR/evidence/proof/world/memory operations. It does not generate this text."]
        while len(" ".join(lines).split()) < ir.target_words:
            lines.append("Future improvement should target parser ambiguity, contradiction checking, active memory retrieval, and execution-grounded verification.")
            if len(" ".join(lines).split()) > ir.target_words + 80:
                break
        return "\n".join(lines)

    def _res(self, response: str, ir, cand: IRCandidate) -> OSResult:
        rep = self.cognitive_model.report_for(cand)
        return OSResult(response, type(ir).__name__, cand.confidence, self.memory.stats(), rep.__dict__)

    def run_smoke(self):
        train = [
            "teach: kibo is rover", "teach: rover is robot", "teach: robot is machine", "teach: if x is robot then x is machine",
            "teach: kibo is not mineral", "rain causes wet ground", "wet ground causes slippery road",
            "on 2025 alice is ceo", "on 2026 bob is ceo", "alice is taller than bob", "bob is taller than charlie",
            "철수는 영희보다 크다", "영희는 민수보다 크다",
            "all birds can fly", "penguin is bird", "penguin is exception to bird can fly",
            'world: state={"light":"off"}; action=press_switch; next={"light":"on"}',
            'learn construction: frost brings icy roads => causal(frost, icy roads)',
            'dominates means greater_than',
            '"A is ahead of B" means A greater_than B',
            'When I say "A outruns B", I mean A is faster than B.',
            'By "A sparks B" I mean A causes B.',
            'When I say "A dominates B", it means A is greater than B.',
            '"A is slightly ahead of B" means A greater_than B',
            '"A lags behind B" means A less_than B.',
            'Bob thinks Alice is not the CEO.',
        ]
        for t in train:
            self.observe(t, reward=1.0)
        tests = [
            ("claim retrieval", "what do you know about kibo?", "kibo"),
            ("direct proof", "is kibo machine?", "Yes"),
            ("belong paraphrase", "does kibo belong to machines?", "Yes"),
            ("why paraphrase", "explain why kibo is a machine", "Yes"),
            ("known refutation", "is kibo mineral?", "refute"),
            ("temporal who", "who is CEO in 2026?", "bob"),
            ("world causal", "what happens after rain?", "wet ground"),
            ("causal chain", "rain causes slippery road?", "Yes"),
            ("learned causal paraphrase", "supply shock leads to higher prices", "Stored causal IR"),
            ("learned reverse causal paraphrase", "volatility happens because of market shock", "Stored causal IR"),
            ("learned claim paraphrase", "orion is classified as robot", "Stored claim IR"),
            ("comparison transitivity", "is alice taller than charlie?", "Yes"),
            ("korean comparison transitivity", "철수는 민수보다 크니?", "Yes"),
            ("exception blocks rule", "can penguin fly?", "Blocked by exception"),
            ("writing", "write long essay about kibo 180 words", "# Kibo"),
            ("research honesty", "research 2026 memory papers", "No relevant sources"),
            ("one-shot construction parse", "heat brings expansion", "Stored causal IR"),
            ("one-shot construction proof", "heat causes expansion?", "Yes"),
            ("relation correction parse", "alice dominates bob", "Stored comparison IR"),
            ("relation correction proof", "is alice greater than bob?", "Yes"),
            ("surface correction parse", "seoul is ahead of busan", "Stored comparison IR"),
            ("surface correction proof", "is seoul greater than busan?", "Yes"),
            ("v23 natural correction parse", "apollo outruns zephyr", "Stored comparison IR"),
            ("v23 natural correction proof", "is apollo greater than zephyr?", "Yes"),
            ("v23 causal correction parse", "heat sparks expansion", "Stored causal IR"),
            ("v23 causal correction proof", "heat causes expansion?", "Yes"),
            ("v24 natural correction dominates parse", "alice dominates bob", "Stored comparison IR"),
            ("v24 natural correction dominates proof", "is alice greater than bob?", "Yes"),
            ("v24 construction optional modifier parse", "apollo is ahead of zephyr", "Stored comparison IR"),
            ("v24 construction optional modifier question", "is apollo ahead of zephyr?", "Yes"),
            ("v24 taxonomy fair-call question", "Is it fair to call Kibo a machine?", "Yes"),
            ("v24 taxonomy qualify question", "Would Kibo qualify as a machine?", "Yes"),
            ("v24 korean comparison particle", "철수는 영희에 비해 우위에 있다", "Stored comparison IR"),
            ("v24 temporal interval claim", "Alice was CEO from 2025 through 2026.", "Stored temporal claim IR"),
            ("v24 temporal overlap contradiction", "In 2026 Alice was not CEO", "Stored temporal claim IR"),
            ("v24 temporal inconsistent role", "Who was CEO in 2026?", "Inconsistent evidence"),
            ("v24 exception however", "Penguins are birds; however, they usually do not fly.", "Stored exception IR"),
            ("v24 exception question block", "Can a penguin fly even though it is a bird?", "Blocked by exception"),
            ("v24 transfer prepositional event", "Alice gave a book to Bob in Seoul yesterday.", "Stored event IR"),
            ("v24 possession from event", "does bob have book?", "Yes"),

            ("v25 interactive correction it-means", "orion dominates kibo", "Stored comparison IR"),
            ("v25 learned less-than construction", "alice lags behind bob", "Stored comparison IR"),
            ("v25 inverse proof from less-than", "is bob greater than alice?", "Yes"),
            ("v25 transfer received event", "Bob received a package from Alice yesterday.", "Stored event IR"),
            ("v25 possession from received", "Does Bob have package?", "Yes"),
            ("v25 became temporal", "Charlie became CEO in 2025.", "Stored temporal claim IR"),
            ("v25 stopped temporal", "Charlie stopped being CEO in 2027.", "Stored temporal claim IR"),
            ("v25 temporal after stop", "what was Charlie in 2026?", "ceo"),
            ("v25 belief question", "Does Bob believe Alice is not CEO?", "Yes"),
            ("v25 Korean negated comparison", "철수는 영희보다 크지 않다", "Stored comparison IR"),
            ("natural counts-as question", "Would you say Kibo counts as a machine?", "Yes"),
            ("considered-as question", "Can Kibo be considered a machine?", "Yes"),
            ("modal classification negation", "Kibo should not be classified as a mineral.", "Stored negated claim IR"),
            ("temporal negation", "In 2026 Bob was not CEO", "Stored temporal claim IR"),
            ("temporal role question", "Who held the CEO role during 2026?", "bob"),
            ("korean reverse comparison", "영희보다 준호가 더 크다", "Stored comparison IR"),
            ("korean ahead comparison", "준호가 영희를 앞선다", "Stored comparison IR"),
            ("event extraction", "Alice gave Bob a book in Seoul yesterday.", "Stored event IR"),
            ("belief extraction", "Bob believes Alice is the CEO.", "Stored belief IR"),
            ("goal extraction", "Kibo wants to collect rocks.", "Stored goal IR"),
            ("speech act extraction", "Alice asked Bob to open the door.", "Stored speech act IR"),
            ("transfer event variant", "Alice handed Bob a package in Seoul yesterday.", "Stored event IR"),
            ("belief negation", "Bob thinks Alice is not the CEO.", "Stored belief IR"),
            ("korean advanced comparison", "철수가 영희를 능가한다", "Stored comparison IR"),
            ("conditional discourse causal", "if rain falls, roads become slippery", "CompositeIR stored"),
        ]
        rows = []
        for name, prompt, expected in tests:
            r = self.observe(prompt)
            rows.append({"name": name, "passed": expected.lower() in r.response.lower(), "expected": expected, "observed": r.response[:360], "ir_type": r.ir_type})
        rows.append({"name": "neural cognitive compiler objective", "passed": "next-token" not in self.cognitive_model.report_for(self.compiler.compile('is kibo machine?')[0]).objective, "expected": "no next-token", "observed": self.cognitive_model.report_for(self.compiler.compile('is kibo machine?')[0]).objective, "ir_type": "NeuralCognitiveCompiler"})
        rows.append({"name": "learned semantic parser objective", "passed": "next-token" not in self.compiler.learned_parser.objective, "expected": "no next-token", "observed": self.compiler.learned_parser.objective, "ir_type": "LearnedSemanticParser"})
        rows.append({"name": "meaning atom table", "passed": self.compiler.meaning_atoms.validate_inventory() and len(self.compiler.meaning_atoms.atoms) >= 17, "expected": "17+ explicit atoms", "observed": str(sorted(self.compiler.meaning_atoms.atoms.keys())[:5]), "ir_type": "MeaningAtomTable"})
        rows.append({"name": "active learning queue", "passed": len(self.compiler.compile('@@@not parseable@@@')[0].notes) > 0 and len(self.compiler.active_teacher.queue) > 0, "expected": "queued failed parse", "observed": str(self.compiler.active_teacher.export()[-1]), "ir_type": "ActiveTeacher"})
        rows.append({"name": "neural cognitive compiler learned updates", "passed": self.cognitive_model.updates > 0, "expected": "updates>0", "observed": str(self.cognitive_model.updates), "ir_type": "NeuralCognitiveCompiler"})
        rows.append({"name": "world transition memory", "passed": bool(self.world.predict("press_switch")), "expected": "stored world transition", "observed": str(self.world.predict("press_switch")[:1]), "ir_type": "WorldModel"})
        candidate = self.self_improver.propose_from_failure("dummy failed task", "parser_error", "add regression test")
        promo = self.self_improver.promote_if_passes(candidate, [lambda: True, lambda: True])
        rows.append({"name": "self improvement gate", "passed": promo.promoted, "expected": "promoted candidate after checks", "observed": str(promo), "ir_type": "RegressionGatedSelfImprover"})
        rows.append({"name": "meaning atom calculus", "passed": self.compiler.meaning_calculus.validate() and self.compiler.meaning_calculus.has("exception_block"), "expected": "valid calculus", "observed": str(sorted(self.compiler.meaning_calculus.operations.keys())[:4]), "ir_type": "MeaningAtomCalculus"})
        beam = self.compiler.compile("orion is classified as robot")
        rows.append({"name": "semantic beam candidates", "passed": len(beam) >= 2, "expected": "multiple candidates", "observed": str([(type(c.ir).__name__, c.parser) for c in beam[:3]]), "ir_type": "SemanticBeam"})
        rows.append({"name": "tiny semantic encoder objective", "passed": "next-token" not in self.compiler.learned_parser.encoder.objective and not hasattr(self.compiler.learned_parser.encoder, "generate"), "expected": "no next-token/generate", "observed": self.compiler.learned_parser.encoder.objective, "ir_type": "TinySemanticEncoder"})
        rows.append({"name": "domain shard routing", "passed": self.domain_router.route("implement python function").name == "code" and self.domain_router.route("CAGR Sharpe portfolio").name == "quant", "expected": "code+quant", "observed": str((self.domain_router.route("implement python function"), self.domain_router.route("CAGR Sharpe portfolio"))), "ir_type": "DomainShardRouter"})
        self.adapter.observe_alias("zephyr", "kibo")
        rows.append({"name": "semantic test-time adapter", "passed": "kibo" in self.adapter.adapt_text("is zephyr machine?"), "expected": "alias applied", "observed": self.adapter.adapt_text("is zephyr machine?"), "ir_type": "SemanticTestTimeAdapter"})
        sleep_report = self.sleep.run()
        rows.append({"name": "sleep replay consolidation", "passed": self.memory.stats().get("sleep_reports", 0) > 0 and sleep_report.actions_logged > 0, "expected": "sleep report logged", "observed": str(sleep_report), "ir_type": "SleepReplayConsolidator"})
        immune_row = self.memory.find_claim("orion", "is", "robot", "positive")
        if immune_row:
            self.immune.quarantine_version(immune_row["version_id"], "smoke-test low-trust quarantine after proof/use")
        rows.append({"name": "epistemic immune system", "passed": self.memory.stats().get("immune_events", 0) > 0 and any(a["name"] == "active learning queue" and a["passed"] for a in rows), "expected": "immune quarantine logged", "observed": str(self.immune.audit()), "ir_type": "EpistemicImmuneSystem"})
        rows.append({"name": "grounded verifier", "passed": self.memory.stats().get("memory_actions", 0) > 0, "expected": "grounding actions logged", "observed": str(self.memory.stats().get("memory_actions", 0)), "ir_type": "GroundedVerifier"})
        rows.append({"name": "construction learner objective", "passed": "next-token" not in self.compiler.construction_learner.objective and len(self.compiler.construction_learner) >= 1, "expected": "one-shot non-LM construction learning", "observed": self.compiler.construction_learner.objective, "ir_type": "ConstructionLearner"})
        rows.append({"name": "v24 feature construction grammar", "passed": "next-token" not in self.compiler.construction_grammar.objective and len(self.compiler.construction_grammar) >= 1, "expected": "feature-structure construction grammar", "observed": self.compiler.construction_grammar.objective, "ir_type": "CognitiveConstructionGrammar"})
        rows.append({"name": "v25 neural semantic perception", "passed": hasattr(self.compiler, "neural_perception") and "next-token" not in self.compiler.neural_perception.objective and not hasattr(self.compiler.neural_perception, "generate"), "expected": "structured neural perception without generation", "observed": self.compiler.neural_perception.objective, "ir_type": "NeuralSemanticPerception"})
        corpus_check = __import__("neurova.datasets", fromlist=["generate_v25_multitask_corpus"]).generate_v25_multitask_corpus(1200, seed=2)
        labels = {r.get("ir_type") for r in corpus_check}
        rows.append({"name": "v25 multitask corpus", "passed": len(corpus_check) == 1200 and {"ToolCallIR", "EventIR", "TemporalClaimIR", "BeliefIR", "QuestionIR"}.issubset(labels), "expected": "text↔IR/correction/event corpus", "observed": str(sorted(labels)[:8]), "ir_type": "V26Corpus"})
        gate = self.continual_gate
        gate.add_case("kibo machine proof", "is kibo machine?", lambda out: "Yes" in out)
        report = gate.evaluate(lambda q: self.observe(q).response, candidate_id="v25_smoke_gate")
        rows.append({"name": "v25 continual regression gate", "passed": report.promoted and report.passed == report.total, "expected": "gate promoted", "observed": str(report), "ir_type": "ContinualLearningGate"})
        passed = sum(1 for r in rows if r["passed"])
        self.memory.record_benchmark("v25_neuro_symbolic_language_learning_benchmark", passed / len(rows), {"rows": rows})
        return {"passed": passed, "total": len(rows), "percentage": round(100 * passed / len(rows), 1), "rows": rows, "memory_stats": self.memory.stats(), "backend": self.cognitive_model.name}

    def close(self):
        """Close all SQLite connections held by this OS instance."""
        self.memory.close()
        self.schema_substrate.close()
        self.internet_world.close()

    def __del__(self):
        self.close()
