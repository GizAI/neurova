from __future__ import annotations
import re
from typing import List, Optional
from ..ir import *


def norm(x: str) -> str:
    x = str(x or '').strip().lower()
    x = re.sub(r'[?.!]+$', '', x)
    x = re.sub(r'^(a|an|the)\s+', '', x)
    x = re.sub(r'\s+', ' ', x)
    return x.strip()


def strip_meta_prefix(s: str) -> str:
    return re.sub(r'^\s*(?:no,\s*|actually,\s*|correction:\s*|in this domain,\s*|for this task,\s*|here,\s*|i meant,\s*|what i meant is\s+that\s*)+', '', s.strip(), flags=re.I)


def plural_singular(x: str) -> str:
    x = norm(x)
    irregular = {'children':'child','people':'person','men':'man','women':'woman','geese':'goose','mice':'mouse'}
    if x in irregular:
        return irregular[x]
    if x.endswith('ies') and len(x) > 4:
        return x[:-3] + 'y'
    if x.endswith('ches') or x.endswith('shes') or x.endswith('xes'):
        return x[:-2]
    if x.endswith('ses') and not x.endswith('sses'):
        return x[:-2]
    if x.endswith('s') and len(x) > 3 and not x.endswith('ss'):
        return x[:-1]
    return x


def parse_simple_target(target: str) -> Optional[CognitiveIR]:
    t = norm(target.strip().strip('"\''))
    m = re.match(r'^(.+?)\s+(?:is\s+)?(?:greater_than|greater\s+than|faster\s+than|ahead\s+of|above|dominant\s+over|outranks|outclasses)\s+(.+)$', t)
    if m:
        return ComparisonIR(left=norm(m.group(1)), comparator='greater_than', right=norm(m.group(2)))
    m = re.match(r'^(.+?)\s+(?:is\s+)?(?:less_than|less\s+than|behind|below|slower\s+than|trails|lags\s+behind)\s+(.+)$', t)
    if m:
        return ComparisonIR(left=norm(m.group(1)), comparator='less_than', right=norm(m.group(2)))
    m = re.match(r'^(.+?)\s+(?:causes|cause|leads\s+to|triggers|gives\s+rise\s+to|sparks)\s+(.+)$', t)
    if m:
        return CausalClaimIR(cause=norm(m.group(1)), effect=norm(m.group(2)))
    m = re.match(r'^(.+?)\s+(?:is|are|classifies\s+as|counts\s+as|falls\s+under|is\s+a\s+type\s+of)\s+not\s+(.+)$', t)
    if m:
        return NegatedClaimIR(subject=norm(m.group(1)), relation='is', object=norm(m.group(2)))
    m = re.match(r'^(.+?)\s+(?:is|are|classifies\s+as|counts\s+as|falls\s+under|is\s+a\s+type\s+of|is\s+kind\s+of)\s+(.+)$', t)
    if m:
        return ClaimIR(subject=norm(m.group(1)), relation='is', object=norm(m.group(2)))
    return None


class V30UnifiedFrontEnd:
    """Wrapper-first, schema-first front end.

    The goal is to stop accumulating one-off regex patches.  This layer emits
    higher-order operations or normalized IR before the legacy parser cascade runs:
    wrapper(P) -> QuestionIR(P), event frame -> fluent world effect, temporal
    schema -> interval query, and dialogue act -> response plan.  It is compact,
    deterministic, and non-autoregressive.
    """
    name = 'v30_unified_frontend'

    def parse(self, text: str) -> List[IRCandidate]:
        raw = text.strip()
        s = strip_meta_prefix(raw).strip()
        low = norm(s)
        out: List[IRCandidate] = []

        # Social/dialogue acts before generic claim parsing.
        if re.search(r'\b(haha|lol|that\'s wild|wild|funny|hilarious|nice)\b', low):
            return [IRCandidate(SpeechActIR(speaker='user', act_type='smalltalk_humor', content=raw), 0.98, self.name, notes=['dialogue_act'])]
        if re.search(r'\b(worried|anxious|stuck|confused|overwhelmed|doing this wrong|don\'t understand|모르겠|헷갈|막혔)\b', low):
            return [IRCandidate(SupportRequestIR(state='confused_or_worried', request='support_and_next_step'), 0.98, self.name, notes=['dialogue_act_support'])]
        m = re.match(r'^(?:what\s+did\s+we\s+(?:just\s+)?learn\s+about|what\s+do\s+we\s+know\s+about)\s+(.+?)\??$', low)
        if m:
            return [IRCandidate(MetaMemoryQuestionIR(target=norm(m.group(1))), 0.98, self.name, notes=['dialogue_act_meta_memory'])]

        # Simple greetings and salutations.
        if re.search(r'^(?:hi|hello|hey|yo|good\s+(?:morning|afternoon|evening)|greetings)[\s.!?,]*$', low):
            return [IRCandidate(SpeechActIR(speaker='user', act_type='smalltalk_humor', content=raw), 0.99, self.name, notes=['greeting'])]

        # Natural correction -> reusable construction.
        m = re.match(r'^(?:when\s+i\s+say\s+)?["“](.+?)["”]\s*(?:,?\s*(?:it\s+means|i\s+mean|means|is\s+equivalent\s+to|should\s+be\s+understood\s+as))\s+(.+)$', s, re.I)
        if not m:
            m = re.match(r'^(?:by\s+)?["“](.+?)["”]\s*,?\s*(?:i\s+mean|means|should\s+be\s+understood\s+as)\s+(.+)$', s, re.I)
        if m:
            surface, target = m.groups()
            target_ir = parse_simple_target(target)
            if target_ir is not None:
                target_s = self._target_string(target_ir)
                return [IRCandidate(ToolCallIR(tool_name='learn_construction', args={'text': surface.strip(), 'target': target_s, 'source': 'v30_natural_feedback'}), 0.99, self.name, notes=['correction_to_construction'])]

        # Event-frame correction: learn a multi-slot frame.
        if re.search(r'\bA\s+(?:carries|transports|moves|relocates)\s+B\s+from\s+C\s+to\s+D\b', s, re.I) and re.search(r'located\s+at\s+D|moves\s+B\s+from\s+C\s+to\s+D', s, re.I):
            frame = EventFrameIR(frame_name='move_transfer', surface_schema='A carries B from C to D', roles={'actor':'A','patient':'B','source':'C','destination':'D'}, effects=[{'subject':'B','relation':'located_at','object':'D'}], variants=['A carried B from C to D','B was carried from C to D by A','A transports B from C to D','A moved B from C to D'])
            return [IRCandidate(frame, 0.99, self.name, notes=['event_frame_learning'])]

        # Wrapper-first decomposition: strip wrapper and ask compiler to compile the inner clause.
        wrapper_patterns = [
            r'^(?:would\s+you\s+say|could\s+we\s+say|can\s+we\s+say|do\s+you\s+think|is\s+it\s+true\s+that|would\s+you\s+classify)\s+(.+?)\??$',
            r'^(?:is\s+it\s+fair\s+to\s+call)\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$',
            r'^(?:could|can|would)\s+(.+?)\s+be\s+(?:regarded|considered|treated|classified)\s+as\s+(?:a\s+|an\s+|the\s+)?(.+?)\??$',
            r'^does\s+(.+?)\s+fall\s+under\s+(.+?)\??$',
            r'^is\s+(.+?)\s+a\s+type\s+of\s+(.+?)\??$',
        ]
        for idx, pat in enumerate(wrapper_patterns):
            m = re.match(pat, low, re.I)
            if m:
                if idx == 0:
                    inner = m.group(1)
                else:
                    inner = f'{m.group(1)} is {m.group(2)}'
                return [IRCandidate(ToolCallIR(tool_name='compile_inner_question', args={'inner': inner, 'schema': 'v30_wrapper_first'}), 0.99, self.name, notes=['wrapper_first_decomposition'])]

        # Do/did/does question over any learned binary construction.
        m = re.match(r'^(?:did|does|do)\s+(.+?)\s+([a-z][a-z\-]+)\s+(.+?)\??$', low, re.I)
        if m:
            subj, verb, obj = m.groups()
            if verb.lower() not in {'believe', 'think', 'believes', 'thinks'}:
                return [IRCandidate(ToolCallIR(tool_name='compile_inner_question', args={'inner': f'{subj} {verb} {obj}', 'verb': verb, 'schema': 'v30_do_question'}), 0.99, self.name, notes=['do_support_wrapper'])]

        # Do-support negation as operation over inner construction.
        m = re.match(r'^(.+?)\s+(?:does|do|did)\s+(?:not|n\'t)\s+([a-z][a-z\-]+)\s+(.+?)$', low, re.I)
        if m:
            subj, verb, obj = m.groups()
            return [IRCandidate(ToolCallIR(tool_name='compile_inner_negation', args={'inner': f'{subj} {verb} {obj}', 'verb': verb, 'schema': 'v30_do_negation'}), 0.99, self.name, notes=['do_support_negation_wrapper'])]

        # Passive voice as operation over inner construction.
        m = re.match(r'^(.+?)\s+(?:is|was|were|been)\s+([a-z][a-z\-]+?)(?:ed)?\s+by\s+(.+?)\??$', low, re.I)
        if m:
            patient, verb, actor = m.groups()
            return [IRCandidate(ToolCallIR(tool_name='compile_inner_assertion', args={'inner': f'{actor} {verb} {patient}', 'verb': verb, 'schema': 'v30_passive'}), 0.99, self.name, notes=['passive_decomposition'])]

        # Goal / intention is a dialogue-cognitive state, not a factual event assertion.
        m = re.match(r'^(.+?)\s+(?:wants|intends|plans\s+on|plans)\s+(?:to\s+)?(.+)$', low)
        if m:
            return [IRCandidate(GoalIR(agent=norm(m.group(1)), desired_state=norm(m.group(2))), 0.97, self.name, notes=['goal_dialogue_state'])]

        # Modal/scope guards: do not assert as fact.
        m = re.match(r'^(.+?)\s+(?:almost|nearly|wanted\s+to|wants\s+to|failed\s+to|tried\s+to|is\s+said\s+to|was\s+expected\s+to)\s+(.+)$', low)
        if m:
            return [IRCandidate(SpeechActIR(speaker='user', act_type='modal_nonassertive', content=raw), 0.96, self.name, notes=['scope_guard_nonassertive'])]

        # Modal/negation classification.
        m = re.match(r'^(.+?)\s+(?:is\s+unlikely\s+to\s+be|cannot\s+be\s+classified\s+as|can\s+not\s+be\s+classified\s+as|is\s+not\s+exactly)\s+(?:a\s+|an\s+|the\s+)?(.+)$', low)
        if m:
            return [IRCandidate(NegatedClaimIR(subject=norm(m.group(1)), relation='is', object=norm(m.group(2)), modality='uncertain_negative'), 0.96, self.name, notes=['modality_negation'])]

        # Belief questions with nested negative propositions.
        m = re.match(r'^does\s+(.+?)\s+(?:believe|think)\s+(?:that\s+)?(.+?)\s+is\s+not\s+(?:the\s+)?(.+?)\??$', low)
        if m:
            holder, subj, obj = m.groups()
            prop = NegatedClaimIR(subject=norm(subj), relation='is', object=norm(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=norm(holder), relation='believes', object=prop.text()), requested_mode='proof'), 0.98, self.name, notes=['belief_negative_question'])]
        m = re.match(r'^does\s+(.+?)\s+(?:believe|think)\s+(?:that\s+)?(.+?)\s+is\s+(?:the\s+)?(.+?)\??$', low)
        if m:
            holder, subj, obj = m.groups()
            prop = ClaimIR(subject=norm(subj), relation='is', object=norm(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=norm(holder), relation='believes', object=prop.text()), requested_mode='proof'), 0.98, self.name, notes=['belief_question'])]


        # Nested belief statements/questions.
        m = re.match(r'^does\s+(.+?)\s+(?:believe|think)\s+(.+?)\s+(?:believes|thinks)\s+(.+?)\s+is\s+(?:the\s+)?(.+?)\??$', low)
        if m:
            holder, inner_holder, subj, obj = m.groups()
            inner_prop = ClaimIR(subject=norm(subj), relation='is', object=norm(obj))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=norm(holder), relation='believes', object=f"{norm(inner_holder)} believes {inner_prop.text()}"), requested_mode='proof'), 0.98, self.name, notes=['nested_belief_question'])]
        m = re.match(r'^(.+?)\s+(?:believes|thinks)\s+(.+?)\s+(?:believes|thinks)\s+(.+?)\s+is\s+(?:the\s+)?(.+?)$', low)
        if m:
            holder, inner_holder, subj, obj = m.groups()
            inner_prop = ClaimIR(subject=norm(subj), relation='is', object=norm(obj))
            nested = ClaimIR(subject=norm(inner_holder), relation='believes', object=inner_prop.text())
            return [IRCandidate(BeliefIR(holder=norm(holder), proposition=nested), 0.97, self.name, notes=['nested_belief_statement'])]

        # Temporal interval/state algebra front-end.
        m = re.match(r'^(.+?)\s+(?:was|served\s+as|acted\s+as)\s+(?:the\s+)?(.+?)\s+(?:from|between)\s+([0-9]{4})\s+(?:to|through|and)\s+([0-9]{4})$', low)
        if m:
            subj, role, start, end = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=norm(subj), relation='is', object=norm(role), time_expr=f'{start}-{end}', valid_during=f'{start}-{end}', valid_from=start, valid_to=end), 0.98, self.name, notes=['temporal_interval_algebra'])]
        m = re.match(r'^(.+?)\s+was\s+not\s+(?:the\s+)?(.+?)\s+(?:during|in)\s+([0-9]{4})$', low)
        if m:
            subj, role, t = m.groups()
            return [IRCandidate(TemporalClaimIR(subject=norm(subj), relation='is', object=norm(role), polarity='negative', time_expr=t, valid_during=t, valid_from=t), 0.98, self.name, notes=['temporal_negative_interval'])]
        m = re.match(r'^who\s+(?:served\s+as|held\s+(?:the\s+)?(?:role\s+of)?|was)\s+(?:the\s+)?(.+?)\s+(?:during|in)\s+([0-9]{4})\??$', low)
        if m:
            role, t = m.groups()
            return [IRCandidate(QuestionIR(target=TemporalClaimIR(subject='?', relation='is', object=norm(role), time_expr=t, valid_during=t), requested_mode='answer'), 0.98, self.name, notes=['temporal_role_question_schema'])]

        # Multi-slot move/carry/transport event frame.
        m = re.match(r'^(.+?)\s+(?:carried|carries|transported|transports|moved|moves|relocated|relocates)\s+(?:the\s+)?(.+?)\s+from\s+(.+?)\s+to\s+(.+?)$', low)
        if m:
            actor, patient, src, dst = m.groups()
            return [IRCandidate(EventIR(actor=norm(actor), action='move', patient=norm(patient), location=norm(dst), time_expr=None), 0.97, self.name, notes=[f'source={norm(src)}','event_frame_move'])]
        m = re.match(r'^(?:where\s+is|where\s+was)\s+(?:the\s+)?(.+?)\??$', low)
        if m:
            obj = norm(m.group(1))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=obj, relation='located_at', object='?'), requested_mode='answer'), 0.98, self.name, notes=['where_query_fluent'])]
        m = re.match(r'^is\s+(?:the\s+)?(.+?)\s+(open|closed)\??$', low)
        if m:
            obj, state = m.groups()
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=norm(obj), relation='is', object=norm(state)), requested_mode='proof'), 0.97, self.name, notes=['state_query_alias'])]

        # Korean grammar operations.
        m = re.match(r'^(.+?)가\s+(.+?)보다\s+(?:더\s+)?우세하다고\s+봐도\s+되나\??$', s)
        if m:
            left, right = m.groups()
            return [IRCandidate(QuestionIR(target=ComparisonIR(left=norm(left), comparator='greater_than', right=norm(right)), requested_mode='proof'), 0.98, self.name, notes=['ko_question_modality'])]
        m = re.match(r'^(.+?)(?:는|은)?\s+(.+?)보다\s+크지\s+않은\s+것\s+같다$', s)
        if m:
            left, right = m.groups()
            return [IRCandidate(ComparisonIR(left=norm(left), comparator='less_than', right=norm(right)), 0.96, self.name, notes=['ko_uncertain_negation'])]
        m = re.match(r'^(.+?)(?:는|은)?\s+(.+?)에\s+비해\s+앞선다$', s)
        if m:
            left, right = m.groups()
            return [IRCandidate(ComparisonIR(left=norm(left), comparator='greater_than', right=norm(right)), 0.96, self.name, notes=['ko_comparison_marker'])]

        # Exception direct question for plural/singular.
        m = re.match(r'^can\s+(?:an?\s+|the\s+)?(.+?)\s+fly\??$', low)
        if m:
            subj = plural_singular(m.group(1))
            return [IRCandidate(QuestionIR(target=ClaimIR(subject=subj, relation='can', object='fly'), requested_mode='proof'), 0.95, self.name, notes=['exception_direct_question'])]
        return out

    @staticmethod
    def _target_string(ir: CognitiveIR) -> str:
        if isinstance(ir, ComparisonIR):
            return f'compare(A,{ir.comparator},B)'
        if isinstance(ir, CausalClaimIR):
            return 'causal(A,B)'
        if isinstance(ir, NegatedClaimIR):
            return 'not_claim(A,is,B)'
        if isinstance(ir, ClaimIR):
            return 'claim(A,is,B)'
        return ''
