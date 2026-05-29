from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
from .ir import ClaimIR, EventIR, SpeechActIR


@dataclass
class FrameEffect:
    claim: ClaimIR
    reason: str


class EventWorldGrounder:
    """Small grounded frame semantics for event -> world-state claim updates."""
    def __init__(self):
        self.dynamic_frames = {}

    def add_dynamic_frame(self, name: str, effect: dict, source: str = "seed", confidence: float = 0.35):
        self.dynamic_frames[(name or "").lower()] = {"effect": effect, "source": source, "confidence": confidence}

    def effects_for(self, event: EventIR | SpeechActIR) -> List[FrameEffect]:
        if isinstance(event, SpeechActIR):
            return self._speech_effects(event)
        return self._event_effects(event)

    def _event_effects(self, e: EventIR) -> List[FrameEffect]:
        action = (e.action or "").lower().strip()
        out: List[FrameEffect] = []
        if action in {"give", "hand", "send", "transfer"} and e.recipient and e.patient:
            out.append(FrameEffect(ClaimIR(subject=e.recipient, relation="has", object=e.patient, confidence=e.confidence), "transfer_frame:recipient_has_patient"))
            if e.actor:
                out.append(FrameEffect(ClaimIR(subject=e.actor, relation="gave", object=e.patient, confidence=e.confidence), "transfer_frame:actor_gave_patient"))
        if action in {"receive", "borrow"} and e.actor and e.patient:
            out.append(FrameEffect(ClaimIR(subject=e.actor, relation="has", object=e.patient, confidence=e.confidence), "receive_frame:receiver_has_patient"))
        if action in {"buy", "purchase", "take"} and e.actor and e.patient:
            out.append(FrameEffect(ClaimIR(subject=e.actor, relation="has", object=e.patient, confidence=e.confidence), f"{action}_frame:buyer_or_taker_has_patient"))
        if action in {"sell"} and e.recipient and e.patient:
            out.append(FrameEffect(ClaimIR(subject=e.recipient, relation="has", object=e.patient, confidence=e.confidence), "sell_frame:buyer_has_patient"))
        if action in {"open", "close", "move", "put", "collect"} and e.patient:
            if action in {"move", "put"} and e.location:
                out.append(FrameEffect(ClaimIR(subject=e.patient, relation="located_at", object=e.location, confidence=e.confidence), f"{action}_frame:patient_location"))
            else:
                state = {"open": "open", "close": "closed", "move": "moved", "put": "placed", "collect": "collected"}[action]
                out.append(FrameEffect(ClaimIR(subject=e.patient, relation="state", object=state, confidence=e.confidence), f"{action}_frame:patient_state"))
                # Natural predicate alias: Is door open? should match open-event effects.
                if action in {"open", "close"}:
                    out.append(FrameEffect(ClaimIR(subject=e.patient, relation="is", object=state, confidence=e.confidence), f"{action}_frame:natural_state_alias"))
        if action in self.dynamic_frames:
            spec = self.dynamic_frames[action]["effect"]
            subject = None
            obj = spec.get("object")
            if "recipient_role" in spec:
                subject = getattr(e, spec["recipient_role"], None)
            if "subject_role" in spec:
                subject = getattr(e, spec["subject_role"], None)
            if "object_role" in spec:
                obj = getattr(e, spec["object_role"], None)
            if subject and obj:
                out.append(FrameEffect(ClaimIR(subject=subject, relation=spec.get("relation", "has"), object=obj, confidence=max(e.confidence, self.dynamic_frames[action].get("confidence", 0.35))), f"dynamic_frame:{action}"))
        return out

    def _speech_effects(self, s: SpeechActIR) -> List[FrameEffect]:
        # A request/order does not make the action true, but records an intention/obligation-like state.
        out: List[FrameEffect] = []
        content = getattr(s.content, "text", lambda: str(s.content))()
        if s.act_type in {"request", "order", "ask"}:
            out.append(FrameEffect(ClaimIR(subject=s.speaker, relation="requested", object=content, confidence=s.confidence), "speech_act_frame:request_record"))
        return out
