from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class SocialState:
    user_mood: str = 'neutral'
    confusion: float = 0.0
    rapport: float = 0.5
    trust: float = 0.5
    current_topic: str = ''
    history: List[str] = field(default_factory=list)

@dataclass
class ResponsePlan:
    primary_act: str
    content: str
    tone: str = 'concise_warm'
    expected_state_change: Dict[str, float] = field(default_factory=dict)
    risk: float = 0.0

class UnifiedActionSelector:
    """One selector for reasoning, repair, support, and small talk with social-state transitions."""
    def __init__(self):
        self.social_state = SocialState()

    def plan_support(self, state: str = 'confused') -> ResponsePlan:
        self.social_state.confusion = min(1.0, self.social_state.confusion + 0.2)
        # State-based response: match response to specific disclosure type
        if self.social_state.user_mood in ('sad', 'tired', 'upset'):
            content = "I hear you. Let's take it step by step. What is the one thing you want to figure out most right now?"
        elif self.social_state.user_mood == 'overwhelmed':
            content = "That is a lot. Let's pick the smallest piece and work through it together."
        else:
            content = "I can help. Let's make it concrete: state the goal, list the known facts, then test one small next step."
        return ResponsePlan(
            primary_act='empathize_and_structure',
            content=content,
            expected_state_change={'confusion': -0.2, 'trust': 0.05},
        )

    def plan_smalltalk(self, text: str) -> ResponsePlan:
        self.social_state.rapport = min(1.0, self.social_state.rapport + 0.05)
        low = text.lower()
        if any(w in low for w in ['haha', 'lol', 'hilarious', 'wild', 'funny']):
            content = "Glad it's landing. Want to turn it into something we test next?"
        elif 'nice' in low:
            content = "Cool. What should we do next?"
        elif 'rough' in low or 'tired' in low or 'sad' in low:
            self.social_state.user_mood = 'tired'
            content = "That sounds rough. We can pause or keep going if you prefer."
        else:
            content = "Right. Want to keep moving forward?"
        return ResponsePlan(primary_act='light_social_ack', content=content, expected_state_change={'rapport': 0.05})

    def render(self, plan: ResponsePlan) -> str:
        return plan.content
