"""Transparent priority and escalation policy.

Sentiment alone must never determine priority.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.config import LOW_CONFIDENCE_THRESHOLD, department_for
from src.preprocess import normalize_text

CRITICAL = (
    r"\bunauthori[sz]ed\b", r"\bfraud(?:ulent)?\b", r"\b(?:account (?:is )?compromised|compromised my account)\b",
    r"\b(?:stolen (?:account|identity|payment)|identity (?:was )?stolen)\b", r"\bsuspicious (?:charge|payment|activity)\b", r"\bsecurity breach\b",
    r"\b(?:do not|don't|cannot|can't) recognize (?:this |the )?(?:charge|transaction|payment)\b",
)
HIGH = (r"\bcharged? (?:me )?twice\b", r"\bdouble (?:charge|charged)\b", r"\b(?:payment|checkout) (?:was )?(?:failed|declined)\b", r"\b(?:locked|lockout|locked out)\b", r"\b(?:urgent|urgently|emergency|immediately)\b", r"\b(?:not arrived|not delivered|still (?:hasn't|has not) arrived)\b", r"\b(?:delivered|delivery).{0,30}\b(?:missing|not here)\b")
HUMAN = (r"\b(?:human|real person|live agent|representative|supervisor)\b", r"\btalk to (?:an? )?(?:agent|person|human)\b", r"\bspeak to (?:an? )?(?:agent|person|human)\b")
VAGUE = (r"\bcannot explain\b", r"\bsomething confusing\b", r"\bnot sure (?:what|how)\b")
ROUTINE_INTENTS = {"payment_methods", "delivery_options", "delivery_address", "refund_policy", "product_information", "newsletter"}


@dataclass(frozen=True)
class TriageDecision:
    priority: str
    department: str
    escalate_to_human: bool
    escalation_reasons: list[str]
    explanation: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def make_decision(query_type: str, intent: str, confidence: float, sentiment: str, text: str, query_confidence: float | None = None) -> TriageDecision:
    text = normalize_text(text).lower()
    priority, escalate, reasons = "low", False, []
    explanation = [f"Fine-intent classifier selected '{intent}' with {confidence:.0%} confidence."]
    if _matches(text, CRITICAL):
        priority, escalate = "critical", True
        reasons.append("Security or suspected-fraud indicator detected")
        explanation.append("Critical risk language overrides the normal route.")
    elif _matches(text, HIGH):
        priority = "high"
        explanation.append("High-impact issue or explicit urgency indicator detected.")
    elif query_type in {"payment", "refund", "account_access", "technical_support"} and intent not in ROUTINE_INTENTS:
        priority = "medium"
        explanation.append("This complaint type requires timely specialist review.")
    elif sentiment == "negative":
        priority = "medium"
        explanation.append("Negative sentiment raises review priority but does not decide it alone.")
    else:
        explanation.append("No critical, high-impact, or negative-sentiment indicator was found.")
    # A low fine-intent score alone should not create needless agent work when
    # the independent broad query-type model is confident about the queue.
    low_routing_confidence = confidence < LOW_CONFIDENCE_THRESHOLD and (
        query_confidence is None or query_confidence < LOW_CONFIDENCE_THRESHOLD
    )
    if low_routing_confidence:
        escalate = True
        reasons.append(f"Low routing confidence (intent {confidence:.0%})")
        explanation.append("A human should verify routing because the classifier is uncertain.")
    if _matches(text, VAGUE):
        escalate = True
        reasons.append("Message lacks enough detail for safe automated routing")
        explanation.append("A human should clarify this vague request before taking action.")
    if _matches(text, HUMAN):
        escalate = True
        reasons.append("Customer explicitly requested a human agent")
        explanation.append("The explicit request for a human agent is honored.")
    department = "Human Escalation Queue" if escalate else department_for(query_type)
    return TriageDecision(priority, department, escalate, reasons, explanation)
