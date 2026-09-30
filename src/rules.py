"""Transparent priority and escalation policy.

Sentiment alone must never determine priority.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.config import LOW_CONFIDENCE_THRESHOLD, department_for
from src.preprocess import normalize_text

CRITICAL = (
    r"\bunauthori[sz]ed\b", r"\bfraud(?:ulent)?\b", r"\baccount (?:is )?compromised\b",
    r"\bstolen (?:card|account|phone)\b", r"\bsuspicious transaction\b", r"\bsecurity breach\b",
    r"\b(?:do not|don't|cannot|can't) recognize (?:this |the )?(?:charge|transaction|payment|cash withdrawal)\b",
)
HIGH = (r"\bcharged? (?:me )?twice\b", r"\bdouble (?:charge|charged)\b", r"\b(?:card )?payment (?:was )?(?:failed|declined)\b", r"\b(?:locked|lockout|locked out)\b", r"\b(?:urgent|urgently|emergency|immediately)\b", r"\bnot (?:arrived|delivered)\b")
HUMAN = (r"\b(?:human|real person|live agent|representative|supervisor)\b", r"\btalk to (?:an? )?(?:agent|person|human)\b", r"\bspeak to (?:an? )?(?:agent|person|human)\b")


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


def make_decision(intent: str, confidence: float, sentiment: str, text: str) -> TriageDecision:
    text = normalize_text(text).lower()
    priority, escalate, reasons = "low", False, []
    explanation = [f"Intent classifier selected '{intent}' with {confidence:.0%} confidence."]
    if _matches(text, CRITICAL):
        priority, escalate = "critical", True
        reasons.append("Security or suspected-fraud indicator detected")
        explanation.append("Critical risk language overrides the normal route.")
    elif _matches(text, HIGH):
        priority = "high"
        explanation.append("High-impact issue or explicit urgency indicator detected.")
    elif any(term in intent.lower() for term in ("card", "payment", "transfer", "cash_withdrawal", "cash withdrawal", "verify", "passcode", "pin")):
        priority = "medium"
        explanation.append("This complaint type requires timely specialist review.")
    elif sentiment == "negative":
        priority = "medium"
        explanation.append("Negative sentiment raises review priority but does not decide it alone.")
    else:
        explanation.append("No critical, high-impact, or negative-sentiment indicator was found.")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        escalate = True
        reasons.append(f"Low intent confidence ({confidence:.0%})")
        explanation.append("A human should verify routing because the classifier is uncertain.")
    if _matches(text, HUMAN):
        escalate = True
        reasons.append("Customer explicitly requested a human agent")
        explanation.append("The explicit request for a human agent is honored.")
    department = "Human Escalation Queue" if escalate else department_for(intent)
    return TriageDecision(priority, department, escalate, reasons, explanation)
