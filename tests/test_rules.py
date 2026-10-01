from src.rules import make_decision


def test_fraud_always_escalates_as_critical():
    decision = make_decision("payment", "payment_issue", .98, "neutral", "I politely report an unauthorized payment.")
    assert decision.priority == "critical"
    assert decision.escalate_to_human
    assert decision.department == "Human Escalation Queue"


def test_unrecognized_transaction_is_treated_as_suspected_fraud():
    decision = make_decision("payment", "payment_issue", .80, "neutral", "I do not recognize the payment.")
    assert decision.priority == "critical"
    assert decision.escalate_to_human


def test_human_request_escalates():
    decision = make_decision("delivery", "track_order", .90, "neutral", "Please connect me to a human agent.")
    assert decision.escalate_to_human
    assert "Customer explicitly requested a human agent" in decision.escalation_reasons


def test_low_confidence_escalates():
    decision = make_decision("other", "general_complaint", .30, "neutral", "I need help with something confusing.")
    assert decision.escalate_to_human


def test_confident_broad_queue_prevents_unnecessary_escalation():
    decision = make_decision("delivery", "track_order", .30, "neutral", "Where is my order?", .92)
    assert not decision.escalate_to_human


def test_positive_sentiment_is_not_automatically_high_priority():
    decision = make_decision("other", "general_complaint", .90, "positive", "Thanks, your team has been great.")
    assert decision.priority == "low"
