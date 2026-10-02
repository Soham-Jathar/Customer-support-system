"""Deterministic follow-up questions for incomplete support requests."""
from __future__ import annotations


def next_clarification(result: dict, customer_text: str) -> str | None:
    entities = result.get("entities", {})
    intent = result.get("intent", "")
    if intent in {"delivery_delay", "missing_delivery", "track_order"} and not (entities.get("order_ids") or entities.get("tracking_ids")):
        return "Please share your order ID or tracking ID so the support team can check the correct shipment."
    if intent in {"payment_issue", "duplicate_charge", "payment_failed", "refund_pending"} and not (entities.get("transaction_ids") or entities.get("card_references")):
        return "Please share a transaction ID or the last four digits of the payment card. Do not share a full card number, password, or OTP."
    if intent.startswith("technical_") and not any(word in customer_text.lower() for word in ("error", "code", "app", "website", "device", "browser")):
        return "Which app, website, device, or browser is affected, and what error message do you see?"
    return None
