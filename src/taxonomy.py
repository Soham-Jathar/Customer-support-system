"""Transparent hierarchy from BANKING77 fine intents to broad query types."""
from __future__ import annotations


QUERY_TYPES = (
    "cash_and_atm", "card_management", "card_payments", "transfers",
    "top_ups", "currency_exchange", "account_and_verification", "other",
)


def query_type_for(intent: str) -> str:
    label = intent.lower().replace("?", "")
    if "cash_withdrawal" in label or label == "atm_support" or "wrong_amount_of_cash" in label:
        return "cash_and_atm"
    if any(term in label for term in ("transfer", "beneficiary", "receiving_money")):
        return "transfers"
    if "top_up" in label or "topping_up" in label:
        return "top_ups"
    if any(term in label for term in ("exchange", "currency", "fiat")):
        return "currency_exchange"
    if any(term in label for term in ("identity", "passcode", "personal_details", "age_limit", "country_support", "terminate_account", "lost_or_stolen_phone", "source_of_funds")):
        return "account_and_verification"
    if any(term in label for term in ("card_payment", "direct_debit", "transaction_charged", "extra_charge", "request_refund", "refund_not", "reverted_card_payment", "pending_card_payment")):
        return "card_payments"
    if "card" in label or any(term in label for term in ("pin", "contactless", "apple_pay", "visa_or_mastercard", "compromised_card")):
        return "card_management"
    return "other"


def display_query_type(query_type: str) -> str:
    return query_type.replace("_", " ").title()
