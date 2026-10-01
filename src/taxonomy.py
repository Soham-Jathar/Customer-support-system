"""Project taxonomy for hierarchical customer-support classification."""
from __future__ import annotations


QUERY_TYPES = ("payment", "delivery", "refund", "account_access", "technical_support", "other")

FINE_INTENT_TO_QUERY_TYPE = {
    "payment_issue": "payment", "duplicate_charge": "payment", "payment_failed": "payment", "payment_methods": "payment",
    "delivery_options": "delivery", "delivery_delay": "delivery", "track_order": "delivery", "delivery_address": "delivery", "missing_delivery": "delivery",
    "refund_policy": "refund", "refund_request": "refund", "refund_pending": "refund",
    "account_access": "account_access", "account_creation": "account_access", "account_management": "account_access",
    "technical_login_error": "technical_support", "technical_app_crash": "technical_support", "technical_checkout_error": "technical_support",
    "technical_website_error": "technical_support", "technical_device_setup": "technical_support", "technical_product_malfunction": "technical_support",
    "human_request": "other", "general_complaint": "other", "order_cancellation": "other", "order_change": "other",
    "cancellation_fee": "other", "invoice_request": "other", "product_information": "other", "newsletter": "other",
}


def query_type_for(intent: str) -> str:
    """Return a broad category for a fine intent, safely defaulting to other."""
    return FINE_INTENT_TO_QUERY_TYPE.get(str(intent).strip().lower(), "other")


def display_query_type(query_type: str) -> str:
    return query_type.replace("_", " ").title()
