"""Minimal and auditable text processing helpers."""
from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract operational IDs without claiming this is a full NER model."""
    text = normalize_text(text)
    return {
        "order_ids": re.findall(r"\b(?:order\s*(?:id|number)?\s*[:#-]?\s*)?((?!(?:TXN|REF)-?)[A-Z]{2,6}-?\d{4,12})\b", text, flags=re.I),
        "transaction_ids": re.findall(r"\b(?:txn|transaction|reference)\s*(?:id|number)?\s*[:#-]?\s*([A-Z0-9-]{5,20})\b", text, flags=re.I),
        "card_references": re.findall(r"\b(?:card\s*)?(?:ending|last\s*(?:four|4)|last\s*digits?)\s*(?:in\s*)?(\d{4})\b", text, flags=re.I),
        "amounts": re.findall(r"(?:₹|\$|€|£)\s?\d+(?:[,.]\d{1,2})?", text),
        "dates": re.findall(r"\b(?:today|yesterday|tomorrow|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b", text, flags=re.I),
    }


def mask_sensitive_data(text: str) -> str:
    """Redact common PII before persistence; do not use this for model inference."""
    value = normalize_text(text)
    value = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", "[EMAIL]", value)
    value = re.sub(r"\b(?:account\s*(?:number|no\.?|#)?\s*[:=-]?\s*)\d{6,18}\b", "account [REDACTED]", value, flags=re.I)
    value = re.sub(r"(?<!\d)(?:\d[ -]?){11,18}\d(?!\d)", "[SENSITIVE_NUMBER]", value)
    value = re.sub(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2,4}\d{3,4}(?!\w)", "[PHONE]", value)
    return value


def mask_entities(entities: dict[str, list[str]]) -> dict[str, list[str]]:
    """Retain only final four characters of stored operational references."""
    masked: dict[str, list[str]] = {}
    for name, values in entities.items():
        if name in {"transaction_ids", "order_ids"}:
            masked[name] = [f"***{value[-4:]}" if len(value) > 4 else "[REDACTED]" for value in values]
        else:
            masked[name] = values
    return masked
