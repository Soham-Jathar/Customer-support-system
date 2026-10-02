"""Mock operational-reference verification.

This module deliberately verifies only whether a supplied reference exists in
the local demonstration records. It never treats text classification or a
record match as proof that the requester owns the record.
"""
from __future__ import annotations

import json
from functools import lru_cache

from src.config import ROOT


def _reference(value: str) -> str:
    return str(value or "").upper().replace(" ", "")


@lru_cache(maxsize=1)
def _records() -> dict:
    return json.loads((ROOT / "data" / "mock_operational_records.json").read_text(encoding="utf-8"))


def verify_operational_references(entities: dict[str, list[str]]) -> dict:
    """Return an auditable, non-identity verification result for a ticket."""
    records = _records()
    order_index = {_reference(item["reference"]): item for item in records["orders"]}
    tracking_index = {_reference(item["tracking_reference"]): item for item in records["orders"] if item.get("tracking_reference")}
    transaction_index = {_reference(item["reference"]): item for item in records["transactions"]}
    checked: list[dict] = []

    for reference in entities.get("order_ids", []):
        record = order_index.get(_reference(reference))
        checked.append({"type": "order", "reference": _reference(reference), "found": bool(record), "record_status": record.get("status") if record else None})
    for reference in entities.get("tracking_ids", []):
        record = tracking_index.get(_reference(reference))
        checked.append({"type": "tracking", "reference": _reference(reference), "found": bool(record), "record_status": record.get("status") if record else None})
    for reference in entities.get("transaction_ids", []):
        record = transaction_index.get(_reference(reference))
        checked.append({"type": "transaction", "reference": _reference(reference), "found": bool(record), "record_status": record.get("status") if record else None})

    if not checked:
        return {
            "status": "no_reference_provided",
            "label": "No operational reference supplied",
            "summary": "No order, tracking, or transaction reference was available for a record lookup.",
            "requires_manual_verification": False,
            "checked_references": [],
        }
    missing = [item for item in checked if not item["found"]]
    found = [item for item in checked if item["found"]]
    if missing:
        return {
            "status": "reference_not_found",
            "label": "Reference not found — manual review required",
            "summary": "At least one supplied reference was not found in the demonstration company records. The request must be checked by an agent.",
            "requires_manual_verification": True,
            "checked_references": checked,
        }
    return {
        "status": "record_found_ownership_required",
        "label": "Record found — ownership check required",
        "summary": "A matching demonstration record was found. Customer ownership still requires secure account-side verification.",
        "requires_manual_verification": True,
        "checked_references": found,
    }
