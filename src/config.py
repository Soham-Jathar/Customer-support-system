"""Central project configuration and transparent support-queue mapping."""
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Separate names prevent an old banking model from accidentally being served by
# the general customer-support application.
MODEL_PATH = ROOT / "models" / "generic_intent_classifier.joblib"
QUERY_TYPE_MODEL_PATH = ROOT / "models" / "generic_query_type_classifier.joblib"
# Broad routing is independently evaluated. A ticket is escalated for model
# uncertainty only when both fine and broad classifiers are below this value.
LOW_CONFIDENCE_THRESHOLD = 0.45

DEPARTMENTS = {
    "payment": "Payments Support",
    "delivery": "Logistics Support",
    "refund": "Returns & Refunds",
    "account_access": "Account Support",
    "technical_support": "Technical Support",
    "other": "General Support",
}


def department_for(query_type: str) -> str:
    """Map a broad, auditable query type to the owning support department."""
    return DEPARTMENTS.get(query_type, DEPARTMENTS["other"])
