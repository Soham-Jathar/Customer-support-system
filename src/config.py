from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
MODEL_PATH = ROOT / "models" / "ticket_classifier.joblib"
QUERY_TYPE_MODEL_PATH = ROOT / "models" / "query_type_classifier.joblib"
LOW_CONFIDENCE_THRESHOLD = 0.55

def department_for(intent: str) -> str:
    """Map BANKING77's fine-grained labels to an auditable support queue."""
    label = intent.lower()
    if any(word in label for word in ("cash_withdrawal", "cash withdrawal", "atm")):
        return "Cash & ATM Operations"
    if any(word in label for word in ("card", "cash_withdrawal", "cash withdrawal")):
        return "Card & Payments Support"
    if any(word in label for word in ("transfer", "beneficiary", "bank_transfer")):
        return "Transfers Support"
    if any(word in label for word in ("exchange", "currency", "fiat", "crypto", "cash_deposit")):
        return "Accounts & Currency Support"
    if any(word in label for word in ("refund", "charged", "payment", "fee", "balance", "top_up")):
        return "Payments Support"
    if any(word in label for word in ("verify", "identity", "passcode", "pin", "password", "terminate", "age_limit", "country")):
        return "Account Security & Verification"
    return "General Banking Support"
