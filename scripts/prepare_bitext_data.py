"""Download Bitext, map its detailed intents to this project's coarse intent taxonomy, and create a balanced training CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

INTENT_MAP = {
    "payment_issue": "payment",
    "delivery_options": "delivery", "delivery_period": "delivery", "track_order": "delivery",
    "change_shipping_address": "delivery", "set_up_shipping_address": "delivery",
    "check_refund_policy": "refund", "get_refund": "refund", "track_refund": "refund",
    "create_account": "account_access", "edit_account": "account_access", "switch_account": "account_access",
    "delete_account": "account_access", "recover_password": "account_access", "registration_problems": "account_access",
    "contact_human_agent": "other", "contact_customer_service": "other", "complaint": "other",
    "cancel_order": "other", "change_order": "other", "check_cancellation_fee": "other",
    "check_invoice": "other", "check_payment_methods": "other", "get_invoice": "other",
    "newsletter_subscription": "other", "place_order": "other", "review": "other",
}


def main(output: str, per_intent: int) -> None:
    raw = pd.read_csv(URL)
    raw.columns = [column.lower() for column in raw.columns]
    required = {"instruction", "intent"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected Bitext columns: {raw.columns.tolist()}")
    raw["intent"] = raw["intent"].map(INTENT_MAP)
    mapped = raw.dropna(subset=["intent"]).rename(columns={"instruction": "text"})[["text", "intent"]]
    mapped = mapped.drop_duplicates("text")
    balanced = (mapped.groupby("intent", group_keys=False).sample(n=per_intent, random_state=42, replace=False).reset_index(drop=True))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(path, index=False)
    print(f"Saved {len(balanced)} mapped Bitext records to {path}")
    print(balanced.intent.value_counts().sort_index().to_string())
    print("Note: Bitext has no dedicated technical_support intent. Add the manually annotated technical subset before final training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/bitext_mapped.csv")
    parser.add_argument("--per-intent", type=int, default=500)
    args = parser.parse_args()
    main(args.output, args.per_intent)
