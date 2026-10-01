"""Download Bitext and preserve a reproducible two-level intent taxonomy."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

URL = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

INTENT_MAP = {
    "payment_issue": ("payment", "payment_issue"),
    "check_payment_methods": ("payment", "payment_methods"),
    "delivery_options": ("delivery", "delivery_options"), "delivery_period": ("delivery", "delivery_delay"),
    "track_order": ("delivery", "track_order"), "change_shipping_address": ("delivery", "delivery_address"),
    "set_up_shipping_address": ("delivery", "delivery_address"),
    "check_refund_policy": ("refund", "refund_policy"), "get_refund": ("refund", "refund_request"), "track_refund": ("refund", "refund_pending"),
    "create_account": ("account_access", "account_creation"), "edit_account": ("account_access", "account_management"),
    "switch_account": ("account_access", "account_management"), "delete_account": ("account_access", "account_management"),
    "recover_password": ("account_access", "account_access"), "registration_problems": ("account_access", "account_access"),
    "contact_human_agent": ("other", "human_request"), "contact_customer_service": ("other", "human_request"),
    "complaint": ("other", "general_complaint"), "cancel_order": ("other", "order_cancellation"),
    "change_order": ("other", "order_change"), "check_cancellation_fee": ("other", "cancellation_fee"),
    "check_invoice": ("other", "invoice_request"), "get_invoice": ("other", "invoice_request"),
    "newsletter_subscription": ("other", "newsletter"), "place_order": ("other", "product_information"), "review": ("other", "product_information"),
}


def main(output: str, per_intent: int) -> None:
    raw = pd.read_csv(URL)
    raw.columns = [column.lower() for column in raw.columns]
    required = {"instruction", "intent"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected Bitext columns: {raw.columns.tolist()}")
    mapped_labels = raw["intent"].map(INTENT_MAP)
    raw["query_type"] = mapped_labels.map(lambda value: value[0] if isinstance(value, tuple) else None)
    raw["intent"] = mapped_labels.map(lambda value: value[1] if isinstance(value, tuple) else None)
    mapped = raw.dropna(subset=["intent"]).rename(columns={"instruction": "text"})[["text", "query_type", "intent"]]
    mapped = mapped.drop_duplicates("text")
    balanced = (mapped.groupby("intent", group_keys=False).sample(n=per_intent, random_state=42, replace=False).reset_index(drop=True))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(path, index=False)
    print(f"Saved {len(balanced)} mapped Bitext records to {path}")
    print(balanced.query_type.value_counts().sort_index().to_string())
    print("Note: Bitext has no dedicated technical_support class. Add the project-owned technical subset before training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/bitext_mapped.csv")
    parser.add_argument("--per-intent", type=int, default=240, help="Maximum examples per fine intent")
    args = parser.parse_args()
    main(args.output, args.per_intent)
