"""Create project-owned, training-only paraphrases for realistic support input.

The public corpus is useful for intent coverage but contains phrasing that can
differ from short, natural messages typed into the demo.  These examples are
explicitly synthetic augmentation.  They are appended only to training data;
the held-out test split remains untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXAMPLES = {
    "delivery_delay": [
        "My laptop order has not arrived after five days.", "I ordered a product last week and it is still not here.",
        "The delivery is late and I need my package urgently.", "Where is my order? It has been delayed for days.",
        "My parcel has not arrived by the promised delivery period.", "I need an update because the order has still not been delivered.",
        "Order ORD-12345 is late and I need it as soon as possible.", "My delivery has been delayed without an explanation.",
    ],
    "track_order": [
        "How can I track order ORD-12345?", "Please show me the current shipment tracking status.",
        "Where can I find my tracking number?", "I want to know where my parcel is right now.",
        "Can you help me follow the delivery for my order?", "The tracking page is unclear; where is my package?",
    ],
    "refund_pending": [
        "I requested a refund but it is still pending.", "My refund has not reached my payment method yet.",
        "When will the pending refund for order ORD-78901 be processed?", "The return was accepted but I have not received my refund.",
        "Please check the status of my refund; it has taken too long.", "Why is my refund still being processed?",
    ],
    "refund_request": [
        "I want a refund for the item I ordered.", "How do I request a refund for order ORD-78901?",
        "The product is not suitable and I would like my money back.", "Can I return this order and receive a refund?",
        "Please explain how to start a refund request.", "I need to cancel this purchase and request a refund.",
    ],
    "payment_issue": [
        "I do not recognize a payment of $250 from my account.", "There is an unfamiliar charge on my payment statement.",
        "My card was charged twice for the same order.", "Payment failed when I tried to complete my order.",
        "I paid for my order but the payment shows an error.", "Why was I charged an extra amount at checkout?",
        "The checkout payment was declined even though my details are correct.", "Please investigate this unexpected payment charge.",
    ],
    "account_access": [
        "I cannot access my account after resetting my password.", "The website will not let me sign in.",
        "My account is locked and I need to regain access.", "I forgot my password and the recovery link is not working.",
        "I cannot log into the customer portal.", "Please help me unlock my account.",
    ],
    "technical_checkout_error": [
        "The mobile app crashes whenever I try to check out.", "Checkout freezes when I press place order.",
        "I get an error while completing the checkout process.", "The app closes as soon as I open my shopping basket.",
        "The checkout screen will not load in the mobile app.", "A technical issue stops me from placing my order.",
    ],
    "technical_app_crash": [
        "The mobile app keeps crashing when I view my orders.", "My shopping app closes unexpectedly every time I open it.",
        "The app freezes and then shuts down.", "I cannot use the app because it crashes immediately.",
        "The latest app update made the application unusable.", "The app becomes unresponsive after I sign in.",
    ],
}


def main(output: str) -> None:
    rows = []
    for intent, texts in EXAMPLES.items():
        query_type = {
            "delivery_delay": "delivery", "track_order": "delivery", "refund_pending": "refund",
            "refund_request": "refund", "payment_issue": "payment", "account_access": "account_access",
            "technical_checkout_error": "technical_support", "technical_app_crash": "technical_support",
        }[intent]
        rows.extend({"text": text, "query_type": query_type, "intent": intent} for text in texts)
    data = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    print(f"Saved {len(data)} realistic, training-only paraphrases to {destination}")
    print(data.intent.value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/realistic_paraphrase_extension.csv")
    main(parser.parse_args().output)
