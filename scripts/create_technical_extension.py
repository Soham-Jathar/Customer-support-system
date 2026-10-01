"""Create a transparent, project-owned technical-support annotation extension.

Bitext's customer-support set does not contain a dedicated technical-support
queue.  This deterministic seed set fills that documented gap.  It is intended
for a prototype and should be manually expanded with real approved tickets
before production use.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TEMPLATES = {
    "technical_login_error": [
        "I cannot sign in to the {surface}; it says {error}.",
        "The {surface} will not let me log in after I enter my email.",
        "I receive a {error} message whenever I try to access my account on the {surface}.",
        "Login is not working on the {surface}; please help me regain access.",
        "The {surface} keeps rejecting my correct login details.",
        "Why does the {surface} show {error} when I sign in?",
    ],
    "technical_app_crash": [
        "The {surface} crashes when I open my order history.",
        "My {surface} closes unexpectedly after the latest update.",
        "The {surface} freezes and then shuts down before I can use it.",
        "I cannot use the {surface} because it keeps crashing.",
        "Opening the {surface} causes a crash on my device.",
        "The {surface} becomes unresponsive every time I view an order.",
    ],
    "technical_checkout_error": [
        "I get {error} when I try to check out on the {surface}.",
        "The checkout page will not complete my order in the {surface}.",
        "I cannot place my order because checkout freezes on the {surface}.",
        "The {surface} shows {error} after I press place order.",
        "Checkout is broken and will not accept my selection on the {surface}.",
        "Please fix the checkout error in the {surface}; I cannot finish my order.",
    ],
    "technical_website_error": [
        "The {surface} page displays {error} instead of my account.",
        "I see a broken page in the {surface} when I search for a product.",
        "The {surface} is not loading correctly in my browser.",
        "A technical error prevents me from using the {surface}.",
        "Buttons on the {surface} do nothing when I click them.",
        "The {surface} keeps showing {error}; can technical support investigate?",
    ],
    "technical_device_setup": [
        "I need help setting up my {product} with the {surface}.",
        "The {product} will not connect to the {surface}.",
        "How can I configure my new {product} in the {surface}?",
        "Setup for the {product} fails with {error}.",
        "My {product} is not recognised by the {surface}.",
        "I cannot complete device setup for the {product}.",
    ],
    "technical_product_malfunction": [
        "My {product} stopped working after delivery.",
        "The {product} is defective and does not operate as described.",
        "I need technical help because my new {product} will not turn on.",
        "The {product} has a fault and I need troubleshooting steps.",
        "Something is wrong with the {product}; it fails during normal use.",
        "Please help diagnose a malfunction with my {product}.",
    ],
}

CONTEXTS = [
    {"surface": "mobile app", "error": "an unexpected error", "product": "smart speaker"},
    {"surface": "website", "error": "error code E500", "product": "wireless printer"},
    {"surface": "customer portal", "error": "a blank screen", "product": "fitness tracker"},
    {"surface": "shopping app", "error": "a server error", "product": "robot vacuum"},
    {"surface": "web store", "error": "error code C102", "product": "security camera"},
    {"surface": "order page", "error": "a loading error", "product": "Bluetooth headset"},
]


def main(output: str) -> None:
    rows = []
    for intent, templates in TEMPLATES.items():
        for template in templates:
            for context in CONTEXTS:
                rows.append({"text": template.format(**context), "query_type": "technical_support", "intent": intent})
    frame = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame)} project-owned technical-support examples to {destination}")
    print(frame.intent.value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/technical_support_extension.csv")
    main(parser.parse_args().output)
