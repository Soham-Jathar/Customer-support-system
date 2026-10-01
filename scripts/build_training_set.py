"""Build reproducible generic train/test splits from public and annotated data."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.taxonomy import query_type_for


LEGACY_INTENTS = {
    "payment": "payment_issue", "delivery": "delivery_delay", "refund": "refund_request",
    "account_access": "account_access", "technical_support": "technical_website_error", "other": "general_complaint",
}


def normalise_schema(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "query_type" not in frame:
        frame["query_type"] = frame["intent"].map(query_type_for)
        frame.loc[frame["query_type"] == "other", "query_type"] = frame.loc[frame["query_type"] == "other", "intent"]
    frame["intent"] = frame["intent"].map(lambda value: LEGACY_INTENTS.get(value, value))
    return frame[["text", "query_type", "intent"]]


def main(bitext_path: str, technical_path: str, realistic_path: str, output: str, test_output: str) -> None:
    root = ROOT
    parts = [pd.read_csv(root / "data" / "demo_tickets.csv"), pd.read_csv(root / "data" / "annotated_extension.csv")]
    optional = Path(bitext_path)
    if optional.exists():
        parts.append(pd.read_csv(optional))
    else:
        print("Bitext mapped file not found; building the annotated-only development set.")
    technical = Path(technical_path)
    if technical.exists():
        parts.append(pd.read_csv(technical))
    else:
        print("Technical extension not found; run scripts/create_technical_extension.py before final training.")
    merged = pd.concat([normalise_schema(part) for part in parts], ignore_index=True).drop_duplicates("text").reset_index(drop=True)
    missing = set(("payment", "delivery", "refund", "account_access", "technical_support", "other")) - set(merged.query_type)
    if missing:
        raise ValueError(f"Training data is missing query types: {sorted(missing)}")
    # Group sampling keeps every fine intent represented in the held-out set
    # without making this preparation script depend on the training library.
    test = merged.groupby("intent", group_keys=False).sample(frac=.20, random_state=42)
    train = merged.drop(index=test.index)
    realistic = Path(realistic_path)
    if realistic.exists():
        # Keep synthetic paraphrases out of the held-out set so reported
        # metrics are not inflated by near-duplicate augmentation templates.
        train = pd.concat([train, normalise_schema(pd.read_csv(realistic))], ignore_index=True).drop_duplicates("text")
    else:
        print("Realistic paraphrase extension not found; run scripts/create_realistic_paraphrases.py for stronger demo robustness.")
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(destination, index=False)
    test_destination = Path(test_output)
    test_destination.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(test_destination, index=False)
    print(f"Saved {len(train)} training rows to {destination}")
    print(f"Saved {len(test)} held-out rows to {test_destination}")
    print(train.query_type.value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitext", default="data/processed/bitext_mapped.csv")
    parser.add_argument("--technical", default="data/technical_support_extension.csv")
    parser.add_argument("--realistic", default="data/realistic_paraphrase_extension.csv")
    parser.add_argument("--output", default="data/generic/train.csv")
    parser.add_argument("--test-output", default="data/generic/test.csv")
    args = parser.parse_args()
    main(args.bitext, args.technical, args.realistic, args.output, args.test_output)
