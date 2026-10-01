"""Append validated human feedback to training data, never to the held-out test set."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(base_path: str, feedback_path: str, output_path: str) -> None:
    base = pd.read_csv(base_path)
    feedback = pd.read_csv(feedback_path)
    required = {"text", "query_type", "intent"}
    if not required.issubset(base.columns) or not required.issubset(feedback.columns):
        raise ValueError("Both files must contain text, query_type, and intent columns.")
    combined = pd.concat([base.assign(source="original_training"), feedback], ignore_index=True)
    combined = combined.drop_duplicates("text", keep="last").sample(frac=1, random_state=42).reset_index(drop=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(destination, index=False)
    print(f"Saved {len(combined)} retraining rows to {destination}")
    print(f"Included {len(feedback)} validated human-review rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/generic/train.csv")
    parser.add_argument("--feedback", default="data/feedback/validated_agent_feedback.csv")
    parser.add_argument("--output", default="data/retraining/train_with_feedback.csv")
    args = parser.parse_args()
    main(args.base, args.feedback, args.output)
