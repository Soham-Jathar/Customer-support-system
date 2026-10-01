"""Validate exported human reviews before they can enter a retraining dataset.

This script deliberately does not train a model.  It keeps only complete human
corrections, normalises text, removes duplicates, and records provenance so an
agent's notes never silently contaminate the held-out evaluation split.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocess import normalize_text
from src.taxonomy import QUERY_TYPES, query_type_for


def main(input_path: str, output_path: str) -> None:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Feedback export not found: {source}")
    data = pd.read_csv(source)
    required = {"customer_message", "final_query_type", "final_intent"}
    if not required.issubset(data.columns):
        raise ValueError(f"Feedback CSV must contain {sorted(required)}")
    clean = data[["customer_message", "final_query_type", "final_intent"]].dropna().copy()
    clean.columns = ["text", "query_type", "intent"]
    clean["text"] = clean.text.map(normalize_text)
    clean["query_type"] = clean.query_type.astype(str).str.strip().str.lower()
    clean["intent"] = clean.intent.astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    clean = clean[(clean.text.str.len() >= 3) & clean.query_type.isin(QUERY_TYPES)]
    consistent = clean.intent.map(query_type_for) == clean.query_type
    rejected = int((~consistent).sum())
    clean = clean[consistent].drop_duplicates("text").assign(source="human_review")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(destination, index=False)
    print(f"Saved {len(clean)} validated human-review labels to {destination}")
    print(f"Rejected {rejected} rows with inconsistent query-type/intent labels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV downloaded from Agent Console")
    parser.add_argument("--output", default="data/feedback/validated_agent_feedback.csv")
    args = parser.parse_args()
    main(args.input, args.output)
