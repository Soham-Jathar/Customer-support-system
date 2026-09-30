"""Combine public mapped data with the project-owned annotated extension."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(bitext_path: str, output: str) -> None:
    root = Path(__file__).resolve().parents[1]
    parts = [pd.read_csv(root / "data" / "demo_tickets.csv"), pd.read_csv(root / "data" / "annotated_extension.csv")]
    optional = Path(bitext_path)
    if optional.exists():
        parts.append(pd.read_csv(optional))
    else:
        print("Bitext mapped file not found; building the annotated-only development set.")
    merged = pd.concat(parts, ignore_index=True).drop_duplicates("text").sample(frac=1, random_state=42).reset_index(drop=True)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    print(f"Saved {len(merged)} rows to {destination}")
    print(merged.intent.value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitext", default="data/processed/bitext_mapped.csv")
    parser.add_argument("--output", default="data/processed/training_tickets.csv")
    args = parser.parse_args()
    main(args.bitext, args.output)
