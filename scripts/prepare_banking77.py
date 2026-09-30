"""Download BANKING77's official split and export reproducible project CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def export_split(split, destination: Path) -> None:
    names = split.features["label"].names
    frame = pd.DataFrame({"text": split["text"], "intent": [names[index] for index in split["label"]]})
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"Saved {len(frame):,} rows and {frame.intent.nunique()} labels to {destination}")


def main(output_dir: str) -> None:
    # BANKING77's official repository supplies a dataset-loading script. This
    # opt-in is required by `datasets` 3.x; it does not grant access beyond
    # downloading and parsing the public dataset selected by the user.
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    output = Path(output_dir)
    export_split(dataset["train"], output / "banking77_train.csv")
    export_split(dataset["test"], output / "banking77_test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "banking77"))
    main(parser.parse_args().output_dir)
