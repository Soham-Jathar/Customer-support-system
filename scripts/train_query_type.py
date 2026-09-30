"""Train a coarse query-type classifier from the BANKING77 fine-intent labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import QUERY_TYPE_MODEL_PATH  # noqa: E402
from src.taxonomy import query_type_for  # noqa: E402
from src.train import build_classifier, load_data  # noqa: E402


def main(data_path: str) -> None:
    data = load_data(data_path)
    data["query_type"] = data.intent.map(query_type_for)
    model = build_classifier().fit(data.text, data.query_type)
    QUERY_TYPE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, QUERY_TYPE_MODEL_PATH)
    print(f"Saved query-type model: {QUERY_TYPE_MODEL_PATH}")
    print(data.query_type.value_counts().sort_index().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "banking77" / "banking77_train.csv"))
    main(parser.parse_args().data)
