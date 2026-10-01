"""Evaluate the separate broad query-type classifier on the held-out split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import QUERY_TYPE_MODEL_PATH  # noqa: E402
from src.taxonomy import query_type_for  # noqa: E402
from src.train import load_data  # noqa: E402


def main(test_path: str) -> None:
    data = load_data(test_path)
    expected = data["query_type"] if "query_type" in data else data.intent.map(query_type_for)
    model = joblib.load(QUERY_TYPE_MODEL_PATH)
    predicted = model.predict(data.text)
    metrics = {
        "examples": len(data), "query_types": len(model.classes_),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, average="macro", zero_division=0),
        "per_type": classification_report(expected, predicted, zero_division=0, output_dict=True),
    }
    output = ROOT / "outputs" / "query_type_test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Query-type accuracy: {metrics['accuracy']:.4f}")
    print(f"Query-type macro F1: {metrics['macro_f1']:.4f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=str(ROOT / "data" / "generic" / "test.csv"))
    main(parser.parse_args().test)
