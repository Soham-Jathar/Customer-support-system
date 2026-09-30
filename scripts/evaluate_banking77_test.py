"""Evaluate the saved classifier on BANKING77's untouched official test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import MODEL_PATH  # noqa: E402
from src.train import load_data  # noqa: E402


def main(test_path: str) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}. Train it before evaluating.")
    test = load_data(test_path)
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(test.text)
    labels = list(model.classes_)
    report = classification_report(test.intent, predicted, labels=labels, zero_division=0, output_dict=True)
    summary = {
        "dataset": str(Path(test_path)), "examples": len(test), "intent_labels": len(labels),
        "accuracy": report["accuracy"],
        "macro_f1": f1_score(test.intent, predicted, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(test.intent, predicted, labels=labels, average="weighted", zero_division=0),
    }
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "banking77_test_metrics.json").write_text(json.dumps({**summary, "per_intent": report}, indent=2), encoding="utf-8")
    matrix = confusion_matrix(test.intent, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(24, 24))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, xticks_rotation=90, colorbar=False, values_format="d")
    axis.tick_params(axis="both", labelsize=5)
    figure.tight_layout()
    figure.savefig(outputs / "banking77_test_confusion_matrix.png", dpi=180)
    print(f"Official test examples: {summary['examples']:,}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}")
    print(f"Saved {outputs / 'banking77_test_metrics.json'}")
    print(f"Saved {outputs / 'banking77_test_confusion_matrix.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=str(ROOT / "data" / "banking77" / "banking77_test.csv"))
    main(parser.parse_args().test)
