"""Evaluate the saved baseline on the reproducible held-out split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import MODEL_PATH  # noqa: E402
from src.train import load_data  # noqa: E402
import joblib


def calibration_metrics(model, texts, expected, predicted) -> dict:
    """Compute top-label expected calibration error without new dependencies."""
    probabilities = model.predict_proba(texts)
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predicted) == expected.to_numpy()
    ece = 0.0
    for lower, upper in zip(np.linspace(0, .9, 10), np.linspace(.1, 1, 10)):
        mask = (confidence >= lower) & (confidence <= upper if upper == 1 else confidence < upper)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {"mean_top_label_confidence": float(confidence.mean()), "expected_calibration_error": ece}


def main(data_path: str) -> None:
    data = load_data(data_path)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Train the baseline first.")
    model = joblib.load(MODEL_PATH)
    predicted = model.predict(data.text)
    metrics = {
        "dataset": str(Path(data_path)), "examples": len(data), "model": "TF-IDF + Logistic Regression",
        "accuracy": accuracy_score(data.intent, predicted),
        "macro_f1": f1_score(data.intent, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(data.intent, predicted, average="weighted", zero_division=0),
        "per_intent": classification_report(data.intent, predicted, zero_division=0, output_dict=True),
        **calibration_metrics(model, data.text, data.intent, predicted),
    }
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro F1:", round(metrics["macro_f1"], 4))
    labels = sorted(data.intent.unique())
    matrix = confusion_matrix(data.intent, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, xticks_rotation=35, colorbar=False)
    figure.tight_layout()
    output = ROOT / "outputs" / "generic_intent_confusion_matrix.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=160)
    metric_output = ROOT / "outputs" / "generic_intent_test_metrics.json"
    metric_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {output}")
    print(f"Saved {metric_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "generic" / "test.csv"))
    main(parser.parse_args().data)
