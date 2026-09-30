"""Evaluate the baseline on a held-out stratified split."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import build_classifier, load_data  # noqa: E402


def main(data_path: str) -> None:
    data = load_data(data_path)
    train, test = train_test_split(data, test_size=.25, random_state=42, stratify=data.intent)
    model = build_classifier().fit(train.text, train.intent)
    predicted = model.predict(test.text)
    print("Macro F1:", round(f1_score(test.intent, predicted, average="macro"), 4))
    print(classification_report(test.intent, predicted, zero_division=0))
    labels = sorted(data.intent.unique())
    matrix = confusion_matrix(test.intent, predicted, labels=labels)
    figure, axis = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(ax=axis, xticks_rotation=35, colorbar=False)
    figure.tight_layout()
    output = ROOT / "outputs" / "confusion_matrix.png"
    output.parent.mkdir(exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "demo_tickets.csv"))
    main(parser.parse_args().data)
