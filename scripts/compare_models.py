"""Create a report-ready comparison from the two official-test metric files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def load(name: str) -> dict:
    path = OUTPUTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run its model evaluation first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    baseline = load("generic_intent_test_metrics.json")
    transformer = load("distilbert_generic_intent_test_metrics.json")
    names = ["TF-IDF +\nLogistic Regression", "DistilBERT"]
    metrics = ["accuracy", "macro_f1", "weighted_f1"]
    labels = ["Accuracy", "Macro F1", "Weighted F1"]
    values = [[baseline[metric], transformer[metric]] for metric in metrics]

    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    colours = ["#155eef", "#7f56d9"]
    for axis, label, pair in zip(axes, labels, values):
        bars = axis.bar(names, pair, color=colours, width=.56)
        axis.set_title(label, fontweight="bold")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", left=False)
        for bar, value in zip(bars, pair):
            axis.text(bar.get_x() + bar.get_width() / 2, value + .02, f"{value:.4f}", ha="center", fontweight="bold")
    figure.suptitle("Generic Customer-Support Held-Out Model Comparison", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(OUTPUTS / "model_comparison.png", dpi=220, bbox_inches="tight")
    summary = (
        "Generic customer-support held-out test comparison\n"
        f"TF-IDF + Logistic Regression: accuracy={baseline['accuracy']:.4f}, macro_f1={baseline['macro_f1']:.4f}, weighted_f1={baseline['weighted_f1']:.4f}\n"
        f"DistilBERT: accuracy={transformer['accuracy']:.4f}, macro_f1={transformer['macro_f1']:.4f}, weighted_f1={transformer['weighted_f1']:.4f}\n"
        "Conclusion: select TF-IDF + Logistic Regression for this prototype because it achieved the higher held-out macro F1."
    )
    (OUTPUTS / "model_comparison_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Saved {OUTPUTS / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
