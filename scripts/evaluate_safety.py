"""Evaluate priority and human-escalation behaviour on the held-out safety set."""
from __future__ import annotations

import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import triage  # noqa: E402


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "safety_eval.csv")
    predictions = [triage(text) for text in data.text]
    data["predicted_priority"] = [item["priority"] for item in predictions]
    data["predicted_escalation"] = [item["escalate_to_human"] for item in predictions]
    print("Priority evaluation")
    print(classification_report(data.expected_priority, data.predicted_priority, zero_division=0))
    truth = data.expected_escalation.astype(str).str.lower().eq("true")
    predicted = data.predicted_escalation.astype(bool)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    print(f"Escalation precision: {precision:.3f}")
    print(f"Escalation recall:    {recall:.3f}")
    print(f"Escalation F1:        {f1:.3f}")
    output = ROOT / "outputs" / "safety_eval_predictions.csv"
    data.to_csv(output, index=False)
    summary = {
        "examples": len(data),
        "priority_macro_f1": f1_score(data.expected_priority, data.predicted_priority, average="macro", zero_division=0),
        "escalation_precision": precision,
        "escalation_recall": recall,
        "escalation_f1": f1,
    }
    summary_path = ROOT / "outputs" / "safety_eval_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Safety summary:", {key: round(value, 4) if isinstance(value, float) else value for key, value in summary.items()})
    print(f"Saved predictions to {output}")
    print(f"Saved metrics to {summary_path}")


if __name__ == "__main__":
    main()
