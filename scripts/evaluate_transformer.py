"""Evaluate a fine-tuned Hugging Face intent classifier on BANKING77's test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import load_data  # noqa: E402


def main(model_dir: str, test_path: str) -> None:
    test = load_data(test_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    id_to_label = {int(key): value for key, value in model.config.id2label.items()}
    predicted_ids: list[int] = []
    for start in range(0, len(test), 64):
        batch = tokenizer(test.text.iloc[start:start + 64].tolist(), truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**batch).logits.numpy()
        predicted_ids.extend(np.argmax(logits, axis=-1).tolist())
    predicted = [id_to_label[index] for index in predicted_ids]
    report = classification_report(test.intent, predicted, zero_division=0, output_dict=True)
    summary = {
        "dataset": str(Path(test_path)), "examples": len(test), "model": model_dir,
        "accuracy": accuracy_score(test.intent, predicted),
        "macro_f1": f1_score(test.intent, predicted, average="macro", zero_division=0),
        "weighted_f1": f1_score(test.intent, predicted, average="weighted", zero_division=0),
        "per_intent": report,
    }
    output = ROOT / "outputs" / "distilbert_banking77_test_metrics.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1']:.4f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "models" / "distilbert-intent"))
    parser.add_argument("--test", default=str(ROOT / "data" / "banking77" / "banking77_test.csv"))
    args = parser.parse_args()
    main(args.model, args.test)
