"""Fine-tune DistilBERT on the same BANKING77 labels as the baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train import load_data  # noqa: E402


class TokenizedTicketDataset(torch.utils.data.Dataset):
    """Native PyTorch dataset: avoids datasets/dill incompatibility on Python 3.14."""
    def __init__(self, texts, label_ids, tokenizer):
        self.encodings = tokenizer(texts.tolist(), truncation=True, max_length=128)
        self.label_ids = label_ids.tolist()

    def __len__(self) -> int:
        return len(self.label_ids)

    def __getitem__(self, index: int) -> dict:
        item = {key: torch.tensor(values[index]) for key, values in self.encodings.items()}
        item["labels"] = torch.tensor(self.label_ids[index])
        return item


def main(data_path: str, output_dir: str) -> None:
    data = load_data(data_path)
    labels = sorted(data.intent.unique())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train, validation = train_test_split(data, test_size=.20, random_state=42, stratify=data.intent)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def prepare(frame):
        label_ids = frame.intent.map(label_to_id)
        return TokenizedTicketDataset(frame.text, label_ids, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label_to_id)
    def compute_metrics(prediction):
        logits, expected = prediction
        predicted = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(expected, predicted),
            "macro_f1": f1_score(expected, predicted, average="macro", zero_division=0),
            "weighted_f1": f1_score(expected, predicted, average="weighted", zero_division=0),
        }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
            learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=16,
            num_train_epochs=3, logging_steps=50, report_to="none",
        ),
        train_dataset=prepare(train), eval_dataset=prepare(validation),
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print({key: round(value, 4) for key, value in metrics.items() if isinstance(value, float)})
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_tickets.csv")
    parser.add_argument("--output", default="models/distilbert-intent")
    args = parser.parse_args()
    main(args.data, args.output)
