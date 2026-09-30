"""Measure Recall@k for knowledge-base retrieval against labelled query-document pairs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.retrieval import KnowledgeRetriever  # noqa: E402


def main() -> None:
    retriever = KnowledgeRetriever()
    rows = list(csv.DictReader((ROOT / "data" / "retrieval_eval.csv").open(encoding="utf-8")))
    metrics = {}
    for k in (1, 3, 5):
        correct = sum(row["expected_document_id"] in {item["id"] for item in retriever.search(row["query"], k)} for row in rows)
        metrics[f"recall_at_{k}"] = correct / len(rows)
        print(f"Recall@{k}: {metrics[f'recall_at_{k}']:.3f} ({correct}/{len(rows)})")
    print("Backend:", retriever.backend)
    output = ROOT / "outputs" / "retrieval_eval_metrics.json"
    output.write_text(json.dumps({"queries": len(rows), "backend": retriever.backend, **metrics}, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
