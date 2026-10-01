"""Train the reproducible TF-IDF + Logistic Regression baseline."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import MODEL_PATH
from src.preprocess import normalize_text


def build_classifier() -> Pipeline:
    base = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ])),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    # Post-hoc sigmoid calibration makes predict_proba scores more meaningful
    # than raw multiclass Logistic Regression scores on natural user phrasing.
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


def load_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path).dropna(subset=["text", "intent"]).copy()
    if not {"text", "intent"}.issubset(data.columns):
        raise ValueError("CSV must contain text and intent columns.")
    data["text"] = data["text"].map(normalize_text)
    data["intent"] = data["intent"].astype(str).str.strip()
    if data.intent.nunique() < 2:
        raise ValueError("Training data must contain at least two intent labels.")
    return data


def train(data_path: str | Path) -> Path:
    data = load_data(data_path)
    classifier = build_classifier().fit(data.text, data.intent)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(data.intent.value_counts().sort_index().to_string())
    return MODEL_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with text,intent")
    train(parser.parse_args().data)
