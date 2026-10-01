"""End-to-end ticket analysis used by the UI and tests."""
from __future__ import annotations

from functools import lru_cache

import joblib

from src.config import MODEL_PATH, QUERY_TYPE_MODEL_PATH
from src.preprocess import extract_entities, normalize_text
from src.rules import make_decision
from src.taxonomy import query_type_for


@lru_cache(maxsize=1)
def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Generic intent model missing. Run the generic data-preparation and training commands in README.md.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_query_type_classifier():
    return joblib.load(QUERY_TYPE_MODEL_PATH) if QUERY_TYPE_MODEL_PATH.exists() else None


def analyse_sentiment(text: str) -> tuple[str, float, str]:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        score = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        source = "VADER"
    except (ImportError, LookupError):
        negative = {"angry", "bad", "broken", "disappointed", "failed", "hate", "late", "problem", "refund", "urgent", "wrong"}
        positive = {"good", "great", "helpful", "love", "perfect", "thanks"}
        tokens = set(normalize_text(text).lower().split())
        score = max(-1.0, min(1.0, .2 * len(tokens & positive) - .2 * len(tokens & negative)))
        source = "lexical fallback; install VADER lexicon for formal evaluation"
    return ("positive" if score >= .2 else "negative" if score <= -.2 else "neutral"), float(score), source


def triage(text: str) -> dict:
    text = normalize_text(text)
    if not text:
        raise ValueError("Enter a customer-support message.")
    model = load_classifier()
    probabilities = model.predict_proba([text])[0]
    labels = model.classes_
    ranking = sorted(zip(labels, probabilities), key=lambda pair: pair[1], reverse=True)
    intent, confidence = str(ranking[0][0]), float(ranking[0][1])
    query_model = load_query_type_classifier()
    if query_model is None:
        query_type, query_confidence, query_source = query_type_for(intent), None, "derived from fine-grained intent"
    else:
        query_probabilities = query_model.predict_proba([text])[0]
        query_index = int(query_probabilities.argmax())
        query_type = str(query_model.classes_[query_index])
        query_confidence = float(query_probabilities[query_index])
        query_source = "TF-IDF + Logistic Regression query-type classifier"
    sentiment, score, sentiment_source = analyse_sentiment(text)
    decision = make_decision(query_type, intent, confidence, sentiment, text, query_confidence)
    return {
        "intent": intent, "confidence": confidence,
        "query_type": query_type, "query_type_confidence": query_confidence, "query_type_source": query_source,
        "top_intents": [{"intent": str(label), "confidence": round(float(value), 4)} for label, value in ranking[:3]],
        "sentiment": sentiment, "sentiment_score": score, "sentiment_source": sentiment_source,
        "entities": extract_entities(text), **decision.to_dict(),
    }
