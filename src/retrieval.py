"""Knowledge-base retrieval with a Sentence Transformer + FAISS path and a local TF-IDF fallback."""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import ROOT

KB_PATH = ROOT / "knowledge_base" / "policies.json"


class KnowledgeRetriever:
    def __init__(self, documents_path: Path = KB_PATH):
        self.documents = json.loads(documents_path.read_text(encoding="utf-8"))
        self.texts = [f"{item['title']} {item['content']}" for item in self.documents]
        self.backend = "TF-IDF fallback"
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.model = self.index = None
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self.model.encode(self.texts, normalize_embeddings=True).astype("float32")
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            self.backend = "Sentence Transformers + FAISS"
        except Exception:
            # The optional embedding model may be unavailable offline; use the
            # evaluated lexical fallback and expose that backend in the UI.
            pass

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.model is not None and self.index is not None:
            vector = self.model.encode([query], normalize_embeddings=True).astype("float32")
            scores, indices = self.index.search(vector, min(k, len(self.documents)))
            return [self._result(int(i), float(score)) for score, i in zip(scores[0], indices[0])]
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        indices = scores.argsort()[::-1][:k]
        return [self._result(int(i), float(scores[i])) for i in indices]

    def _result(self, index: int, score: float) -> dict:
        document = self.documents[index]
        return {"id": document["id"], "title": document["title"], "content": document["content"], "score": round(score, 4)}


def grounded_suggestion(results: list[dict], needs_human: bool) -> dict:
    """Never use retrieval to claim live account, transaction, or refund facts."""
    if needs_human:
        return {"answer": "Your ticket has been escalated to a human agent. They will securely verify the relevant account, order, delivery, or payment details before taking action.", "grounded": False}
    if not results or results[0]["score"] < 0.10:
        return {"answer": "I could not find a supported policy answer for this request, so a human agent should review it.", "grounded": False}
    source = results[0]
    return {"answer": f"Based on the knowledge base: {source['content']}", "grounded": True, "source_id": source["id"]}
