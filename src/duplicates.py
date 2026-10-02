"""Similarity-based duplicate-ticket candidates for the local support queue."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select

from src.database import SessionLocal, Ticket


def find_duplicate_candidates(message: str, threshold: float = 0.62, limit: int = 3) -> list[dict]:
    """Return likely duplicates; an agent must confirm before merging tickets."""
    with SessionLocal() as session:
        tickets = session.scalars(select(Ticket).where(Ticket.status != "resolved")).all()
        if not tickets:
            return []
        corpus = [message, *[ticket.customer_message for ticket in tickets]]
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(corpus)
        scores = cosine_similarity(matrix[0], matrix[1:])[0]
        matches = []
        for ticket, score in zip(tickets, scores):
            if float(score) >= threshold:
                matches.append({
                    "ticket_id": ticket.id,
                    "similarity": round(float(score), 4),
                    "status": ticket.status,
                    "department": ticket.department,
                })
        return sorted(matches, key=lambda item: item["similarity"], reverse=True)[:limit]
