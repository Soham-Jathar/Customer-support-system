"""Application service combining NLP triage and policy retrieval."""
from __future__ import annotations

from functools import lru_cache

from src.generation import generate_grounded_response
from src.handoff import build_handoff
from src.pipeline import triage
from src.retrieval import KnowledgeRetriever, grounded_suggestion


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def analyse_ticket(text: str) -> dict:
    result = triage(text)
    retrieved = get_retriever().search(text, k=3)
    answer = generate_grounded_response(text, retrieved, result["escalate_to_human"])
    if answer is None:
        answer = grounded_suggestion(retrieved, result["escalate_to_human"])
    handoff = build_handoff(result, retrieved)
    return {**result, "retrieval_backend": get_retriever().backend, "retrieved_sources": retrieved, "suggested_response": answer, "agent_handoff": handoff}
