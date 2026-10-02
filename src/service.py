"""Application service combining NLP triage and policy retrieval."""
from __future__ import annotations

from functools import lru_cache

from src.generation import generate_grounded_response
from src.handoff import build_handoff
from src.pipeline import triage
from src.preprocess import extract_entities
from src.retrieval import KnowledgeRetriever, grounded_suggestion
from src.verification import verify_operational_references
from src.duplicates import find_duplicate_candidates
from src.clarification import next_clarification
from src.multilingual import prepare_multilingual_text


@lru_cache(maxsize=1)
def get_retriever() -> KnowledgeRetriever:
    return KnowledgeRetriever()


def analyse_ticket(text: str) -> dict:
    language = prepare_multilingual_text(text)
    result = triage(language["analysis_text"])
    # Preserve references supplied in the source language, even if translation
    # is unavailable or changes surrounding wording.
    result["entities"] = extract_entities(text)
    verification = verify_operational_references(result["entities"])
    if verification["status"] == "reference_not_found":
        result["escalate_to_human"] = True
        result["department"] = "Human Escalation Queue"
        result["escalation_reasons"] = [*result["escalation_reasons"], "Operational reference not found in company records"]
        result["explanation"] = [*result["explanation"], "A supplied operational reference was not found, so an agent must verify it safely."]
    if language["source_language_code"] != "en" and not language["translated"]:
        result["escalate_to_human"] = True
        result["department"] = "Human Escalation Queue"
        result["escalation_reasons"] = [*result["escalation_reasons"], f"{language['source_language']} message requires a configured translation model"]
        result["explanation"] = [*result["explanation"], "The deployed classifier is English-first, so a human must review untranslated text."]
    retrieved = get_retriever().search(language["analysis_text"], k=3)
    answer = generate_grounded_response(language["analysis_text"], retrieved, result["escalate_to_human"])
    if answer is None:
        answer = grounded_suggestion(retrieved, result["escalate_to_human"])
    handoff = build_handoff(result, retrieved, verification)
    return {
        **result,
        "language": {key: value for key, value in language.items() if key != "analysis_text"},
        "clarification_question": next_clarification(result, text),
        "duplicate_candidates": find_duplicate_candidates(text),
        "verification": verification,
        "retrieval_backend": get_retriever().backend,
        "retrieved_sources": retrieved,
        "suggested_response": answer,
        "agent_handoff": handoff,
    }
