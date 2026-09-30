"""Build deterministic, non-generative summaries for human support handoff."""
from __future__ import annotations


def build_handoff(result: dict, retrieved_sources: list[dict]) -> dict:
    entities = [f"{name.replace('_', ' ')}: {', '.join(values)}" for name, values in result.get("entities", {}).items() if values]
    parts = [
        f"Query type: {result.get('query_type', 'other').replace('_', ' ')}",
        f"Predicted intent: {result['intent'].replace('_', ' ')} ({result['confidence']:.0%} confidence)",
        f"Priority: {result['priority']}",
        f"Sentiment: {result['sentiment']}",
        f"Route: {result['department']}",
    ]
    if entities:
        parts.append("Extracted references: " + "; ".join(entities))
    if result.get("escalation_reasons"):
        parts.append("Escalation reasons: " + "; ".join(result["escalation_reasons"]))
    policy_ids = [source["id"] for source in retrieved_sources]
    if policy_ids:
        parts.append("Retrieved policies: " + ", ".join(policy_ids))
    return {"summary": ". ".join(parts) + ".", "policy_ids": policy_ids}
