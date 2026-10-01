"""Optional local-LLM response generation, constrained to retrieved evidence."""
from __future__ import annotations

import os

import requests


def generate_grounded_response(customer_message: str, sources: list[dict], needs_human: bool) -> dict | None:
    """Return None unless the user deliberately enables a local Ollama model."""
    if needs_human or not sources or os.getenv("ENABLE_LLM_GENERATION", "false").lower() != "true":
        return None
    context = "\n\n".join(f"[{item['id']}] {item['content']}" for item in sources)
    prompt = f"""You are a customer-support drafting assistant. Use only the policy evidence below.
Never claim a refund is approved, an order has a particular status, or a delivery will arrive on a date.
If the evidence does not answer the request, say a human agent must verify it.
Keep the answer under 80 words and cite the policy ID in square brackets.

Customer message: {customer_message}
Policy evidence:
{context}
"""
    try:
        response = requests.post(
            os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate"),
            json={"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "prompt": prompt, "stream": False},
            timeout=20,
        )
        response.raise_for_status()
        draft = response.json().get("response", "").strip()
        if draft:
            return {"answer": draft, "grounded": True, "generator": "Ollama (local)"}
    except requests.RequestException:
        return None
    return None
