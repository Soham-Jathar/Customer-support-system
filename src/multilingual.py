"""Conservative language detection for safe multilingual ticket handling.

The project classifiers are trained on English customer-support data. A
non-English ticket is detected and deliberately routed for human review rather
than being translated by an extra local model at runtime.
"""
from __future__ import annotations


LANGUAGE_NAMES = {
    "en": "English", "as": "Assamese", "bn": "Bengali", "gu": "Gujarati",
    "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "ne": "Nepali", "or": "Odia", "pa": "Punjabi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu", "sa": "Sanskrit", "es": "Spanish",
    "fr": "French", "de": "German",
}

SCRIPT_FALLBACKS = (
    ("\u0980", "\u09ff", "bn"),  # Bengali / Assamese; detector distinguishes when available.
    ("\u0a00", "\u0a7f", "pa"),  # Gurmukhi Punjabi
    ("\u0a80", "\u0aff", "gu"),  # Gujarati
    ("\u0b00", "\u0b7f", "or"),  # Odia
    ("\u0b80", "\u0bff", "ta"),  # Tamil
    ("\u0c00", "\u0c7f", "te"),  # Telugu
    ("\u0c80", "\u0cff", "kn"),  # Kannada
    ("\u0d00", "\u0d7f", "ml"),  # Malayalam
    ("\u0600", "\u06ff", "ur"),  # Urdu / Perso-Arabic script
)


def _safe_fallback_language(text: str) -> str:
    """Avoid treating clearly non-English script as English when detection fails."""
    if any("\u0900" <= character <= "\u097f" for character in text):
        return "hi"
    for lower, upper, language_code in SCRIPT_FALLBACKS:
        if any(lower <= character <= upper for character in text):
            return language_code
    return "en" if text.isascii() else "unknown"


def prepare_multilingual_text(text: str) -> dict:
    """Return language metadata while preserving original text for safe review."""
    try:
        from langdetect import detect
        code = detect(text)
    except Exception:
        code = _safe_fallback_language(text)
    return {
        "source_language": LANGUAGE_NAMES.get(code, "Unknown language"),
        "source_language_code": code,
        "translated": False,
        "translation_available": False,
        "translation_backend": None,
        "analysis_text": text,
    }
