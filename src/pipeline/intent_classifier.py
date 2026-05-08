"""
Intent Classifier Module
Classifies query intent using LLM (Groq), with keyword fallback.
"""
import re
from typing import Optional


# Intent labels
INTENT_SUMMARY = "summary"
INTENT_EXTRACTION = "extraction"
INTENT_REASONING = "reasoning"
INTENT_COMPARISON = "comparison"

VALID_INTENTS = {INTENT_SUMMARY, INTENT_EXTRACTION, INTENT_REASONING, INTENT_COMPARISON}

_CLASSIFICATION_PROMPT = """\
Classify the following query into exactly one category:

- summary: asking for an overview, description, high-level summary, or what the document is about
- extraction: asking for a specific fact, value, name, date, number, list, or direct lookup
- reasoning: asking why, how, what caused something, or requiring a review, evaluation, critique, or multi-step inference
- comparison: asking to compare, contrast, or find differences between two or more things

Query: "{query}"

Return ONLY the single label. No explanation, no punctuation."""


def classify_query_intent(query: str, groq_client=None) -> str:
    """
    Classify query intent via LLM (Groq) when available, else keyword fallback.

    Args:
        query: The user query string.
        groq_client: An initialised groq.Groq client (or None for fallback).

    Returns:
        One of: "summary" | "extraction" | "reasoning" | "comparison"
    """
    if groq_client is not None:
        intent = _classify_with_llm(query, groq_client)
        if intent in VALID_INTENTS:
            return intent
        # LLM returned something unexpected — fall through to keyword rules
        print(f"  [IntentClassifier] Unexpected LLM label '{intent}', using keyword fallback.")

    return _classify_with_keywords(query)


# ---------------------------------------------------------------------------
# LLM-based classification
# ---------------------------------------------------------------------------

def _classify_with_llm(query: str, groq_client) -> str:
    """Call Groq to classify intent. Returns raw label string."""
    try:
        from config.settings import CLOUD_MODEL  # type: ignore
        model = CLOUD_MODEL
    except ImportError:
        model = "llama3-8b-8192"

    try:
        user_prompt = _CLASSIFICATION_PROMPT.format(query=query)
        
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip()
        
        # Enhanced terminal logging for transparency
        print(f"\n\n{'='*80}")
        print(f" \U0001f310 CLOUD HANDSHAKE: Intent Classification")
        print(f"{'='*80}")
        print(f"\nUSER PROMPT:\n{'-'*20}")
        print(user_prompt)
        print(f"\n\u2705 CLOUD RESPONSE:\n{'-'*20}")
        print(label)
        print(f"{'='*80}\n")

        # Normalise: strip quotes / punctuation
        label_norm = re.sub(r"[^a-z]", "", label.lower())
        return label_norm
    except Exception as e:
        print(f"\n\u274c [IntentClassifier] LLM call failed ({e}), using keyword fallback.")
        return ""


# ---------------------------------------------------------------------------
# Keyword-based fallback classifier
# ---------------------------------------------------------------------------

def _classify_with_keywords(query: str) -> str:
    """Lightweight keyword-based intent classifier used as fallback."""
    ql = re.sub(r"\s+", " ", (query or "").strip().lower())

    # Summary
    summary_phrases = [
        "what is this document about", "what is this paper about",
        "what is this report about", "what does this document talk about",
        "what does this paper talk about", "give me an overview",
        "high level overview", "summarize", "summary of",
        "describe this document", "describe this paper",
        "document overview", "paper overview", "main idea", "what is this about",
        "overview of",
    ]
    if any(p in ql for p in summary_phrases):
        return INTENT_SUMMARY

    # Comparison
    comparison_markers = [
        "compare", "difference between", "differences between",
        " vs ", " vs.", "versus", "contrast",
    ]
    if any(m in ql for m in comparison_markers):
        return INTENT_COMPARISON

    # Reasoning
    reasoning_markers = [
        "why", "how", "explain", "reason", "cause", "despite",
        "impact", "effect", "drivers", "because", "justify",
        "review", "evaluate", "evaluation", "critique", "assess",
    ]
    if any(re.search(rf"\b{re.escape(m)}\b", ql) for m in reasoning_markers):
        return INTENT_REASONING

    # Extraction (default)
    return INTENT_EXTRACTION
