"""
Validation Layer
Checks that a generated answer is:
  1. Grounded in the retrieved context (no hallucination)
  2. Actually answers the original query
Uses Groq when available; falls back to a local heuristic check.
"""
import re
from typing import Any, Dict, Optional


_VALIDATION_PROMPT = """\
You are a Grounding Auditor. Evaluate the Answer against the Context below.

QUERY:
{query}

CONTEXT (the only valid source of truth):
{context}

ANSWER:
{answer}

Evaluate on three criteria:
1. grounded: Is every factual claim in the Answer supported by the Context?
2. query_answered: Does the Answer actually address what the Query asked?
3. hallucination_detected: Does the Answer contain specific facts NOT present in the Context?

Return ONLY a JSON object:
{{
  "valid": <true if grounded AND query_answered AND NOT hallucination_detected>,
  "score": <float 0.0 to 1.0 representing overall quality>,
  "grounded": <true|false>,
  "query_answered": <true|false>,
  "hallucination_detected": <true|false>,
  "critique": "<one sentence explanation>"
}}"""

_INSUFFICIENT_DATA_MARKERS = [
    "insufficient data",
    "not stated in",
    "not found in",
    "no relevant",
    "cannot find",
    "data_absent",
    "not mentioned",
]


def validate(
    answer: str,
    context: str,
    query: str,
    groq_client=None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate the answer against the retrieved context and original query.

    Args:
        answer: The generated answer string.
        context: The retrieved document context used to produce the answer.
        query: The original user query.
        groq_client: Optional Groq client for LLM-based validation.
        model: Groq model to use (falls back to config default).

    Returns:
        {
            "valid": bool,
            "score": float,
            "grounded": bool,
            "query_answered": bool,
            "hallucination_detected": bool,
            "critique": str,
        }
    """
    if not answer or not answer.strip():
        return _invalid("Empty answer returned.")

    # If the answer itself is an "insufficient data" response, treat as valid
    al = answer.lower()
    if any(marker in al for marker in _INSUFFICIENT_DATA_MARKERS):
        return {
            "valid": True,
            "score": 0.5,
            "grounded": True,
            "query_answered": True,
            "hallucination_detected": False,
            "critique": "Answer correctly reports insufficient data.",
        }

    if groq_client is not None:
        result = _validate_with_llm(answer, context, query, groq_client, model)
        if result is not None:
            return result

    return _validate_with_heuristics(answer, context, query)


# ---------------------------------------------------------------------------
# LLM-based validation
# ---------------------------------------------------------------------------

def _validate_with_llm(
    answer: str,
    context: str,
    query: str,
    groq_client,
    model: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Use Groq to validate. Returns None on any error (triggers heuristic fallback)."""
    if model is None:
        try:
            from config.settings import CLOUD_MODEL  # type: ignore
            model = CLOUD_MODEL
        except ImportError:
            model = "llama3-8b-8192"

    # Truncate context only if extremely long to avoid token limit (Groq supports 8k tokens ~ 32k chars)
    safe_context = context[:25000] if len(context) > 25000 else context

    user_prompt = _VALIDATION_PROMPT.format(
        query=query, context=safe_context, answer=answer
    )
    print(f"\n--- [Validator] CLOUD PROMPT ---\n{user_prompt}\n" + "-"*40)
    import json
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=200,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "valid": bool(data.get("valid", True)),
            "score": float(data.get("score", 0.8)),
            "grounded": bool(data.get("grounded", True)),
            "query_answered": bool(data.get("query_answered", True)),
            "hallucination_detected": bool(data.get("hallucination_detected", False)),
            "critique": str(data.get("critique", "")),
        }
    except Exception as e:
        print(f"  [Validator] LLM validation failed ({e}), using heuristic fallback.")
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback validator
# ---------------------------------------------------------------------------

def _validate_with_heuristics(answer: str, context: str, query: str) -> Dict[str, Any]:
    """
    Lightweight local validity check — no LLM required.
    Checks: answer length, query term overlap, and context overlap.
    """
    if len(answer.strip()) < 20:
        return _invalid("Answer too short to be meaningful.")

    # Query-answer overlap: at least some content words from query appear in answer
    query_words = set(re.findall(r"\b[a-z]{4,}\b", query.lower()))
    answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer.lower()))
    overlap_ratio = len(query_words & answer_words) / max(len(query_words), 1)

    query_answered = overlap_ratio >= 0.2

    # Context-answer overlap: some words from context appear in answer
    context_words = set(re.findall(r"\b[a-z]{5,}\b", context.lower()))
    ctx_overlap = len(context_words & answer_words) / max(len(context_words), 1)
    grounded = ctx_overlap >= 0.05  # low threshold — we just want some overlap

    score = round(0.5 * float(grounded) + 0.3 * float(query_answered) + 0.2 * min(len(answer) / 500, 1.0), 2)
    valid = grounded and query_answered

    return {
        "valid": valid,
        "score": score,
        "grounded": grounded,
        "query_answered": query_answered,
        "hallucination_detected": not grounded,
        "critique": f"Heuristic check: context_overlap={ctx_overlap:.2f}, query_overlap={overlap_ratio:.2f}",
    }


def _invalid(reason: str) -> Dict[str, Any]:
    return {
        "valid": False,
        "score": 0.0,
        "grounded": False,
        "query_answered": False,
        "hallucination_detected": False,
        "critique": reason,
    }
