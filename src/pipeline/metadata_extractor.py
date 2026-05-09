"""
Document Metadata Extractor Module
Uses the local model (Ollama) to extract structured metadata from document chunks.
Only the metadata dict — NOT the raw content — is forwarded to the cloud planner.
"""
import json
import re
import requests
from typing import Dict, Any, Optional


_METADATA_PROMPT = """\
### TASK: Extract structured metadata from the document excerpt below.

### DOCUMENT EXCERPT:
{context}

### INSTRUCTIONS:
Analyse the excerpt and return a JSON object with EXACTLY these fields:
- "document_type": a short string describing what kind of document this is
  (e.g. "research paper", "financial report", "clinical note", "legal contract", "news article", "manual")
- "sections": a JSON array of section headings or major topics present in the excerpt (max 8 items)
- "key_fields": a JSON array of the most important data fields or concepts mentioned (max 10 items)
- "data_types": a JSON array containing any of: "numerical", "text", "dates", "entities", "tables"

Return ONLY valid JSON. No explanation, no markdown fences, no extra text.

Example output:
{{"document_type": "clinical note", "sections": ["Patient History", "Treatment"], "key_fields": ["diagnosis", "medication"], "data_types": ["text", "entities"]}}

### JSON OUTPUT:"""

_FALLBACK_METADATA: Dict[str, Any] = {
    "document_type": "unknown",
    "sections": [],
    "key_fields": [],
    "data_types": ["text"],
}


def extract_document_metadata(
    context: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "phi3:mini",
) -> Dict[str, Any]:
    """
    Extract structured metadata from document context using the local model.

    Args:
        context: Raw text from retrieved document chunks (truncated to safe size).
        ollama_url: Base URL for the Ollama API.
        model: Local Ollama model name.

    Returns:
        Dict with keys: document_type, sections, key_fields, data_types
    """
    # Truncate context to avoid overwhelming the local model
    safe_context = context[:4000] if len(context) > 4000 else context
    prompt = _METADATA_PROMPT.format(context=safe_context)

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 400,
                    "stop": ["###", "\n\n\n"],
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        print(f"  [MetadataExtractor] Raw model output:\n{raw[:600]}\n{'─'*40}")

        metadata = _parse_json_robust(raw)

        if metadata is None:
            print("  [MetadataExtractor] All JSON parsing strategies failed — using fallback.")
            return _FALLBACK_METADATA.copy()

        # Validate and normalise shape
        result = {
            "document_type": str(metadata.get("document_type", "unknown")),
            "sections":   _ensure_list(metadata.get("sections", [])),
            "key_fields": _ensure_list(metadata.get("key_fields", [])),
            "data_types": _ensure_list(metadata.get("data_types", ["text"])),
        }
        print(
            f"  [MetadataExtractor] OK — type='{result['document_type']}' | "
            f"sections={result['sections'][:3]} | fields={result['key_fields'][:4]}"
        )
        return result

    except requests.RequestException as e:
        print(f"  [MetadataExtractor] Ollama request failed: {e} — using fallback.")
        return _FALLBACK_METADATA.copy()


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _parse_json_robust(raw: str) -> Optional[Dict]:
    """
    Try five strategies to extract a valid JSON object from the model's response.
    Returns a dict on success, None if all strategies fail.
    """
    # Strategy 1: Direct parse (ideal case — model returned clean JSON)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown fences  ```json ... ```
    fenced = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    fenced = re.sub(r"\s*```$", "", fenced.strip())
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    # Strategy 3: First {...} block (simple, handles trailing prose)
    brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Greedy brace-depth scan (handles nested objects)
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start: i + 1])
                    except json.JSONDecodeError:
                        break

    # Strategy 5: Partial field extraction from key:"value" patterns
    partial = _extract_partial_fields(raw)
    if partial:
        print("  [MetadataExtractor] Used partial field extraction (last resort).")
        return partial

    return None


def _extract_partial_fields(text: str) -> Optional[Dict]:
    """
    Scan raw text for recognisable key-value patterns and assemble a
    best-effort metadata dict.  Returns None if nothing useful found.
    """
    result: Dict[str, Any] = {}

    m = re.search(r'"document_type"\s*:\s*"([^"]+)"', text)
    if m:
        result["document_type"] = m.group(1)

    m = re.search(r'"sections"\s*:\s*(\[[^\]]*\])', text)
    if m:
        try:
            result["sections"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'"key_fields"\s*:\s*(\[[^\]]*\])', text)
    if m:
        try:
            result["key_fields"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r'"data_types"\s*:\s*(\[[^\]]*\])', text)
    if m:
        try:
            result["data_types"] = json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    return result if result else None


def _ensure_list(value) -> list:
    """Coerce value to a flat list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []
