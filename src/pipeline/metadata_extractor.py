"""
Document Metadata Extractor Module
Uses the local model (Ollama) to extract structured metadata from document chunks.
Only the metadata dict — NOT the raw content — is forwarded to the cloud planner.
"""
import json
import requests
from typing import Dict, Any


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

Return ONLY valid JSON. No explanation, no markdown fences.

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
    # Truncate to avoid overwhelming the local model
    safe_context = context[:6000] if len(context) > 6000 else context

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
                    "num_predict": 300,
                    "stop": ["###"],
                },
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Parse JSON — handle model wrapping output in markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()

        metadata = json.loads(raw)

        # Validate expected shape
        result = {
            "document_type": str(metadata.get("document_type", "unknown")),
            "sections": _ensure_list(metadata.get("sections", [])),
            "key_fields": _ensure_list(metadata.get("key_fields", [])),
            "data_types": _ensure_list(metadata.get("data_types", ["text"])),
        }
        print(f"  [MetadataExtractor] type='{result['document_type']}' | "
              f"sections={result['sections'][:3]} | fields={result['key_fields'][:4]}")
        return result

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"  [MetadataExtractor] Failed to extract metadata ({e}), using fallback.")
        return _FALLBACK_METADATA.copy()


def _ensure_list(value) -> list:
    """Coerce value to a flat list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    return []
