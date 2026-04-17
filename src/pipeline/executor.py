"""
Step-wise Execution Engine
Runs a structured plan (list of steps) against the vector store using the local model.
Each step gets its own targeted retrieval query, structured data extraction, and reasoning call.
"""
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def execute_steps(
    steps: List[Dict[str, Any]],
    query: str,
    vector_store,
    local_llm,
    selected_document: Optional[str] = None,
    top_k: int = 5,
    expert_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run a structured plan step-by-step using local retrieval and local reasoning.

    Args:
        steps: List of step dicts, each with keys: id, action, target
        query: The original user query (used for context and fallback retrieval)
        vector_store: VectorStore instance with a .search() method
        local_llm: LocalLLMExecutor instance (must have ._call_ollama and ._clean_response)
        selected_document: Optional filename to scope retrieval
        top_k: Number of chunks to retrieve per step

    Returns:
        {
            "memory": { "data": [...], "intermediate_results": [...], "insights": [...] },
            "all_retrieved_docs": [...],
            "combined_findings": str,
        }
    """
    memory: Dict[str, List] = {
        "data": [],            # raw retrieved chunk texts per step
        "intermediate_results": [],  # extracted structured data per step
        "insights": [],        # step-level reasoned conclusions
    }
    all_retrieved_docs: List[Dict[str, Any]] = []

    for step in steps:
        step_id = step.get("id", "?")
        action = step.get("action", "extract")
        target = step.get("target", query)

        # 1. Generate targeted retrieval query from action + target
        retrieval_query = _make_retrieval_query(action, target, query)
        print(f"  [Executor] Step {step_id} [{action}] → retrieving: '{retrieval_query[:80]}'")

        # 2. Retrieve relevant chunks
        docs = _retrieve(vector_store, retrieval_query, query, top_k, selected_document)
        all_retrieved_docs.extend(docs)
        context = local_llm._prepare_context(docs) if docs else "No relevant context retrieved."
        memory["data"].append({"step": step_id, "docs_retrieved": len(docs)})

        # 3. Extract structured data (numbers, entities, dates) from context
        structured = _extract_structured_data(context)
        memory["intermediate_results"].append({"step": step_id, "extracted": structured})

        # 4. Execute step reasoning using local model
        previous = "\n".join(memory["insights"][-2:]) if memory["insights"] else "None"
        step_insight = _reason_step(
            local_llm=local_llm,
            step_id=step_id,
            action=action,
            target=target,
            query=query,
            context=context,
            structured_data=structured,
            previous_insights=previous,
            expert_mode=expert_mode,
        )
        memory["insights"].append(step_insight)
        print(f"  [Executor] Step {step_id} insight: {step_insight[:100]}...")

    # Combine all insights into a single findings string for synthesis
    combined_findings = _combine_findings(memory, steps)

    return {
        "memory": memory,
        "all_retrieved_docs": all_retrieved_docs,
        "combined_findings": combined_findings,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_retrieval_query(action: str, target: str, original_query: str) -> str:
    """
    Build a clean, specific retrieval query from a step's action and target.
    We do NOT append the original query — long mixed queries hurt FAISS retrieval.
    """
    # Use the target directly; action words confuse semantic search
    return target.strip()


def _retrieve(
    vector_store,
    primary_query: str,
    fallback_query: str,
    top_k: int,
    selected_document: Optional[str],
) -> List[Dict[str, Any]]:
    """Retrieve docs for a step; deduped by (source, chunk_id)."""
    seen = set()
    results = []

    for q in [primary_query, fallback_query]:
        for doc in vector_store.search(q, top_k=top_k, source=selected_document):
            key = (doc.get("source"), doc.get("chunk_id"), doc.get("start_idx"))
            if key not in seen:
                seen.add(key)
                results.append(doc)
        if len(results) >= top_k:
            break

    return results[:top_k]


def _extract_structured_data(context: str) -> Dict[str, List[str]]:
    """
    Lightweight regex-based extraction of key value types from context.
    Returns structured data dict — passed to the reasoning prompt for grounding.
    """
    numbers = re.findall(r"\b\d+(?:[.,]\d+)*(?:\s*%|[KkMmBb])?\b", context)
    dates = re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}|\d{4})\b",
        context,
    )
    # Named entities: capitalised multi-word sequences
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", context)

    return {
        "numbers": list(dict.fromkeys(numbers))[:20],
        "dates": list(dict.fromkeys(dates))[:10],
        "entities": list(dict.fromkeys(entities))[:15],
    }


def _reason_step(
    local_llm,
    step_id,
    action: str,
    target: str,
    query: str,
    context: str,
    structured_data: Dict[str, List[str]],
    previous_insights: str,
    expert_mode: bool = False,
) -> str:
    """Execute one reasoning step using the local model."""
    structured_block = ""
    if any(structured_data.values()):
        parts = []
        if structured_data["numbers"]:
            parts.append(f"Numbers found: {', '.join(structured_data['numbers'][:10])}")
        if structured_data["dates"]:
            parts.append(f"Dates found: {', '.join(structured_data['dates'][:5])}")
        if structured_data["entities"]:
            parts.append(f"Entities found: {', '.join(structured_data['entities'][:8])}")
        structured_block = "### STRUCTURED DATA EXTRACTED:\n" + "\n".join(parts) + "\n\n"

    instruction_4 = (
        "Be detailed and comprehensive. Retain specific statistics, dates, names, and nuanced arguments. Do NOT over-summarize."
        if expert_mode
        else "Be concise and direct — 2 to 5 sentences maximum."
    )

    prompt = f"""### TASK: {action.upper()} — {target}
### PRIMARY QUERY: {query}

### RETRIEVED CONTEXT:
{context}

{structured_block}### PREVIOUS INSIGHTS:
{previous_insights}

### INSTRUCTIONS:
1. Perform the task above using ONLY the Retrieved Context and Structured Data.
2. If the required information is absent, state: "DATA_ABSENT: [what is missing]"
3. Ground every claim in the context. Do NOT speculate.
4. {instruction_4}

### ANALYSIS:"""

    try:
        response = local_llm._call_ollama(prompt, max_tokens=400)
        return local_llm._clean_response(response)
    except Exception as e:
        return f"Step {step_id} failed: {e}"


def _combine_findings(memory: Dict[str, List], steps: List[Dict]) -> str:
    """Merge step insights into a flat findings string for synthesis."""
    lines = []
    for i, insight in enumerate(memory["insights"]):
        step = steps[i] if i < len(steps) else {}
        action = step.get("action", "step")
        target = step.get("target", "")
        lines.append(f"[{action.upper()} — {target}]\n{insight}")
    return "\n\n".join(lines)
