"""
Router Module
Maps classified intent to an execution path.
"""

# Execution path labels
PATH_SUMMARY = "summary_path"
PATH_EXTRACTION = "extraction_path"
PATH_REASONING = "reasoning_path"
PATH_COMPARISON = "comparison_path"

# Whether each path uses the cloud planner
PATH_USES_PLANNER = {
    PATH_SUMMARY: True,
    PATH_EXTRACTION: False,
    PATH_REASONING: True,
    PATH_COMPARISON: True,
}

_INTENT_TO_PATH = {
    "summary": PATH_REASONING,     # Now uses planner for deep summaries
    "extraction": PATH_REASONING,  # Now uses planner for deep extraction
    "reasoning": PATH_REASONING,
    "comparison": PATH_COMPARISON,
}


def route(intent: str) -> str:
    """
    Map an intent label to an execution path.

    Args:
        intent: One of "summary" | "extraction" | "reasoning" | "comparison"

    Returns:
        Path constant (e.g. "reasoning_path"). Defaults to extraction_path
        for any unrecognised intent.
    """
    path = _INTENT_TO_PATH.get(intent, PATH_EXTRACTION)
    uses_planner = PATH_USES_PLANNER[path]
    print(f"  [Router] intent='{intent}' → path='{path}' (planner={'yes' if uses_planner else 'no'})")
    return path


def uses_planner(path: str) -> bool:
    """Return True if this execution path invokes the cloud planner."""
    return PATH_USES_PLANNER.get(path, False)
