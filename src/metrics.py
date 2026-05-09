"""
Metrics Tracker
Lightweight performance tracker for the Hybrid LLM pipeline.
Tracks retrieval, generation, and total latency; prints a formatted report.
"""


class MetricsTracker:
    """Tracks timing metrics for a single query lifecycle."""

    def __init__(self):
        self.retrieval_time: float = 0.0
        self.generation_time: float = 0.0
        self.total_time: float = 0.0

    def print_report(self) -> None:
        """Print a concise performance summary to stdout."""
        print("\n" + "─" * 50)
        print("  📊 PERFORMANCE REPORT")
        print("─" * 50)
        if self.retrieval_time:
            print(f"  ⏱  Retrieval   : {self.retrieval_time:.3f}s")
        if self.generation_time:
            print(f"  ⏱  Generation  : {self.generation_time:.3f}s")
        print(f"  ⏱  Total       : {self.total_time:.3f}s")
        print("─" * 50 + "\n")
