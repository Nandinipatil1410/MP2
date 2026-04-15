class MetricsTracker:
    def __init__(self):
        self.retrieval_time = 0
        self.generation_time = 0
        self.total_time = 0

    def print_report(self):
        print("\n --- Latency Breakdown ---")
        print(f"Retrieval Time: {self.retrieval_time:.4f}s")
        print(f"Generation Time: {self.generation_time:.4f}s")
        print(f"Total Time: {self.total_time:.4f}s")
