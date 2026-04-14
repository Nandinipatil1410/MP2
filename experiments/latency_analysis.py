from src.orchestrator import HybridLLMOrchestrator

orch = HybridLLMOrchestrator()

# Update this path if needed
orch.load_documents(["data/sample.txt"])

queries = [
    "What is the diagnosis?",
    "Summarize the contract",
    "Explain the research paper"
]

for q in queries:
    print("\n======================")
    print("QUERY:", q)

    print("\n--- HYBRID ---")
    orch.process_query_hybrid(q)

    print("\n--- LOCAL ---")
    orch.process_query_local_only(q)
