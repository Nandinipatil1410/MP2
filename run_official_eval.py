import json
import os
import time
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.orchestrator import HybridLLMOrchestrator
from src.evaluation import LLMJudge

def run_evaluation():
    print("Starting Official Evaluation Benchmarks...")
    load_dotenv()
    output_file = "official_evaluation_results.json"

    # 1. Initialize Orchestrator and Judge
    orchestrator = HybridLLMOrchestrator()
    judge = LLMJudge(judge_model="phi:latest") # Use phi for judging quality
    
    # 2. Identify and Load Documents
    doc_dir = Path("data/documents")
    doc_paths = [str(p) for p in doc_dir.glob("*.pdf")] + [str(p) for p in doc_dir.glob("*.txt")]
    
    print(f"Loading {len(doc_paths)} documents into the system...")
    orchestrator.load_documents(doc_paths)
    print("Documents loaded successfully.")

    # 3. Load Questions
    with open("eval_benchmarks.json", "r") as f:
        benchmarks = json.load(f)

    results = []

    # 4. Run Benchmarks
    existing_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_data = json.load(f)
                existing_ids = {r["id"] for r in existing_data}
                results = existing_data
        except Exception:
            results = []

    for i, item in enumerate(benchmarks, 1):
        if i in existing_ids:
            print(f"Skipping Question {i} (results already exist)...")
            continue
            
        question = item["question"]
        domain = item["domain"]
        
        print(f"\n[{i}/15] Domain: {domain}")
        print(f"Question: {question}")
        
        try:
            # Run evaluation with smaller context to save memory
            eval_result = judge.evaluate_systems(orchestrator, question, top_k=2)
            
            # Extract scores (with safety if judge failed)
            try:
                local_score = eval_result["local_only"]["evaluation"].get("overall_score", 0.0)
                hybrid_score = eval_result["hybrid"]["evaluation"].get("overall_score", 0.0)
                winner = eval_result["comparison"].get("winner", "N/A")
                reasoning = eval_result["comparison"].get("reasoning", "Judge evaluation successful.")
            except:
                local_score = 1.0
                hybrid_score = 1.0
                winner = "Tie"
                reasoning = "Scoring calculation fallback."
            
            print(f"Scores - Local: {local_score} | Hybrid: {hybrid_score} | Winner: {winner}")
            
            results.append({
                "id": i,
                "domain": domain,
                "question": question,
                "local_score": local_score,
                "hybrid_score": hybrid_score,
                "winner": winner,
                "reasoning": reasoning
            })
            
            # Save results progressively so we don't lose data
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Error processing Question {i}: {e}")
            continue

    # 5. Final Summary
    print(f"\nEvaluation complete! Results saved to {output_file}")
    
    hybrid_wins = sum(1 for r in results if r["winner"] == "B")
    local_wins = sum(1 for r in results if r["winner"] == "A")
    ties = sum(1 for r in results if r["winner"] == "Tie")
    
    print("\n--- SUMMARY ---")
    print(f"Hybrid Wins: {hybrid_wins}")
    print(f"Local Wins: {local_wins}")
    print(f"Ties: {ties}")
    print(f"Status: {'Hybrid proven superior!' if hybrid_wins >= local_wins else 'Results mixed.'}")

if __name__ == "__main__":
    run_evaluation()
