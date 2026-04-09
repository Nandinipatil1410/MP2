"""
Test and Demo Script
Privacy-Preserving Hybrid LLM System
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from orchestrator import HybridLLMOrchestrator
from evaluation import LLMJudge
import os
from dotenv import load_dotenv

load_dotenv()


def create_sample_documents():
    """Create sample documents for testing"""
    print("\n📝 Creating sample documents...")
    
    docs_dir = Path("./data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample document 1: AI Research
    doc1 = """
    Artificial Intelligence Research Summary
    
    Machine Learning is a subset of AI that focuses on enabling systems to learn from data.
    Deep Learning uses neural networks with multiple layers to process complex patterns.
    
    Key findings:
    1. Deep learning has revolutionized computer vision
    2. Transformer models have improved natural language processing
    3. Reinforcement learning shows promise in robotics
    
    Applications include autonomous vehicles, medical diagnosis, and natural language understanding.
    """
    
    with open(docs_dir / "ai_research.txt", "w") as f:
        f.write(doc1)
    
    # Sample document 2: Privacy Policy
    doc2 = """
    Privacy and Data Protection Guidelines
    
    Our organization is committed to protecting user privacy and ensuring data security.
    
    Key principles:
    - Data minimization: Collect only necessary information
    - Encryption: All sensitive data must be encrypted
    - Access control: Implement strict access permissions
    - Transparency: Users must be informed about data usage
    
    We comply with GDPR and other privacy regulations.
    """
    
    with open(docs_dir / "privacy_policy.txt", "w") as f:
        f.write(doc2)
    
    # Sample document 3: Project Report
    doc3 = """
    Q3 Project Report
    
    Project: Document Management System
    Status: In Progress
    
    Milestones achieved:
    - Database schema designed
    - User authentication implemented
    - Document upload functionality completed
    
    Next steps:
    - Implement search functionality
    - Add version control
    - Deploy to production
    
    Team members have been working efficiently to meet deadlines.
    """
    
    with open(docs_dir / "project_report.txt", "w") as f:
        f.write(doc3)
    
    print(f"✓ Created 3 sample documents in {docs_dir}")
    return [
        str(docs_dir / "ai_research.txt"),
        str(docs_dir / "privacy_policy.txt"),
        str(docs_dir / "project_report.txt")
    ]


def test_system():
    """Run system tests"""
    print("\n" + "="*70)
    print("PRIVACY-PRESERVING HYBRID LLM SYSTEM - DEMO")
    print("="*70)
    
    # Create sample documents
    doc_paths = create_sample_documents()
    
    # Initialize system
    print("\n🚀 Initializing system...")
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    orchestrator = HybridLLMOrchestrator(groq_api_key=groq_api_key)
    
    # Load documents
    print("\n📚 Loading documents...")
    orchestrator.load_documents(doc_paths)
    
    # Test queries
    test_queries = [
        "What are the key findings about deep learning?",
        "What are the privacy principles mentioned?",
        "What is the status of the project?"
    ]
    
    print("\n" + "="*70)
    print("TESTING QUERIES")
    print("="*70)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}: {query}")
        print(f"{'='*70}")
        
        # Test Hybrid mode
        print("\n--- HYBRID MODE ---")
        result_hybrid = orchestrator.process_query_hybrid(query)
        if result_hybrid['success']:
            print(f"\nAnswer: {result_hybrid['answer'][:200]}...")
            print(f"Latency: {result_hybrid['latency']:.2f}s")
            print(f"Privacy Score: {result_hybrid['privacy_score']:.2f}")
            print(f"Documents Used: {result_hybrid['documents_used']}")
        
        # Test Local-only mode
        print("\n--- LOCAL-ONLY MODE ---")
        result_local = orchestrator.process_query_local_only(query)
        if result_local['success']:
            print(f"\nAnswer: {result_local['answer'][:200]}...")
            print(f"Latency: {result_local['latency']:.2f}s")
            print(f"Privacy Score: {result_local['privacy_score']:.2f}")
    
    # Comparison test
    print("\n" + "="*70)
    print("COMPARING ALL APPROACHES")
    print("="*70)
    
    comparison_query = "Summarize the main topics across all documents"
    results = orchestrator.compare_approaches(comparison_query)
    
    print("\nResults:")
    for mode, result in results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Latency: {result['latency']:.2f}s")
        print(f"  Privacy: {result.get('privacy_score', 'N/A')}")
        if result['success']:
            print(f"  Answer: {result['answer'][:150]}...")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\nTo use the web interface, run:")
    print("   python src/app.py")


def quick_test():
    """Quick test without documents"""
    print("\n🔍 Quick System Test")
    print("="*50)
    
    orchestrator = HybridLLMOrchestrator()
    
    # Check stats
    stats = orchestrator.get_system_stats()
    print(f"\nSystem Statistics:")
    print(f"  Documents Loaded: {stats['documents_loaded']}")
    print(f"  Vector DB Size: {stats['vector_store']['total_documents']}")
    print(f"  Local Model: {stats['local_model']}")
    print(f"  Cloud Model: {stats['cloud_model']}")
    
    print("\n✓ System is operational!")


def run_automated_judge_demo(judge_model: str = "llama-3.3-70b-versatile"):
    """Automated Local-vs-Hybrid evaluation using LLM-as-a-judge."""
    print("\n" + "=" * 70)
    print("AUTOMATED LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    doc_paths = create_sample_documents()
    groq_api_key = os.getenv("GROQ_API_KEY", "")

    print("\nInitializing orchestrator...")
    orchestrator = HybridLLMOrchestrator(groq_api_key=groq_api_key)

    print("\nLoading sample documents...")
    orchestrator.load_documents(doc_paths)

    judge = LLMJudge(judge_model=judge_model, allow_groq_fallback=False)
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("\nNo GEMINI_API_KEY/GOOGLE_API_KEY found. Judge will use fallback heuristic mode.")
    else:
        print(f"\nUsing judge model: {judge_model}")

    eval_queries = [
        "What are the key findings about deep learning and where are they applied?",
        "List the privacy principles and explain why each one matters.",
        "Summarize project status, completed milestones, and next steps.",
    ]

    batch_result = judge.evaluate_batch(orchestrator, eval_queries, top_k=3)

    print("\n" + "=" * 70)
    print("PER-QUESTION RESULTS")
    print("=" * 70)
    for idx, item in enumerate(batch_result["per_question"], start=1):
        comparison = item["comparison"]
        local_score = item["local_only"]["evaluation"]["overall_score"]
        hybrid_score = item["hybrid"]["evaluation"]["overall_score"]

        winner_label = {"A": "Local-Only", "B": "Hybrid", "Tie": "Tie"}[comparison["winner"]]
        print(f"\nQ{idx}: {item['question']}")
        print(f"  Local score : {local_score:.2f}")
        print(f"  Hybrid score: {hybrid_score:.2f}")
        print(f"  Winner      : {winner_label} ({comparison['preference_strength']})")
        print("  Why:")
        for diff in comparison.get("key_differences", [])[:3]:
            print(f"    - {diff}")

    summary = batch_result["summary"]
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated : {summary['questions_evaluated']}")
    print(f"Average Local score : {summary['average_local_score']:.2f}")
    print(f"Average Hybrid score: {summary['average_hybrid_score']:.2f}")
    print(f"Hybrid wins         : {summary['hybrid_wins']}")
    print(f"Local wins          : {summary['local_wins']}")
    print(f"Ties                : {summary['ties']}")
    print(f"Overall winner      : {summary['overall_winner']}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Privacy-Preserving Hybrid LLM System")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument("--full", action="store_true", help="Run full demo with documents")
    parser.add_argument("--judge", action="store_true", help="Run automated LLM-as-a-judge evaluation")
    parser.add_argument(
        "--judge-model",
        default="llama-3.3-70b-versatile",
        help="Judge model to use for automated evaluation",
    )
    
    args = parser.parse_args()
    
    if args.judge:
        run_automated_judge_demo(judge_model=args.judge_model)
    elif args.full:
        test_system()
    else:
        quick_test()
        print("\nFor full demo with documents, run: python test_demo.py --full")
        print("For automated Local-vs-Hybrid judge evaluation, run: python test_demo.py --judge")
