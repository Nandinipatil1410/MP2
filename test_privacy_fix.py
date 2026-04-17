
import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from orchestrator import HybridLLMOrchestrator

load_dotenv()

def test_nandini_patil_grounding():
    print("\n🔍 Testing Grounding for 'Nandini Patil' (Private Data)...")
    
    # 1. Create a dummy document with the name
    Path("./data/documents").mkdir(parents=True, exist_ok=True)
    with open("./data/documents/patient_nandini.txt", "w") as f:
        f.write("Patient Name: Nandini Patil. Diagnosis: Mild seasonal allergies. Recommendation: Over-the-counter antihistamines.")
    
    # 2. Initialize Orchestrator
    orchestrator = HybridLLMOrchestrator(groq_api_key=os.getenv("GROQ_API_KEY"))
    orchestrator.load_documents(["./data/documents/patient_nandini.txt"])
    
    # 3. Process Query
    query = "Who is the patient and what is their diagnosis?"
    print(f"Query: {query}")
    
    result = orchestrator.process_query_hybrid(query)
    
    print("\n" + "="*50)
    print("RESULT")
    print("="*50)
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer']}")
    
    if "REDACTED" in result['answer']:
        print("\n❌ FAILURE: Answer was redacted (False Positive Hallucination).")
    elif "Nandini Patil" in result['answer']:
        print("\n✅ SUCCESS: Answer contains the restored name.")
        print("Note: If the cloud synthesis worked, it processed [NAME_1] but we restored it locally.")
    else:
        print("\n⚠️ UNKNOWN: Answer doesn't match expected pattern.")

if __name__ == "__main__":
    test_nandini_patil_grounding()
