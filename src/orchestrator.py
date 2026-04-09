"""
Main Orchestrator
Coordinates the entire privacy-preserving hybrid LLM pipeline
"""
from typing import Dict, List
import time
from pathlib import Path

from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_abstractor import QueryAbstractor
from cloud_planner import CloudReasoningPlanner
from local_llm import LocalLLMExecutor


class HybridLLMOrchestrator:
    """Main orchestrator for the privacy-preserving system"""
    
    def __init__(self, groq_api_key: str = None):
        """Initialize all components"""
        print("🚀 Initializing Privacy-Preserving Hybrid LLM System...")
        
        # Initialize components
        self.doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        self.vector_store = VectorStore()
        self.query_abstractor = QueryAbstractor()
        self.cloud_planner = CloudReasoningPlanner(api_key=groq_api_key)
        self.local_executor = LocalLLMExecutor()
        
        # State
        self.documents_loaded = False
        
        print("✓ All components initialized")
    
    def load_documents(self, file_paths: List[str]):
        """Load and process documents into vector store"""
        print(f"\n📄 Processing {len(file_paths)} documents...")
        
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.doc_processor.process_and_chunk(file_path)
                all_chunks.extend(chunks)
                print(f"  ✓ Processed: {Path(file_path).name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  ✗ Error processing {file_path}: {e}")
        
        if all_chunks:
            self.vector_store.build_index(all_chunks)
            self.documents_loaded = True
            print(f"\n✓ Loaded {len(all_chunks)} document chunks into vector store")
        else:
            print("\n⚠️  No documents were successfully processed")
    
    def process_query_hybrid(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """
        Process query using HYBRID approach:
        1. Abstract query
        2. Get reasoning plan from cloud
        3. Execute locally with private documents
        """
        start_time = time.time()
        
        if not self.documents_loaded:
            return {
                'success': False,
                'error': 'No documents loaded. Please load documents first.',
                'mode': 'hybrid'
            }
        
        print(f"\n🔍 Processing Query (HYBRID MODE): {query[:100]}...")
        
        # Step 1: Abstract the query
        print("  1️⃣  Abstracting query...")
        abstracted_query, abstraction_meta = self.query_abstractor.abstract_query(query)
        privacy_score = self.query_abstractor.calculate_privacy_score(abstracted_query)
        print(f"     Privacy Score: {privacy_score:.2f}")
        
        # Step 2: Get reasoning plan from cloud (no private data sent)
        print("  2️⃣  Generating reasoning plan (cloud)...")
        plan_result = self.cloud_planner.generate_reasoning_plan(abstracted_query)
        reasoning_plan = plan_result.get('plan', [])
        print(f"     Generated {len(reasoning_plan)} reasoning steps")
        
        # Step 3: Retrieve relevant documents locally
        print("  3️⃣  Retrieving relevant documents (local)...")
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        print(f"     Retrieved {len(retrieved_docs)} documents")
        
        # Step 4: Execute plan locally with private documents
        print("  4️⃣  Executing plan (local LLM)...")
        execution_result = self.local_executor.execute_plan(
            reasoning_plan, 
            retrieved_docs, 
            query
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"  ✓ Complete in {latency:.2f}s")
        
        # Calculate heuristics
        answer = execution_result.get('answer', '')
        completeness = min(1.0, len(answer.split()) / 50.0) if len(answer) > 0 else 0.0
        
        return {
            'success': execution_result['success'],
            'answer': answer,
            'mode': 'hybrid',
            'abstracted_query': abstracted_query,
            'original_query': query,
            'reasoning_plan': reasoning_plan,
            'reasoning_depth': len(reasoning_plan),
            'completeness_score': completeness,
            'documents_used': len(retrieved_docs),
            'privacy_score': privacy_score,
            'latency': latency,
            'abstraction_metadata': abstraction_meta
        }
    
    def process_query_local_only(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """
        Process query using LOCAL-ONLY approach:
        Everything happens locally without cloud
        """
        start_time = time.time()
        
        if not self.documents_loaded:
            return {
                'success': False,
                'error': 'No documents loaded. Please load documents first.',
                'mode': 'local_only'
            }
        
        print(f"\n🔍 Processing Query (LOCAL-ONLY MODE): {query[:100]}...")
        
        # Retrieve documents
        print("  1️⃣  Retrieving relevant documents...")
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        print(f"     Retrieved {len(retrieved_docs)} documents")
        
        # Create simple local prompt
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        prompt = f"""Answer the following question based on the provided context.

Question: {query}

Context:
{context}

Answer:"""
        
        # Execute locally
        print("  2️⃣  Generating answer (local LLM)...")
        answer = self.local_executor.generate_simple(prompt)
        
        end_time = time.time()
        latency = end_time - start_time
        
        print(f"  ✓ Complete in {latency:.2f}s")
        
        # Calculate heuristics
        completeness = min(1.0, len(answer.split()) / 50.0) if len(answer) > 0 else 0.0
        
        return {
            'success': True,
            'answer': answer,
            'mode': 'local_only',
            'reasoning_depth': 1.0,  # Local only is single step
            'completeness_score': completeness,
            'documents_used': len(retrieved_docs),
            'privacy_score': 1.0,  # Perfect privacy (all local)
            'latency': latency
        }
    
    def process_query_cloud_only(self, query: str, top_k: int = 3) -> Dict[str, any]:
        """
        Process query using CLOUD-ONLY approach:
        Sends private data directly to cloud (for comparison/risk demo)
        """
        start_time = time.time()
        
        print(f"\n🔍 Processing Query (CLOUD-ONLY MODE): {query[:100]}...")
        
        # Retrieve relevant documents locally to send to cloud
        retrieved_docs = []
        if self.documents_loaded:
            print("  1️⃣  Retrieving private documents to send to cloud...")
            retrieved_docs = self.vector_store.search(query, top_k=top_k)
        
        # Create prompt with full context (Privacy RISK!)
        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        system_prompt = "You are a helpful assistant. Answer the question based ONLY on the provided context."
        user_prompt = f"Question: {query}\n\nContext:\n{context}"
        
        # Execute on cloud
        print("  2️⃣  Generating answer (cloud LLM - PRIVACY RISK)...")
        answer = self.cloud_planner.get_completion(system_prompt, user_prompt)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Calculate heuristics
        completeness = min(1.0, len(answer.split()) / 50.0) if len(answer) > 0 else 0.0
        
        return {
            'success': True,
            'answer': answer,
            'mode': 'cloud_only',
            'reasoning_depth': 1.5,  # Cloud direct is better than local but less than planned hybrid
            'completeness_score': completeness,
            'privacy_score': 0.0,  # No privacy (data sent to cloud)
            'latency': latency,
            'warning': '⚠️ DATA EXPOSURE: Private context was sent to cloud provider'
        }
    
    def estimate_reasoning_depth(self, result: Dict[str, any]) -> float:
        """Estimate the depth/quality of reasoning (0.0 - 1.0)"""
        if not result.get('success', False):
            return 0.0
        
        mode = result.get('mode', '')
        answer = result.get('answer', '')
        
        # Base depth based on mode
        if mode == 'hybrid':
            # Hybrid mode uses cloud planning, so it's generally deeper
            depth = 0.8
            # Add value for specific steps if plan exists
            plan_len = len(result.get('reasoning_plan', []))
            if plan_len > 0:
                depth += min(0.2, plan_len * 0.05)
        elif mode == 'cloud_only':
            # Cloud only is the most capable
            depth = 0.95
        else:
            # Local only is simpler
            depth = 0.5
            
        # Refine based on answer length and structure
        if len(answer.split()) > 100:
            depth += 0.05
        if "\n" in answer: # List-like or structured
            depth += 0.05
            
        return min(1.0, depth)

    def compare_approaches(self, query: str) -> Dict[str, Dict]:
        """Compare all three approaches with enriched metrics"""
        print("\n" + "="*60)
        print("COMPARING ALL APPROACHES")
        print("="*60)
        
        results = {}
        
        # Hybrid
        results['hybrid'] = self.process_query_hybrid(query)
        results['hybrid']['reasoning_depth'] = self.estimate_reasoning_depth(results['hybrid'])
        
        # Local-only
        results['local_only'] = self.process_query_local_only(query)
        results['local_only']['reasoning_depth'] = self.estimate_reasoning_depth(results['local_only'])
        
        # Cloud-only (simulated)
        results['cloud_only'] = self.process_query_cloud_only(query)
        results['cloud_only']['reasoning_depth'] = self.estimate_reasoning_depth(results['cloud_only'])
        
        return results
    
    def get_system_stats(self) -> Dict[str, any]:
        """Get system statistics"""
        return {
            'vector_store': self.vector_store.get_stats(),
            'documents_loaded': self.documents_loaded,
            'local_model': self.local_executor.model,
            'cloud_model': self.cloud_planner.model
        }


if __name__ == "__main__":
    # Test the orchestrator
    orchestrator = HybridLLMOrchestrator()
    
    print("\n" + "="*60)
    print("Privacy-Preserving Hybrid LLM System - Test")
    print("="*60)
    
    # Display stats
    stats = orchestrator.get_system_stats()
    print(f"\nSystem Stats:")
    print(f"  Documents loaded: {stats['documents_loaded']}")
    print(f"  Local model: {stats['local_model']}")
    print(f"  Cloud model: {stats['cloud_model']}")
