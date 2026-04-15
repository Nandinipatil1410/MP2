import sys
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import LOCAL_MODEL, CLOUD_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_abstractor import QueryAbstractor
from cloud_planner import CloudReasoningPlanner
from local_llm import LocalLLMExecutor

from metrics import MetricsTracker


class HybridLLMOrchestrator:
    
    def __init__(self, groq_api_key: str = None):
        print("🚀 Initializing Privacy-Preserving Hybrid LLM System...")
        
        self.doc_processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.vector_store = VectorStore()
        self.query_abstractor = QueryAbstractor()
        self.cloud_planner = CloudReasoningPlanner(api_key=groq_api_key, model=CLOUD_MODEL)
        self.local_executor = LocalLLMExecutor()
        
        self.documents_loaded = False
        
        print("✓ All components initialized")
    
    def load_documents(self, file_paths: List[str]):
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

    def get_system_stats(self) -> Dict[str, any]:
        """Get system statistics for the dashboard"""
        return {
            'documents_count': len(self.vector_store.documents) if hasattr(self.vector_store, 'documents') else 0,
            'chunks_count': self.vector_store.index.ntotal if (hasattr(self.vector_store, 'index') and self.vector_store.index) else 0,
            'is_ready': self.documents_loaded
        }

    def process_query_hybrid(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, any]:
        
        metrics = MetricsTracker()
        total_start = time.time()

        if not self.documents_loaded:
            return {'success': False, 'error': 'No documents loaded.', 'mode': 'hybrid'}

        print(f"\n🔍 HYBRID MODE: {query[:100]}")

        # Query abstraction
        abstracted_query, abstraction_meta = self.query_abstractor.abstract_query(query)
        privacy_score = self.query_abstractor.calculate_privacy_score(abstracted_query)

        # Cloud planning
        plan_result = self.cloud_planner.generate_reasoning_plan(abstracted_query)
        reasoning_plan = plan_result.get('plan', [])

        # Retrieval (timed)
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        metrics.retrieval_time = time.time() - retrieval_start

        # Generation (timed)
        generation_start = time.time()
        execution_result = self.local_executor.execute_plan(
            reasoning_plan,
            retrieved_docs,
            query
        )
        metrics.generation_time = time.time() - generation_start

        metrics.total_time = time.time() - total_start
        metrics.print_report()

        return {
            'success': execution_result['success'],
            'answer': execution_result.get('answer', ''),
            'mode': 'hybrid',
            'abstracted_query': abstracted_query,
            'latency': metrics.total_time,
            'privacy_score': privacy_score,
            'reasoning_depth': len(reasoning_plan)
        }

    def process_query_local_only(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> Dict[str, any]:
        
        metrics = MetricsTracker()
        total_start = time.time()

        if not self.documents_loaded:
            return {'success': False, 'error': 'No documents loaded.', 'mode': 'local'}

        print(f"\n🔍 LOCAL MODE: {query[:100]}")

        # Retrieval (timed)
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.search(query, top_k=top_k)
        metrics.retrieval_time = time.time() - retrieval_start

        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"

        # Generation (timed)
        generation_start = time.time()
        answer = self.local_executor.generate_simple(prompt)
        metrics.generation_time = time.time() - generation_start

        metrics.total_time = time.time() - total_start
        metrics.print_report()

        return {
            'success': True,
            'answer': answer,
            'mode': 'local',
            'abstracted_query': query, # No abstraction in local
            'latency': metrics.total_time,
            'privacy_score': 1.0,
            'reasoning_depth': 2
        }

    def process_query_cloud_only(self, query: str) -> Dict[str, any]:
        """Process query using only cloud LLM (no RAG)"""
        metrics = MetricsTracker()
        total_start = time.time()
        
        # Cloud generation (no retrieval)
        gen_start = time.time()
        try:
            answer = self.cloud_planner.get_completion(
                "You are a helpful expert assistant.", 
                query
            )
            success = True
        except Exception as e:
            answer = f"Cloud execution error: {str(e)}"
            success = False
            
        metrics.generation_time = time.time() - gen_start
        metrics.total_time = time.time() - total_start
        
        return {
            'success': success,
            'answer': answer,
            'mode': 'cloud-only',
            'abstracted_query': query, # No abstraction in cloud baseline
            'latency': metrics.total_time,
            'privacy_score': 0.1, # Sending full query to cloud
            'reasoning_depth': 1
        }

    def estimate_reasoning_depth(self, result: Dict[str, any]) -> int:
        """Estimate reasoning depth based on answer length and structure"""
        answer = result.get('answer', '')
        if not answer: return 0
        
        depth = 1
        if len(answer) > 200: depth += 1
        if len(answer) > 500: depth += 1
        if "\n" in answer: depth += 1
        if "." in answer: depth += 1
        
        return min(depth, 5)

    def compare_approaches(self, query: str) -> Dict[str, any]:
        """Compare all three processing modes in parallel"""
        print(f"\n📊 --- Comparing All Approaches for: {query} ---")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            hybrid_future = executor.submit(self.process_query_hybrid, query)
            local_future = executor.submit(self.process_query_local_only, query)
            cloud_future = executor.submit(self.process_query_cloud_only, query)
            
            hybrid_result = hybrid_future.result()
            local_result = local_future.result()
            cloud_result = cloud_future.result()
            
        # Re-estimate depth for cloud-only and local-only if needed
        local_result['reasoning_depth'] = self.estimate_reasoning_depth(local_result)
        cloud_result['reasoning_depth'] = self.estimate_reasoning_depth(cloud_result)
        
        return {
            'query': query,
            'hybrid': hybrid_result,
            'local_only': local_result,
            'cloud_only': cloud_result
        }
