import sys
from typing import Dict, List
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import LOCAL_MODEL, CLOUD_MODEL
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_abstractor import QueryAbstractor
from cloud_planner import CloudReasoningPlanner
from local_llm import LocalLLMExecutor

from metrics import MetricsTracker


class HybridLLMOrchestrator:
    
    def __init__(self, groq_api_key: str = None):
        print("🚀 Initializing Privacy-Preserving Hybrid LLM System...")
        
        self.doc_processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
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

    def process_query_hybrid(self, query: str, top_k: int = 3) -> Dict[str, any]:
        
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
            'latency': metrics.total_time,
            'privacy_score': privacy_score,
            'reasoning_depth': len(reasoning_plan)
        }

    def process_query_local_only(self, query: str, top_k: int = 3) -> Dict[str, any]:
        
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
            'latency': metrics.total_time
        }
