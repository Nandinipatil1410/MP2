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
        print(" Initializing Privacy-Preserving Hybrid LLM System...")
        
        self.doc_processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.vector_store = VectorStore()
        self.query_abstractor = QueryAbstractor()
        self.cloud_planner = CloudReasoningPlanner(api_key=groq_api_key, model=CLOUD_MODEL)
        self.local_executor = LocalLLMExecutor()
        
        self.documents_loaded = False
        self.initialized = True
        
        print(" All components initialized")
    
    def load_documents(self, file_paths: List[str]):
        print(f"\n Processing {len(file_paths)} documents...")
        
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.doc_processor.process_and_chunk(file_path)
                all_chunks.extend(chunks)
                print(f"   Processed: {Path(file_path).name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")
        
        if all_chunks:
            self.vector_store.build_index(all_chunks)
            self.documents_loaded = True
            print(f"\n Loaded {len(all_chunks)} document chunks into vector store")
            return True
        return False

    def get_system_stats(self) -> Dict[str, any]:
        """Get system statistics for the dashboard"""
        vector_stats = self.vector_store.get_stats()
        return {
            'documents_loaded': self.documents_loaded,
            'is_ready': self.documents_loaded,
            'documents_count': vector_stats['total_documents'],
            'chunks_count': vector_stats['total_documents'],
            'vector_store': vector_stats,
            'local_model': LOCAL_MODEL,
            'cloud_model': CLOUD_MODEL
        }

    def get_document_list(self) -> List[str]:
        """Get unique list of loaded document filenames"""
        if not self.documents_loaded:
            return []
        sources = {doc.get('source') for doc in self.vector_store.documents if doc.get('source')}
        return sorted(list(sources))

    def process_query_hybrid(self, query: str, top_k: int = TOP_K_RETRIEVAL, expert_mode: bool = False, selected_document: str = None) -> Dict[str, any]:
        """
        State-of-the-art hybrid reasoning:
        1. Local abstraction of PII
        2. Cloud reasoning planning (no private data)
        3. Local execution of plan on private data
        """
        metrics = MetricsTracker()
        total_start = time.time()

        if not self.documents_loaded:
            return {'success': False, 'error': 'No documents loaded.', 'mode': 'hybrid'}

        print(f"\n HYBRID MODE: {query[:100]}")

        # 1. Local Abstraction
        abstracted_query, abstraction_meta = self.query_abstractor.abstract_query(query)
        privacy_score = self.query_abstractor.calculate_privacy_score(query)
        
        # 2. Cloud Planning (Privacy-Preserving)
        print(f"   Generating reasoning plan for intent: '{abstracted_query[:60]}...'")
        plan_result = self.cloud_planner.generate_reasoning_plan(abstracted_query, expert_mode=expert_mode)
        reasoning_plan = plan_result.get('plan', [])

        print(f"\n🧠 CLOUD REASONING PLAN (Expert Mode: {expert_mode})")
        print("="*60)
        for i, step in enumerate(reasoning_plan, 1):
            print(f"  Step {i}: {step}")
        print("="*60 + "\n")

        # Retrieval (timed)
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.search(query, top_k=top_k, source=selected_document)
        
        # Improvement: If a specific document is selected, bridge the 'Head' (intro)
        # This fixes Step 1 failures where the semantic query misses the introduction.
        if selected_document and len(self.vector_store.documents) > 0:
            doc_all_chunks = [d for d in self.vector_store.documents if d.get('source') == selected_document]
            doc_all_chunks.sort(key=lambda x: x.get('chunk_id', 0))
            head_chunks = doc_all_chunks[:3] # Get first 3 chunks (Intro/Framework)
            
            # Merge without duplicates (using text-hash-like ID check)
            existing_texts = {d['text'] for d in retrieved_docs}
            new_head = [hc for hc in head_chunks if hc['text'] not in existing_texts]
            retrieved_docs = new_head + retrieved_docs
            
            # Trim to top_k + 2 to keep context manageable
            retrieved_docs = retrieved_docs[:top_k + 2]

        metrics.retrieval_time = time.time() - retrieval_start

        # Generation (timed)
        generation_start = time.time()
        execution_result = self.local_executor.execute_plan(
            reasoning_plan,
            retrieved_docs,
            query,
            expert_mode=expert_mode,
            vector_store=self.vector_store,
            selected_document=selected_document,
            cloud_planner=self.cloud_planner
        )
        metrics.generation_time = time.time() - generation_start

        metrics.total_time = time.time() - total_start
        metrics.print_report()

        # Completeness scoring (0-1)
        # Based on reasoning depth and answer verbosity
        completeness_score = min(1.0, (len(reasoning_plan) * 0.1) + (len(execution_result.get('answer', '')) / 1000) * 0.4)

        return {
            'success': execution_result['success'],
            'answer': execution_result.get('answer', ''),
            'mode': 'hybrid',
            'abstracted_query': abstracted_query,
            'latency': metrics.total_time,
            'privacy_score': privacy_score,
            'completeness_score': completeness_score,
            'reasoning_depth': len(reasoning_plan),
            'reasoning_plan': reasoning_plan,
            'documents_used': [doc.get('source', 'Unknown') for doc in retrieved_docs],
            'retrieved_context': retrieved_docs
        }

    def process_query_local_only(self, query: str, top_k: int = TOP_K_RETRIEVAL, selected_document: str = None) -> Dict[str, any]:
        
        metrics = MetricsTracker()
        total_start = time.time()

        if not self.documents_loaded:
            return {'success': False, 'error': 'No documents loaded.', 'mode': 'local'}

        print(f"\n LOCAL MODE: {query[:100]}")

        # Retrieval (timed)
        retrieval_start = time.time()
        retrieved_docs = self.vector_store.search(query, top_k=top_k, source=selected_document)
        metrics.retrieval_time = time.time() - retrieval_start

        context = "\n\n".join([doc['text'] for doc in retrieved_docs])
        
        # Hardened one-shot prompt to prevent conversational "Sure, I'd be happy to help" responses
        prompt = f"""### SYSTEM INSTRUCTION:
You are a specialized document reasoning agent. Answer the question below based ONLY on the provided context. 
If the information is not present, explicitly state that you cannot find the answer in the loaded documents.
DO NOT provide generic greetings, DO NOT offer general assistance, and DO NOT use conversational filler.

### CONTEXT:
{context if context.strip() else "[No relevant information found in retrieved documents]"}

### QUESTION:
{query}

### FINAL ANSWER:"""

        # Generation (timed)
        generation_start = time.time()
        answer = self.local_executor.generate_simple(prompt)
        metrics.generation_time = time.time() - generation_start

        metrics.total_time = time.time() - total_start
        metrics.print_report()

        # For local, completeness is purely based on answer quality
        completeness_score = min(0.8, (len(answer) / 1000) * 0.5)

        return {
            'success': True,
            'answer': answer,
            'mode': 'local',
            'abstracted_query': query, # No abstraction in local
            'latency': metrics.total_time,
            'privacy_score': 1.0,
            'completeness_score': completeness_score,
            'reasoning_depth': 2,
            'documents_used': [doc.get('source', 'Unknown') for doc in retrieved_docs],
            'retrieved_context': retrieved_docs
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

    def compare_approaches(self, query: str, expert_mode: bool = False, selected_document: str = None) -> Dict[str, any]:
        """Run and compare all processing modes"""
        print(f"\n📊 --- Comparing All Approaches for: {query} ---")
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            hybrid_future = executor.submit(self.process_query_hybrid, query, expert_mode=expert_mode, selected_document=selected_document)
            local_future = executor.submit(self.process_query_local_only, query, selected_document=selected_document)
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
