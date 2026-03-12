from modules.vector_store import VectorStoreManager
from modules.query_abstraction import QueryAbstrator
from modules.cloud_planner import CloudPlanner
from modules.local_executor import LocalExecutor

class HybridOrchestrator:
    def __init__(self):
        self.vector_store = VectorStoreManager()
        self.abstraction = QueryAbstrator()
        self.planner = CloudPlanner()
        self.executor = LocalExecutor()

    def process_query_hybrid(self, query):
        """Complete hybrid workflow."""
        # 1. Abstract the query
        abstract_prompt, masked_query = self.abstraction.abstract_query(query)
        
        # 2. Get reasoning plan from cloud
        reasoning_plan = self.planner.generate_plan(abstract_prompt)
        
        # 3. Retrieve relevant local documents
        context_docs = self.vector_store.similarity_search(query)
        
        # 4. Execute plan locally
        final_answer = self.executor.execute_plan(query, reasoning_plan, context_docs)
        
        return {
            "masked_query": masked_query,
            "reasoning_plan": reasoning_plan,
            "final_answer": final_answer,
            "context_sources": [doc.metadata.get("source", "Unknown") for doc in context_docs]
        }

    def process_query_local_only(self, query):
        """Baseline local-only workflow for comparison."""
        context_docs = self.vector_store.similarity_search(query)
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = (
            f"Question: {query}\n\n"
            f"Context: {context_text}\n\n"
            "Please answer the question based strictly on the context provided."
        )
        
        return self.executor.llm.invoke(prompt)
