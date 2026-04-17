"""
Hybrid LLM Orchestrator
Intent-driven, metadata-aware coordinator for the RAG pipeline.
Routes queries based on LLM-classified intent — no domain-based branching.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.settings import LOCAL_MODEL, CLOUD_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL
from document_processor import DocumentProcessor
from vector_store import VectorStore
from query_abstractor import QueryAbstractor
from cloud_planner import CloudReasoningPlanner
from local_llm import LocalLLMExecutor
from metrics import MetricsTracker

# New pipeline modules
from pipeline.intent_classifier import classify_query_intent
from pipeline.router import route, uses_planner
from pipeline.metadata_extractor import extract_document_metadata
from pipeline.executor import execute_steps
from pipeline.validator import validate


class HybridLLMOrchestrator:

    def __init__(self, groq_api_key: str = None):
        print(" Initializing Intent-Driven Hybrid LLM System...")

        self.doc_processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.vector_store = VectorStore()
        self.query_abstractor = QueryAbstractor()         # kept for PII masking
        self.cloud_planner = CloudReasoningPlanner(api_key=groq_api_key, model=CLOUD_MODEL)
        self.local_executor = LocalLLMExecutor()

        self.documents_loaded = False
        self.initialized = True

        print(" All components initialized")

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def load_documents(self, file_paths: List[str]) -> bool:
        print(f"\n Processing {len(file_paths)} documents...")
        all_chunks = []
        for file_path in file_paths:
            try:
                chunks = self.doc_processor.process_and_chunk(file_path)
                all_chunks.extend(chunks)
                print(f"  Processed: {Path(file_path).name} ({len(chunks)} chunks)")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")

        if all_chunks:
            self.vector_store.build_index(all_chunks)
            self.documents_loaded = True
            print(f"\n Loaded {len(all_chunks)} document chunks into vector store")
            return True
        return False

    def get_document_list(self) -> List[str]:
        if not self.documents_loaded:
            return []
        sources = {doc.get("source") for doc in self.vector_store.documents if doc.get("source")}
        return sorted(list(sources))

    def get_system_stats(self) -> Dict[str, Any]:
        vector_stats = self.vector_store.get_stats()
        return {
            "documents_loaded": self.documents_loaded,
            "is_ready": self.documents_loaded,
            "documents_count": vector_stats["total_documents"],
            "chunks_count": vector_stats["total_documents"],
            "vector_store": vector_stats,
            "local_model": LOCAL_MODEL,
            "cloud_model": CLOUD_MODEL,
        }

    # ------------------------------------------------------------------
    # Main entry: intent-driven routing
    # ------------------------------------------------------------------

    def process_query_hybrid(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        expert_mode: bool = False,
        selected_document: Optional[str] = None,
        domain: str = "general",          # accepted but ignored — domain-free
    ) -> Dict[str, Any]:
        """
        Intent-Driven Hybrid Pipeline:
          1. Classify intent (LLM → Groq, fallback → keywords)
          2. Route to execution path
          3. For reasoning/comparison: extract metadata → plan → step-wise execute
          4. Synthesize → Validate → Return grounded answer
        """
        metrics = MetricsTracker()
        total_start = time.time()

        if not self.documents_loaded:
            return {"success": False, "error": "No documents loaded.", "mode": "hybrid"}

        print(f"\n HYBRID MODE: {query[:100]}")

        # ── Step 1: Intent Classification ────────────────────────────────
        intent = classify_query_intent(query, groq_client=self.cloud_planner.client)
        path = route(intent)
        print(f"  → Intent: '{intent}' | Path: '{path}'")

        # Dynamically scale context window based on intent and mode
        if expert_mode:
            if intent in ("reasoning", "comparison"):
                top_k = max(top_k, 20)
            elif intent == "summary":
                top_k = max(top_k, 15)

        # ── Step 2: Route to fast paths (no planner needed) ──────────────
        if path == "summary_path":
            return self._run_summary(query, top_k, selected_document, mode="hybrid",
                                     intent=intent, metrics=metrics, total_start=total_start)

        if path == "extraction_path":
            return self._run_extraction(query, top_k, selected_document, mode="hybrid",
                                        intent=intent, metrics=metrics, total_start=total_start)

        # ── Step 3: Reasoning / Comparison path (uses planner) ───────────
        return self._run_reasoning(
            query=query,
            intent=intent,
            path=path,
            top_k=top_k,
            expert_mode=expert_mode,
            selected_document=selected_document,
            mode="hybrid",
            metrics=metrics,
            total_start=total_start,
        )

    def process_query_local_only(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        selected_document: Optional[str] = None,
        domain: str = "general",          # accepted but ignored
    ) -> Dict[str, Any]:
        """Local-only mode: intent-driven routing without any cloud calls."""
        metrics = MetricsTracker()
        total_start = time.time()

        if not self.documents_loaded:
            return {"success": False, "error": "No documents loaded.", "mode": "local"}

        print(f"\n LOCAL MODE: {query[:100]}")

        # Use keyword fallback for intent (no cloud in local-only mode)
        from pipeline.intent_classifier import _classify_with_keywords
        intent = _classify_with_keywords(query)
        path = route(intent)
        
        # Dynamically scale context window (local models might struggle with 20 chunks, cap at 10)
        if intent in ("reasoning", "comparison", "summary"):
            top_k = max(top_k, 10)

        if path == "summary_path":
            return self._run_summary(query, top_k, selected_document, mode="local",
                                     intent=intent, metrics=metrics, total_start=total_start)

        if path == "extraction_path":
            return self._run_extraction(query, top_k, selected_document, mode="local",
                                        intent=intent, metrics=metrics, total_start=total_start)

        # Reasoning/comparison in local-only: agentic steps, no cloud planner
        retrieval_start = time.time()
        docs = self._retrieve(query, query, top_k, selected_document)
        metrics.retrieval_time = time.time() - retrieval_start

        context = self.local_executor._prepare_context(docs)

        prompt = f"""### SYSTEM INSTRUCTION:
You are a document reasoning agent. Answer the question below based ONLY on the provided context.
If the information is not present, explicitly state: "Not stated in the loaded documents."
Do NOT provide generic greetings or conversational filler.

### CONTEXT:
{context if context.strip() else "[No relevant information found]"}

### QUESTION:
{query}

### FINAL ANSWER:"""

        generation_start = time.time()
        answer = self.local_executor.generate_simple(prompt)
        answer = self.local_executor._clean_response(answer)
        metrics.generation_time = time.time() - generation_start
        metrics.total_time = time.time() - total_start
        metrics.print_report()

        return {
            "success": True,
            "answer": answer,
            "mode": "local",
            "intent": intent,
            "abstracted_query": query,
            "latency": metrics.total_time,
            "privacy_score": 1.0,
            "completeness_score": min(0.8, len(answer) / 1000 * 0.5),
            "reasoning_depth": 2,
            "documents_used": [d.get("source", "Unknown") for d in docs],
            "retrieved_context": docs,
        }

    def process_query_cloud_only(self, query: str) -> Dict[str, Any]:
        """Cloud-only baseline mode (no RAG — used for benchmarking)."""
        metrics = MetricsTracker()
        total_start = time.time()

        gen_start = time.time()
        try:
            answer = self.cloud_planner.get_completion(
                "You are a helpful expert assistant.", query
            )
            success = True
        except Exception as e:
            answer = f"Cloud execution error: {str(e)}"
            success = False

        metrics.generation_time = time.time() - gen_start
        metrics.total_time = time.time() - total_start

        return {
            "success": success,
            "answer": answer,
            "mode": "cloud-only",
            "abstracted_query": query,
            "latency": metrics.total_time,
            "privacy_score": 0.1,
            "reasoning_depth": 1,
        }

    def compare_approaches(
        self,
        query: str,
        expert_mode: bool = False,
        selected_document: Optional[str] = None,
        domain: str = "general",   # accepted but ignored
    ) -> Dict[str, Any]:
        """Run and compare all processing modes (for benchmark UI)."""
        print(f"\n📊 Comparing All Approaches for: {query}")

        with ThreadPoolExecutor(max_workers=3) as executor:
            hybrid_future = executor.submit(
                self.process_query_hybrid, query,
                top_k=TOP_K_RETRIEVAL, expert_mode=expert_mode,
                selected_document=selected_document,
            )
            local_future = executor.submit(
                self.process_query_local_only, query,
                top_k=TOP_K_RETRIEVAL, selected_document=selected_document,
            )
            cloud_future = executor.submit(self.process_query_cloud_only, query)

            hybrid_result = hybrid_future.result()
            local_result = local_future.result()
            cloud_result = cloud_future.result()

        local_result["reasoning_depth"] = self.estimate_reasoning_depth(local_result)
        cloud_result["reasoning_depth"] = self.estimate_reasoning_depth(cloud_result)

        return {
            "query": query,
            "hybrid": hybrid_result,
            "local_only": local_result,
            "cloud_only": cloud_result,
        }

    def estimate_reasoning_depth(self, result: Dict[str, Any]) -> int:
        answer = result.get("answer", "")
        if not answer:
            return 0
        depth = 1
        if len(answer) > 200: depth += 1
        if len(answer) > 500: depth += 1
        if "\n" in answer: depth += 1
        if "." in answer: depth += 1
        return min(depth, 5)

    # ------------------------------------------------------------------
    # Execution paths
    # ------------------------------------------------------------------

    def _run_summary(
        self, query, top_k, selected_document, mode, intent, metrics, total_start
    ) -> Dict[str, Any]:
        """Fast path: retrieve + summarise (no planner)."""
        retrieval_start = time.time()
        docs = self._retrieve(
            query, "abstract introduction conclusion overview summary", top_k + 3, selected_document
        )
        metrics.retrieval_time = time.time() - retrieval_start

        context = self.local_executor._prepare_context(docs)

        generation_start = time.time()
        answer = self.local_executor.create_overview(
            context=context, selected_document=selected_document
        )
        metrics.generation_time = time.time() - generation_start
        metrics.total_time = time.time() - total_start
        metrics.print_report()

        return {
            "success": True,
            "answer": answer,
            "mode": mode,
            "intent": intent,
            "abstracted_query": query,
            "latency": metrics.total_time,
            "privacy_score": 1.0,
            "completeness_score": 0.6,
            "validation": {"valid": True, "score": 0.7, "critique": "Summary route — no cloud validation."},
            "reasoning_depth": 0,
            "documents_used": list({d.get("source", "Unknown") for d in docs}),
            "retrieved_context": docs[:10],
        }

    def _run_extraction(
        self, query, top_k, selected_document, mode, intent, metrics, total_start
    ) -> Dict[str, Any]:
        """Fast path: retrieve + grounded direct answer (no planner)."""
        retrieval_start = time.time()
        docs = self._retrieve(query, query, top_k, selected_document)
        metrics.retrieval_time = time.time() - retrieval_start

        context = self.local_executor._prepare_context(docs)

        generation_start = time.time()
        answer = self.local_executor.answer_with_context(query=query, context=context)
        metrics.generation_time = time.time() - generation_start
        metrics.total_time = time.time() - total_start
        metrics.print_report()

        return {
            "success": True,
            "answer": answer,
            "mode": mode,
            "intent": intent,
            "abstracted_query": query,
            "latency": metrics.total_time,
            "privacy_score": 1.0,
            "completeness_score": min(0.8, len(answer) / 1000 * 0.5),
            "validation": {"valid": True, "score": 0.7, "critique": "Extraction route — direct grounded answer."},
            "reasoning_depth": 1,
            "documents_used": [d.get("source", "Unknown") for d in docs],
            "retrieved_context": docs[:10],
        }

    def _run_reasoning(
        self,
        query: str,
        intent: str,
        path: str,
        top_k: int,
        expert_mode: bool,
        selected_document: Optional[str],
        mode: str,
        metrics: MetricsTracker,
        total_start: float,
    ) -> Dict[str, Any]:
        """
        Full reasoning/comparison path:
          1. Abstract query (PII removal)
          2. Retrieve initial context
          3. Extract document metadata (local)
          4. Generate structured plan (cloud, metadata-aware)
          5. Step-wise execution (local retrieval + reasoning)
          6. Signal filtering
          7. Cloud synthesis → validation → repair if needed
        """
        # 1. Abstraction (PII removal for cloud safety)
        abstracted_query, _ = self.query_abstractor.abstract_query(query)
        privacy_score = self.query_abstractor.calculate_privacy_score(query)

        # 2. Initial retrieval for metadata extraction
        print("  → Retrieving initial context for metadata extraction...")
        retrieval_start = time.time()
        initial_docs = self._retrieve(query, abstracted_query, top_k, selected_document)
        initial_context = self.local_executor._prepare_context(initial_docs)

        # 3. Document metadata extraction (local model only)
        print("  → Extracting document metadata (local)...")
        metadata = extract_document_metadata(
            initial_context,
            ollama_url=self.local_executor.ollama_url,
            model=self.local_executor.model,
        )

        # 4. Generate structured plan (cloud receives only abstracted query + metadata)
        print("  → Generating structured plan (cloud, metadata-aware)...")
        plan_result = self.cloud_planner.generate_structured_plan(
            query=abstracted_query,
            metadata=metadata,
            expert_mode=expert_mode,
        )
        steps = plan_result.get("steps", [])

        print(f"\n REASONING PLAN ({len(steps)} steps):")
        for s in steps:
            print(f"   Step {s.get('id')}: [{s.get('action')}] → {s.get('target')}")
        print()

        # 5. Execution
        all_retrieved_docs = initial_docs
        memory = {"data": [], "intermediate_results": [], "insights": []}

        # Bypass step-wise local reasoning if we have a cloud planner in hybrid mode
        if mode == "hybrid" and self.cloud_planner.client is not None:
            print("  → [Hybrid Fast Path] Retrieving step context and synthesizing via cloud...")
            
            for step in steps:
                from pipeline.executor import _make_retrieval_query
                rq = _make_retrieval_query(step.get("action", ""), step.get("target", ""), query)
                docs = self._retrieve(query, rq, top_k, selected_document)
                all_retrieved_docs.extend(docs)
                
            metrics.retrieval_time = time.time() - retrieval_start
            
            # Deduplicate docs
            seen = set()
            unique_docs = []
            for d in all_retrieved_docs:
                key = (d.get("source"), d.get("chunk_id"), d.get("start_idx"))
                if key not in seen:
                    seen.add(key)
                    unique_docs.append(d)
            all_retrieved_docs = unique_docs
            
            context = self.local_executor._prepare_context(all_retrieved_docs)
            masked_context, entity_map, counters = self.local_executor._mask_entities(context)
            
            generation_start = time.time()
            final_answer = self.cloud_planner.synthesize(
                abstracted_query, masked_context, intent=intent, expert_mode=expert_mode
            )
            
            validation = {"valid": True, "score": 1.0, "critique": "Validation skipped."}
            if final_answer:
                print("  → Validating answer grounding...")
                validation = validate(
                    answer=final_answer,
                    context=masked_context,
                    query=abstracted_query,
                    groq_client=self.cloud_planner.client,
                )
                if not validation.get("valid") and validation.get("hallucination_detected"):
                    repaired = self.cloud_planner.synthesize_repair(
                        abstracted_query, masked_context, final_answer,
                        critique=validation.get("critique"),
                        expert_mode=expert_mode,
                    )
                    if repaired:
                        final_answer = repaired
                final_answer = self.local_executor._restore_entities(final_answer, entity_map)
            else:
                final_answer = "Cloud synthesis failed."

            metrics.generation_time = time.time() - generation_start
            metrics.total_time = time.time() - total_start
            metrics.print_report()
            
            completeness_score = min(1.0, len(steps) * 0.15 + len(final_answer) / 1000 * 0.3)
            return self._build_result(
                success=True, answer=final_answer, mode=mode, intent=intent,
                abstracted_query=abstracted_query, metrics=metrics, privacy_score=privacy_score,
                completeness_score=completeness_score, steps=steps, docs=all_retrieved_docs,
                memory=memory, validation=validation,
            )

        # Local-only execution path
        print("  → [Local Path] Executing step-wise reasoning loop...")
        exec_result = execute_steps(
            steps=steps,
            query=query,
            vector_store=self.vector_store,
            local_llm=self.local_executor,
            selected_document=selected_document,
            top_k=top_k,
            expert_mode=expert_mode,
        )
        metrics.retrieval_time = time.time() - retrieval_start

        all_retrieved_docs = initial_docs + exec_result["all_retrieved_docs"]
        combined_findings = exec_result["combined_findings"]
        memory = exec_result["memory"]

        # 6. Signal vs Noise filtering
        print("  → Filtering findings (signal vs noise)...")
        filtered = self.local_executor.filter_relevance(abstracted_query, combined_findings)
        signal_support = (filtered.get("signal") or []) + (filtered.get("support") or [])
        final_findings = "\n".join(signal_support) if signal_support else combined_findings

        # Fallback if nothing useful was extracted by the local model
        if not final_findings.strip() or "DATA_ABSENT" in combined_findings:
            print("  → Local reasoning failed to find signal, falling back to direct context answer...")
            fallback_docs = self._retrieve(query, abstracted_query, max(top_k, 8), selected_document)
            if fallback_docs:
                context = self.local_executor._prepare_context(fallback_docs)
                answer = self.local_executor.answer_with_context(query=query, context=context)
                metrics.total_time = time.time() - total_start
                metrics.print_report()
                
                return self._build_result(
                    success=True, answer=answer, mode=mode, intent=intent,
                    abstracted_query=abstracted_query, metrics=metrics, privacy_score=privacy_score,
                    completeness_score=0.4, steps=steps, docs=all_retrieved_docs + fallback_docs,
                    memory=memory,
                    validation={"valid": True, "score": 0.5, "critique": "Fallback: local synthesis from raw context."},
                )
            
            return self._build_result(
                success=True,
                answer="Insufficient data in document to answer this query.",
                mode=mode, intent=intent, abstracted_query=abstracted_query,
                metrics=metrics, privacy_score=privacy_score, completeness_score=0.1,
                steps=steps, docs=all_retrieved_docs, memory=memory,
                validation={"valid": True, "score": 0.2, "critique": "No relevant evidence retrieved."},
            )

        # 7. Local Synthesis
        print("  → Synthesizing final answer locally...")
        generation_start = time.time()
        # Local synthesis fallback
        synthesis_prompt = self.local_executor._create_synthesis_prompt(
            query, [final_findings], expert_mode=expert_mode
        )
        final_answer = self.local_executor._call_ollama(synthesis_prompt, max_tokens=600)
        final_answer = self.local_executor._clean_response(final_answer)
        validation = {"valid": True, "score": 1.0, "critique": "Local fallback synthesis."}

        metrics.generation_time = time.time() - generation_start
        metrics.total_time = time.time() - total_start
        metrics.print_report()

        completeness_score = min(1.0, len(steps) * 0.15 + len(final_answer) / 1000 * 0.3)

        return self._build_result(
            success=True, answer=final_answer, mode=mode, intent=intent,
            abstracted_query=abstracted_query, metrics=metrics, privacy_score=privacy_score,
            completeness_score=completeness_score, steps=steps, docs=all_retrieved_docs,
            memory=memory, validation=validation,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _retrieve(
        self,
        primary_query: str,
        secondary_query: str,
        top_k: int,
        selected_document: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Deduplicated multi-query retrieval."""
        seen = set()
        results = []
        for q in [primary_query, secondary_query]:
            for doc in self.vector_store.search(q, top_k=top_k, source=selected_document):
                key = (doc.get("source"), doc.get("chunk_id"), doc.get("start_idx"))
                if key not in seen:
                    seen.add(key)
                    results.append(doc)
        return results[: max(top_k, 10)]

    def _build_result(
        self,
        success: bool,
        answer: str,
        mode: str,
        intent: str,
        abstracted_query: str,
        metrics: MetricsTracker,
        privacy_score: float,
        completeness_score: float,
        steps: List[Dict],
        docs: List[Dict],
        memory: Dict,
        validation: Dict,
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "answer": answer,
            "mode": mode,
            "intent": intent,
            "abstracted_query": abstracted_query,
            "latency": metrics.total_time,
            "privacy_score": privacy_score,
            "completeness_score": completeness_score,
            "validation": validation,
            "reasoning_depth": len(steps),
            "reasoning_plan": [s.get("action") + ": " + s.get("target", "") for s in steps],
            "documents_used": list({d.get("source", "Unknown") for d in docs}),
            "retrieved_context": docs[:10],
            "memory": memory,
        }
