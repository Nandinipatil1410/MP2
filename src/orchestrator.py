"""
Hybrid LLM Orchestrator
Intent-driven, metadata-aware coordinator for the RAG pipeline.
Routes queries based on LLM-classified intent — no domain-based branching.
"""
import sys
import time
import re
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

        # If VectorStore auto-loaded a saved index from disk, reflect that here
        self.documents_loaded = len(self.vector_store.documents) > 0
        self.initialized = True

        if self.documents_loaded:
            print(f" Restored {len(self.vector_store.documents)} chunks from persisted vector store")
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
            system_prompt = (
                "You are a Senior Strategic Intelligence Analyst. "
                "Your job is to provide a comprehensive, deeply-reasoned report based ONLY on findings. "
                "NEVER give short answers. Use a structured, professional report format. "
                "Always include: 1. Executive Summary, 2. Key Findings, 3. Detailed Analysis, 4. Conclusion."
            )
            answer = self.cloud_planner.get_completion(system_prompt, query)
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
        """Fast path: retrieve + summarise."""
        # Heuristic: If query mentions specific keywords or sections, it's a targeted summary
        targeted_keywords = ["section", "part", "chapter", "page", "item", "clause", "provision", "paragraph", "article"]
        is_targeted = any(k in query.lower() for k in targeted_keywords) or len(query.split()) > 5

        retrieval_start = time.time()
        if is_targeted:
            # For targeted summaries, the query itself is the best retrieval string
            retrieval_query = query
        else:
            # For global overviews, use broader summary terms
            retrieval_query = "abstract introduction conclusion overview summary key findings"
            
        docs = self._retrieve(
            query, retrieval_query, top_k + 10, selected_document
        )
        metrics.retrieval_time = time.time() - retrieval_start

        context = self.local_executor._prepare_context(docs)

        generation_start = time.time()
        if is_targeted:
            print(f"  → Targeted summary detected ('{query[:40]}...'). Using synthesis path.")
            if mode == "hybrid" and self.cloud_planner.client is not None:
                # Abstract query and seed entity map so the same name gets the
                # SAME placeholder in both the query and the masked context.
                abstracted_query, _ = self.query_abstractor.abstract_query(query)
                seed_map = dict(self.query_abstractor.replacements)
                seed_counters = {}
                for ph in seed_map:
                    m_ph = re.match(r'\[([A-Z]+)_(\d+)\]', ph)
                    if m_ph:
                        label, idx = m_ph.group(1), int(m_ph.group(2))
                        seed_counters[label] = max(seed_counters.get(label, 1), idx + 1)
                masked_context, entity_map, counters = self.local_executor._mask_entities(
                    context, existing_map=seed_map, existing_counters=seed_counters, query=query
                )
                answer = self.cloud_planner.synthesize(
                    abstracted_query, masked_context, intent="summary", expert_mode=True
                )
                if answer:
                    full_map = {**self.query_abstractor.replacements, **entity_map}
                    answer = self.local_executor._restore_entities(answer, full_map)
                    answer = self.local_executor._remove_placeholder_meta_commentary(answer)
                else:
                    answer = self.local_executor.answer_with_context(query, context)
            else:
                answer = self.local_executor.answer_with_context(query, context)
        else:
            print("  → Global document summary detected.")
            # Global overview for generic summary queries
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
        if mode == "hybrid" and self.cloud_planner.client is not None:
            # Hybrid mode: Deep analysis
            # 1. Abstract the query first so placeholders are consistent between
            #    the query text and the masked context that goes to the cloud.
            abstracted_query, _ = self.query_abstractor.abstract_query(query)
            privacy_score = self.query_abstractor.calculate_privacy_score(query)
            if abstracted_query != query:
                print(f"\n\U0001f6e1\ufe0f  PRIVACY LAYER (extraction): Abstracted query")
                print(f"   Original:   \"{query}\"")
                print(f"   Cloud-Ready: \"{abstracted_query}\"")

            # 2. Seed entity masking with the query abstractor's replacements so
            #    the same name gets the SAME placeholder in both query and context.
            seed_map = dict(self.query_abstractor.replacements)  # e.g. {"[PERSON_0]": "Mr Tan Ah Kow"}
            # Build seed counters from existing placeholders (e.g. PERSON already at 1)
            seed_counters = {}
            for ph in seed_map:
                m = re.match(r'\[([A-Z]+)_(\d+)\]', ph)
                if m:
                    label, idx = m.group(1), int(m.group(2))
                    seed_counters[label] = max(seed_counters.get(label, 1), idx + 1)

            masked_context, entity_map, counters = self.local_executor._mask_entities(
                context, existing_map=seed_map, existing_counters=seed_counters, query=query
            )

            answer = self.cloud_planner.synthesize(
                abstracted_query, masked_context, intent="extraction", expert_mode=True
            )
            if answer and answer != "Cloud extraction failed.":
                # Restore using the full combined map (query abstractor + context masking)
                full_map = {**self.query_abstractor.replacements, **entity_map}
                answer = self.local_executor._restore_entities(answer, full_map)
                answer = self.local_executor._remove_placeholder_meta_commentary(answer)
            else:
                print("  [Fallback] Cloud synthesis failed, using high-quality local fallback...")
                answer = self.local_executor.comprehensive_answer(query, context)
        else:
            # Local-only mode: Fast & Concise
            privacy_score = 1.0
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
            "privacy_score": privacy_score if mode == "hybrid" else 1.0,
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
          6. Synthesis (Cloud for Hybrid, Local for Local)
        """
        # 1. Abstraction (PII removal for cloud safety)
        abstracted_query, _ = self.query_abstractor.abstract_query(query)
        privacy_score = self.query_abstractor.calculate_privacy_score(query)
        
        if abstracted_query != query:
            print(f"\n🛡️  PRIVACY LAYER: Abstracted query to protect sensitive data")
            print(f"   Original: \"{query}\"")
            print(f"   Cloud-Ready: \"{abstracted_query}\"")
            print(f"   Privacy Score: {privacy_score * 100:.0f}%\n")

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

        # 5. Execution (Unified Path for both modes to ensure visibility)
        print(f"  → [{mode.upper()} Path] Executing step-wise reasoning loop...")
        from pipeline.executor import execute_steps
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

        # 6. Synthesis
        generation_start = time.time()
        if mode == "hybrid" and self.cloud_planner.client is not None:
            print("  → Synthesizing final answer via Cloud...")
            context = self.local_executor._prepare_context(all_retrieved_docs)
            # Seed entity masking with the query abstractor's replacements so the
            # same person/entity gets the SAME placeholder in both query and context.
            seed_map = dict(self.query_abstractor.replacements)
            seed_counters = {}
            for ph in seed_map:
                m = re.match(r'\[([A-Z]+)_(\d+)\]', ph)
                if m:
                    label, idx = m.group(1), int(m.group(2))
                    seed_counters[label] = max(seed_counters.get(label, 1), idx + 1)
            masked_context, entity_map, counters = self.local_executor._mask_entities(
                context, existing_map=seed_map, existing_counters=seed_counters, query=query
            )
            
            final_answer = self.cloud_planner.synthesize(
                abstracted_query, masked_context, intent=intent, expert_mode=expert_mode
            )
            
            validation = {"valid": True, "score": 1.0, "critique": "Hybrid Cloud Synthesis"}
            if final_answer:
                # Merge query replacements and context entity map for full restoration
                full_map = {**self.query_abstractor.replacements, **entity_map}
                final_answer = self.local_executor._restore_entities(final_answer, full_map)
                final_answer = self.local_executor._remove_placeholder_meta_commentary(final_answer)
            else:
                print("  → Cloud synthesis failed. Falling back to Local synthesis...")
                # Use the local model to synthesize a clean answer from findings
                # We use the abstracted_query so the local model sees consistent placeholders
                local_synthesis_prompt = self.local_executor._create_synthesis_prompt(
                    abstracted_query, [combined_findings], expert_mode=expert_mode
                )
                # Add a system-level hint to the prompt to avoid meta-commentary
                local_synthesis_prompt += "\nIMPORTANT: Treat placeholders like [PERSON_0] as the actual subject. Do not comment on them."
                final_answer = self.local_executor._call_ollama(local_synthesis_prompt, max_tokens=800)
                final_answer = f"⚠️ [Cloud Rate Limit] Synthesized via Local Analyst:\n\n{self.local_executor._clean_response(final_answer)}"
        else:
            print("  → Synthesizing final answer via Local model...")
            # For local synthesis, we use the combined findings from the steps
            final_answer = combined_findings
            validation = {"valid": True, "score": 0.8, "critique": "Local Step-wise Synthesis"}

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
        """Deduplicated multi-query retrieval with hybrid keyword fallback."""
        seen = set()
        results = []
        
        # 1. Semantic Search Pass
        for q in [primary_query, secondary_query]:
            for doc in self.vector_store.search(q, top_k=top_k, source=selected_document):
                key = (doc.get("source"), doc.get("chunk_id"), doc.get("start_idx"))
                if key not in seen:
                    seen.add(key)
                    results.append(doc)
        
        # 2. Keyword Search Pass (Hybrid Fallback)
        # For data extraction, exact term matching is often more reliable than dense embeddings
        keyword_results = self._keyword_search(primary_query, top_k=top_k // 2)
        for doc in keyword_results:
            key = (doc.get("source"), doc.get("chunk_id"), doc.get("start_idx"))
            if key not in seen:
                if selected_document and doc.get("source") != selected_document:
                    continue
                seen.add(key)
                results.insert(0, doc) # Prioritize keyword matches
                
        return results[: max(top_k, 20)]

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Simple keyword matching to catch exact terms that dense embeddings might miss."""
        if not hasattr(self.vector_store, "documents") or not self.vector_store.documents:
            return []
        
        # Extract meaningful keywords (ignore very short words)
        query_words = [w.lower() for w in re.findall(r'\b\w{3,}\b', query)]
        if not query_words:
            return []
            
        scored_docs = []
        for doc in self.vector_store.documents:
            text = doc['text'].lower()
            score = 0
            for word in query_words:
                if word in text:
                    score += 1
            if score > 0:
                # Boost score if multiple unique keywords match
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

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
