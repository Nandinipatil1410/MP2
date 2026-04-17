"""
Local LLM Executor Module
Executes reasoning using the local Ollama model.
Domain-agnostic: no financial/medical/legal/research prompt branching.
"""
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import LOCAL_MODEL


class LocalLLMExecutor:
    """Execute reasoning using local Ollama LLM."""

    def __init__(self, model: str = None):
        self.model = model or LOCAL_MODEL
        self.ollama_url = "http://localhost:11434"
        self._check_ollama()

    # ------------------------------------------------------------------
    # Ollama connectivity
    # ------------------------------------------------------------------

    def _check_ollama(self):
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print(" Ollama is running")
                return True
        except Exception:
            pass
        print("  Warning: Ollama is not running or not accessible")
        print("   Start Ollama with: ollama serve")
        return False

    def check_model_available(self) -> bool:
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m["name"].startswith(self.model) for m in models)
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        reasoning_plan: List[str],
        retrieved_docs: List[Dict],
        original_query: str,
        expert_mode: bool = False,
        vector_store=None,
        selected_document: str = None,
        cloud_planner=None,
    ) -> Dict:
        """
        3-Phase Extract → Critique → Synthesize pipeline.
        When cloud_planner is provided (hybrid mode), Phase 3 uses the cloud model.
        """
        context = self._prepare_context(retrieved_docs)
        print(f"  Executing 3-phase reasoning pipeline locally...")
        print(f"    [Context Load]: {len(context)} chars")

        try:
            # Phase 1: Extract
            print("  → Phase 1 [Extract]: Pulling key document claims...")
            extracted = self._call_ollama(
                self._phase_extract_prompt(original_query, context), max_tokens=400
            )
            extracted = self._clean_response(extracted)

            # Phase 2: Critique (expert mode only)
            if expert_mode:
                print("  → Phase 2 [Critique]: Identifying strengths & gaps...")
                critique = self._call_ollama(
                    self._phase_critique_prompt(original_query, context, extracted), max_tokens=350
                )
                critique = self._clean_response(critique)
                combined_findings = f"EXTRACTED CLAIMS:\n{extracted}\n\nCRITICAL EVALUATION:\n{critique}"
            else:
                combined_findings = f"EXTRACTED CLAIMS:\n{extracted}"

            # Phase 3: Synthesis
            print("  → Phase 3 [Synthesize]: Writing final response...")
            final_answer = None

            if cloud_planner is not None:
                masked_findings, entity_map, counters = self._mask_entities(combined_findings)
                if entity_map:
                    print(f"    [Privacy]: Masked {len(entity_map)} entity/entities before cloud call")
                cloud_response = cloud_planner.synthesize(
                    original_query, masked_findings, expert_mode=expert_mode
                )
                if cloud_response:
                    final_answer = self._restore_entities(cloud_response, entity_map)

            if not final_answer:
                synthesis_prompt = self._create_synthesis_prompt(
                    original_query, [combined_findings], expert_mode=expert_mode
                )
                final_answer = self._call_ollama(synthesis_prompt, max_tokens=600)

            final_answer = self._clean_response(final_answer)
            return {
                "success": True,
                "answer": final_answer,
                "model": self.model,
                "docs_used": len(retrieved_docs),
                "steps_executed": 3,
            }

        except Exception as e:
            print(f"Error in pipeline execution: {e}")
            import traceback; traceback.print_exc()
            return {
                "success": False,
                "answer": self._fallback_response(original_query, retrieved_docs),
            }

    def create_overview(
        self,
        context: str,
        selected_document: Optional[str] = None,
    ) -> str:
        """High-level document overview for summary queries."""
        doc_line = f"Selected document: {selected_document}\n" if selected_document else ""
        prompt = f"""### SYSTEM:
You are a document classifier and summariser. Answer: "What is this document about?"
Stay grounded in the provided context. Do not do detailed analysis unless asked.

### CONTEXT (evidence):
{context if context.strip() else "[No relevant context retrieved]"}

### INSTRUCTIONS:
{doc_line}- Output 4-8 concise bullet points.
- Include: document type, main topics, intended audience, and major sections.
- Use ONLY the context. If something is unclear, write "Not stated in the context."

### OVERVIEW:"""
        try:
            response = self._call_ollama(prompt, max_tokens=450)
            return self._clean_response(response)
        except Exception:
            return "Error: Local model unavailable. Please ensure Ollama is installed and running."

    def answer_with_context(self, query: str, context: str) -> str:
        """Direct grounded answer for extraction-style questions (retrieve → answer)."""
        prompt = self._create_grounded_answer_prompt(query=query, context=context)
        try:
            response = self._call_ollama(prompt, max_tokens=500)
            return self._clean_response(response)
        except Exception:
            return "Error: Local model unavailable. Please ensure Ollama is installed and running."

    def execute_reasoning_step(
        self,
        original_query: str,
        step_intent: str,
        context: str,
        previous_memory: str = "None",
    ) -> str:
        """Execute a single targeted reasoning step with memory of previous steps."""
        prompt = f"""### ROLE: Senior Intelligence Analyst
### TASK: {step_intent}
### PRIMARY QUERY: {original_query}

### CONTEXT (Private Evidence):
{context}

### INTERMEDIATE MEMORY:
{previous_memory}

### INSTRUCTIONS:
1. Extract ONLY information directly relevant to the TASK and PRIMARY QUERY.
2. If the data is missing, say "DATA_ABSENT: [missing item]".
3. DO NOT hallucinate. Ground every number in the CONTEXT.
4. Distinguish between 'Confirmed Signal' and 'Supporting Context'.

### ANALYSIS:"""
        try:
            response = self._call_ollama(prompt, max_tokens=500)
            return self._clean_response(response)
        except Exception as e:
            return f"Error: Local model unavailable. ({e})"

    def filter_relevance(self, query: str, raw_findings: str) -> Dict[str, List[str]]:
        """Signal vs Noise filter — classifies extracted information."""
        prompt = f"""### TASK: Filter the following findings for relevance to the query.
### QUERY: {query}
### FINDINGS:
{raw_findings}

### INSTRUCTIONS:
Classify each point into one of:
- [SIGNAL]: Direct answer or core evidence.
- [SUPPORT]: Useful context but not the primary answer.
- [NOISE]: Irrelevant info to be discarded.

### CLASSIFIED FINDINGS:"""

        response = self._call_ollama(prompt, max_tokens=600)
        lines = response.split("\n")
        result: Dict[str, List[str]] = {"signal": [], "support": [], "noise": []}
        current_cat = "support"

        for line in lines:
            line = line.strip()
            if "[SIGNAL]" in line.upper():
                current_cat = "signal"
            elif "[SUPPORT]" in line.upper():
                current_cat = "support"
            elif "[NOISE]" in line.upper():
                current_cat = "noise"
            elif line and not any(t in line.upper() for t in ["[SIGNAL]", "[SUPPORT]", "[NOISE]"]):
                result[current_cat].append(line.lstrip("- *"))

        return result

    def generate_simple(self, prompt: str) -> str:
        """Simple generation without plan execution."""
        try:
            return self._call_ollama(prompt)
        except Exception as e:
            return f"Error: {str(e)}"

    # ------------------------------------------------------------------
    # Prompt builders — domain-agnostic
    # ------------------------------------------------------------------

    def _phase_extract_prompt(self, query: str, context: str) -> str:
        """Phase 1: Extract key factual claims from document context."""
        return f"""### TASK: Read the document excerpt and answer the question below.
### QUESTION: {query}
### DOCUMENT:
{context}

### INSTRUCTIONS:
1. Provide a direct, factual answer based ONLY on the evidence provided above.
2. Use bullet points for key findings.
3. If the answer is not in the document, say "Information not found in context."
4. DO NOT invent information, statistics, or dates.
5. Ground every point in a direct reference to [Document X].

### FACTUAL FINDINGS:"""

    def _phase_critique_prompt(self, query: str, context: str, extracted: str) -> str:
        """Phase 2: Identify one strength and one gap in the evidence."""
        return f"""### TASK: Critically evaluate the document evidence for strengths and gaps.
### QUESTION: {query}
### EXTRACTED FINDINGS (from Phase 1):
{extracted}
### DOCUMENT EXCERPT:
{context[:4000]}

### INSTRUCTIONS:
- STRENGTH: Identify ONE clear strength (e.g. data granularity, clarity, depth).
- GAP: Identify ONE clear gap or limitation (e.g. missing data points, no comparison baseline).
- Be specific but concise (2-3 sentences each). Reference the document content.

### STRENGTH:"""

    def _create_grounded_answer_prompt(self, query: str, context: str) -> str:
        """Prompt for direct extraction-style questions."""
        return f"""### SYSTEM:
You answer questions using ONLY the provided context.
If the answer is not present, say: "Not stated in the context."

### CONTEXT:
{context if context.strip() else "[No relevant context retrieved]"}

### QUESTION:
{query}

### INSTRUCTIONS:
- Quote or paraphrase only what the context supports.
- Do not invent numbers, entities, dates, or causal explanations.
- Keep the answer concise and direct.

### FINAL ANSWER:"""

    def _create_synthesis_prompt(
        self, query: str, findings: List[str], expert_mode: bool = False
    ) -> str:
        """Final synthesis prompt — domain-agnostic."""
        findings_text = "\n\n".join(findings)
        role = "Senior Expert Analyst" if expert_mode else "Lead Assistant"
        detail = "thorough and structured" if expert_mode else "concise and direct"

        return f"""### ROLE: {role}
### TASK: Provide a {detail} answer based on findings.
### QUESTION: {query}
### FINDINGS:
{findings_text}

### INSTRUCTIONS:
1. Use ONLY the FINDINGS. If a detail is missing, say "Not stated in the findings."
2. Directly answer the query. No filler, no generic statements.
3. Ground every claim in the findings.

### FINAL ANSWER:"""

    # ------------------------------------------------------------------
    # Privacy: PII masking / restoration
    # ------------------------------------------------------------------

    def _mask_entities(self, text: str, existing_map=None, existing_counters=None):
        """
        Selectively mask PII from findings before sending to cloud.
        Masks emails, URLs, phones, specific numbers, years, proper names, paper IDs.
        Returns: (masked_text, entity_map, counters)
        """
        entity_map = existing_map if existing_map is not None else {}
        counters = existing_counters if existing_counters is not None else {}

        def replace(pattern, label, text):
            def _sub(m):
                val = m.group(0)
                for ph, orig in entity_map.items():
                    if orig == val:
                        return ph
                n = counters.get(label, 1)
                counters[label] = n + 1
                placeholder = f"[{label}_{n}]"
                entity_map[placeholder] = val
                return placeholder
            return re.sub(pattern, _sub, text)

        text = replace(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', "EMAIL", text)
        text = replace(r'https?://[^\s<>"{}|\\^`]+', "URL", text)
        text = replace(r'(\+?\d[\d\s\-().]{7,}\d)', "PHONE", text)
        text = replace(r'\b\d{1,3}\.\d+%?\b', "NUM", text)
        text = replace(r'\b(19|20)\d{2}\b', "YEAR", text)
        text = replace(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', "NAME", text)
        text = replace(r'arXiv:\d{4}\.\d{4,5}', "PAPERID", text)

        return text, entity_map, counters

    def _restore_entities(self, text: str, entity_map: dict) -> str:
        """Replace placeholders in cloud-generated text with original PII values."""
        for placeholder, original in entity_map.items():
            text = text.replace(placeholder, original)
            text = text.replace(placeholder.lower(), original)
        return text

    # ------------------------------------------------------------------
    # Context preparation
    # ------------------------------------------------------------------

    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context from retrieved documents with safety character limit."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        char_count = 0
        MAX_CHARS = 15000

        for i, doc in enumerate(documents, 1):
            text = doc["text"]
            if char_count + len(text) > MAX_CHARS:
                text = text[: MAX_CHARS - char_count] + "..."
            context_parts.append(f"[Document {i}]: {text}\n")
            char_count += len(text)
            if char_count >= MAX_CHARS:
                break

        return "\n".join(context_parts)

    # ------------------------------------------------------------------
    # Response cleaning
    # ------------------------------------------------------------------

    def _clean_response(self, text: str, step_num: int = 0) -> str:
        """Strip looping patterns, step prefixes, and summary filler from model output."""
        if step_num > 0:
            prefixes = [f"Step {step_num}:", f"Step {step_num} ", f"Step {step_num}"]
            for p in prefixes:
                if text.lower().startswith(p.lower()):
                    text = text[len(p):].strip()

        for marker in ["\nIn summary", "\nin summary", "\nIn conclusion", "\nin conclusion"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()

        # Deduplicate paragraphs
        paragraphs = text.split("\n\n")
        seen_paras, unique_paragraphs = [], []
        for p in paragraphs:
            normalized = re.sub(r"\s+", " ", p).strip()
            if normalized and normalized not in seen_paras:
                seen_paras.append(normalized)
                unique_paragraphs.append(p)
        text = "\n\n".join(unique_paragraphs)

        # Deduplicate sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        seen_sents, unique_sents = [], []
        for s in sentences:
            norm = re.sub(r"\s+", " ", s).strip()
            if norm and norm not in seen_sents:
                seen_sents.append(norm)
                unique_sents.append(s)
        text = " ".join(unique_sents)

        return text.strip()

    # ------------------------------------------------------------------
    # Ollama API
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call Ollama API for generation."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                "repeat_penalty": 1.3,
                "top_k": 20,
                "top_p": 0.6,
                "stop": [
                    "###", "ROLE:", "TASK:", "EVIDENCE:",
                    "In summary:", "In conclusion:", "The review is divided",
                ],
            },
        }
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "").strip()

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_response(self, query: str, documents: List[Dict]) -> str:
        if not documents:
            return "Insufficient data in document to answer this query."
        relevant_text = " ".join([doc["text"][:200] for doc in documents[:2]])
        return f"Based on available documents: {relevant_text}..."
