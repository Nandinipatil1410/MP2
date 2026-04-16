"""
Local LLM Executor Module
Executes reasoning plans on private documents using local Ollama
"""
import subprocess
import json
import sys
from pathlib import Path
from typing import List, Dict
import requests

# Add root to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import LOCAL_MODEL


class LocalLLMExecutor:
    """Execute reasoning plans using local Ollama LLM"""
    
    def __init__(self, model: str = None):
        self.model = model or LOCAL_MODEL
        self.ollama_url = "http://localhost:11434"
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                print(f" Ollama is running")
                return True
        except:
            print("  Warning: Ollama is not running or not accessible")
            print("   Start Ollama with: ollama serve")
            return False
    
    def execute_plan(self, 
                    reasoning_plan: List[str], 
                    retrieved_docs: List[Dict[str, str]], 
                    original_query: str,
                    expert_mode: bool = False,
                    vector_store=None,
                    selected_document: str = None,
                    cloud_planner=None) -> Dict[str, any]:
        """
        3-Phase Extract → Critique → Synthesize pipeline.
        When cloud_planner is provided (hybrid mode), Phase 3 uses the cloud model
        (LLaMA 70B) to produce a richer final response from sanitized local findings.
        Local-only mode omits cloud_planner, using local synthesis for Phase 3.
        """
        context = self._prepare_context(retrieved_docs)
        print(f"   Executing 3-phase reasoning pipeline locally...")
        print(f"       [Context Load]: {len(context)} chars")
        print(f"       [Context Sample]: {context[:80].replace(chr(10), ' ')}...")

        try:
            # ── PHASE 1: EXTRACTION ─────────────────────────────────────────
            # Force the model to list ONLY verbatim facts from the document.
            # Prevents hallucination by demanding direct quotes/paraphrases.
            print("     -> Phase 1 [Extract]: Pulling key document claims...")
            extract_prompt = self._phase_extract_prompt(original_query, context)
            extracted = self._call_ollama(extract_prompt, max_tokens=400)
            extracted = self._clean_response(extracted)
            print(f"       [Extracted]: {extracted[:120]}...")

            # ── PHASE 2: CRITIQUE ────────────────────────────────────────────
            # Force ONE specific strength and ONE specific weakness.
            # Small models can handle binary analysis but not open-ended critique.
            if expert_mode:
                print("     -> Phase 2 [Critique]: Identifying strengths & weaknesses...")
                critique_prompt = self._phase_critique_prompt(original_query, context, extracted)
                critique = self._call_ollama(critique_prompt, max_tokens=350)
                critique = self._clean_response(critique)
                print(f"       [Critique]: {critique[:120]}...")
                combined_findings = f"EXTRACTED CLAIMS:\n{extracted}\n\nCRITICAL EVALUATION:\n{critique}"
            else:
                combined_findings = f"EXTRACTED CLAIMS:\n{extracted}"

            # ── PHASE 3: SYNTHESIS ───────────────────────────────────────────
            # In hybrid mode: try cloud synthesis first (LLaMA 70B via Groq)
            # In local-only mode: always use local synthesis
            print("     -> Phase 3 [Synthesize]: Writing final response...")
            final_answer = None

            if cloud_planner is not None:
                # ─ Pseudonymize: replace PII before sending to cloud ─────────
                masked_findings, entity_map = self._mask_entities(combined_findings)
                if entity_map:
                    print(f"       [Privacy]: Masked {len(entity_map)} entity/entities before cloud call")

                cloud_response = cloud_planner.synthesize(
                    original_query, masked_findings, expert_mode=expert_mode
                )

                if cloud_response:
                    # ─ Restore real values back into cloud's write-up ───────
                    final_answer = self._restore_entities(cloud_response, entity_map)

            if not final_answer:
                # Fallback: local synthesis (no masking needed — never leaves machine)
                synthesis_prompt = self._create_synthesis_prompt(
                    original_query, [combined_findings], expert_mode=expert_mode
                )
                final_answer = self._call_ollama(synthesis_prompt, max_tokens=600)

            final_answer = self._clean_response(final_answer)

            return {
                'success': True,
                'answer': final_answer,
                'model': self.model,
                'docs_used': len(retrieved_docs),
                'steps_executed': 3
            }

        except Exception as e:
            print(f"Error in pipeline execution: {e}")
            import traceback; traceback.print_exc()
            return {
                'success': False,
                'answer': self._fallback_response(original_query, retrieved_docs)
            }
    

    def _mask_entities(self, text: str):
        """
        Selectively mask PII from findings before sending to cloud.
        Only masks structured PII (emails, URLs, phones, specific numbers)
        to avoid degrading the cloud model's reasoning quality.
        Conceptual/semantic terms (diseases, methods, concepts) are left intact.

        Returns: (masked_text, entity_map)
        entity_map maps placeholder -> original value for restoration.
        """
        import re
        entity_map = {}
        counters = {}

        def replace(pattern, label, text):
            """Find all matches and replace with typed placeholders."""
            def _sub(m):
                val = m.group(0)
                # Reuse same placeholder if we've seen this exact value before
                for ph, orig in entity_map.items():
                    if orig == val:
                        return ph
                n = counters.get(label, 1)
                counters[label] = n + 1
                placeholder = f"[{label}_{n}]"
                entity_map[placeholder] = val
                return placeholder
            return re.sub(pattern, _sub, text)

        # ─ Emails ────────────────────────────────────────────────────
        text = replace(
            r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
            'EMAIL', text
        )
        # ─ URLs ────────────────────────────────────────────────────
        text = replace(
            r'https?://[^\s<>"{}|\\^`]+',
            'URL', text
        )
        # ─ Phone numbers (common formats) ─────────────────────────────
        text = replace(
            r'(\+?\d[\d\s\-().]{7,}\d)',
            'PHONE', text
        )
        # ─ Standalone numeric data points (stats, percentages, years) ──────
        # Only mask if it looks like a specific value (not a generic count like "3")
        text = replace(
            r'\b\d{1,3}\.\d+%?\b',  # e.g. 94.2%, 0.87, 12.5
            'NUM', text
        )
        text = replace(
            r'\b(19|20)\d{2}\b',    # years like 2021, 1998
            'YEAR', text
        )
        # ─ arXiv / DOI / paper IDs ───────────────────────────────────
        text = replace(
            r'arXiv:\d{4}\.\d{4,5}',
            'PAPERID', text
        )

        return text, entity_map

    def _restore_entities(self, text: str, entity_map: dict) -> str:
        """
        Replace placeholders in cloud-generated text with original PII values.
        Simple string substitution; handles cases where cloud slightly reformats
        the placeholder (e.g. lowercases it) by also trying lowercase variants.
        """
        for placeholder, original in entity_map.items():
            # Exact match
            text = text.replace(placeholder, original)
            # Also try if cloud changed case (e.g. [email_1])
            text = text.replace(placeholder.lower(), original)
        return text

    def _clean_response(self, text: str, step_num: int = 0) -> str:
        """Strip looping patterns, step prefixes and filler from model output."""
        import re
        
        # Strip common "Step N:" prefixes
        if step_num > 0:
            prefixes = [f"Step {step_num}:", f"Step {step_num} ", f"Step {step_num}"]
            for p in prefixes:
                if text.lower().startswith(p.lower()):
                    text = text[len(p):].strip()
        
        # Cut off anything from the first "In summary" or "In conclusion" (case-insensitive)
        for marker in ["\nIn summary", "\nin summary", "\nIn conclusion", "\nin conclusion"]:
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx].strip()
        
        # Remove duplicate paragraphs
        paragraphs = text.split('\n\n')
        seen_paras = []
        unique_paragraphs = []
        for p in paragraphs:
            normalized = re.sub(r'\s+', ' ', p).strip()
            if normalized and normalized not in seen_paras:
                seen_paras.append(normalized)
                unique_paragraphs.append(p)
        text = '\n\n'.join(unique_paragraphs)

        # Also strip duplicate sentences within paragraphs
        sentences = re.split(r'(?<=[.!?])\s+', text)
        seen_sents = []
        unique_sents = []
        for s in sentences:
            norm = re.sub(r'\s+', ' ', s).strip()
            if norm and norm not in seen_sents:
                seen_sents.append(norm)
                unique_sents.append(s)
        text = ' '.join(unique_sents)

        return text.strip()
    
    def _prepare_context(self, documents: List[Dict[str, str]]) -> str:
        """Prepare context from retrieved documents (with safety limit)"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        char_count = 0
        MAX_CHARS = 15000 # ~4k tokens, safe for most local LLMs
        
        for i, doc in enumerate(documents, 1):
            text = doc['text']
            if char_count + len(text) > MAX_CHARS:
                text = text[:MAX_CHARS - char_count] + "..."
            
            context_parts.append(f"[Document {i}]: {text}\n")
            char_count += len(text)
            if char_count >= MAX_CHARS:
                break
        
        return "\n".join(context_parts)
    
    def _create_step_prompt(self, 
                           step_intent: str,
                           step_num: int,
                           total_steps: int,
                           query: str, 
                           context: str,
                           previous_findings: List[str],
                           expert_mode: bool = False) -> str:
        """Create a precision instruction for a single reasoning hop"""
        history = "\n".join([f"- {f[:500]}" for f in previous_findings]) if previous_findings else "None."
        
        role = "Senior Expert Analyst" if expert_mode else "Lead Assistant"
        
        # Step 1 Resilience: If this is the start and it's an intro step, help the model anchor.
        is_intro_step = any(kw in step_intent.lower() for kw in ["introduction", "contextual", "framework", "overview"])
        task = step_intent
        if step_num == 1 and is_intro_step:
            task = f"{step_intent} (Note: Use current evidence to establish context.)"
            
        if expert_mode and step_num > 1:
            # Only add specific analytical mandates if the step looks evaluative
            if any(kw in step_intent.lower() for kw in ["evaluate", "assess", "rigor", "technical", "limitations", "comparison"]):
                task = f"{step_intent} (Be technical. Identify one specific strength and one weakness.)"

        return f"""### ROLE: {role}
### TASK: {task}
### EVIDENCE:
{context}

### PREVIOUS FINDINGS:
{history}

### INSTRUCTIONS:
1. Provide a direct {role} response based ONLY on the EVIDENCE above.
2. DO NOT mention previous steps, "Step X", or meta-commentary like "I will now...".
3. Cite [Document X] where appropriate.
4. If the evidence is lacking, say "Insufficient data".

### {role.upper()} ANALYSIS:"""

    def _phase_extract_prompt(self, query: str, context: str) -> str:
        """Phase 1: Extract key topics and claims from compressed document context."""
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
        """Phase 2: Binary strength/weakness evaluation against standard academic criteria."""
        return f"""### TASK: Critically evaluate the document for strengths and limitations.
### QUESTION: {query}
### DOCUMENT THEMES (from Phase 1):
{extracted}
### DOCUMENT EXCERPT:
{context[:4000]}

### INSTRUCTIONS:
- STRENGTH: Identify ONE clear academic strength (e.g. breadth of coverage, novelty of framework, clarity of classification).
- WEAKNESS: Identify ONE clear academic weakness or gap (e.g. lack of empirical data, no comparison baseline, scope too narrow).
- Be specific but concise (2-3 sentences each). Reference the document's content.
- Do NOT say "the document does not mention" unless it is truly absent.

### STRENGTH:"""

    def _create_synthesis_prompt(self, query: str, findings: List[str], expert_mode: bool = False) -> str:
        """Create final prompt to synthesize all findings into a professional answer"""
        findings_text = "\n\n".join(findings)
        role = "Master Evaluator" if expert_mode else "Lead Assistant"
        
        if expert_mode:
            return f"""### ROLE: Senior Expert Analyst
### TASK: Write a professional expert review answering: {query}
### RESEARCH FINDINGS:
{findings_text}

### INSTRUCTIONS:
1. Write 3-4 paragraphs. Do NOT use bullet points.
2. Paragraph 1: What the document covers and its core contribution.
3. Paragraph 2: Strengths — what the document does well with specific reference.
4. Paragraph 3: Weaknesses or gaps — what is missing or could be improved.
5. Paragraph 4 (optional): Overall assessment and recommendation.
6. STOP after the final paragraph. Do not add summaries or conclusions.

### EXPERT REVIEW:"""
        else:
            return f"""### ROLE: {role}
### TASK: Provide a clear answer based on findings.
### QUESTION: {query}
### FINDINGS:
{findings_text}

### INSTRUCTIONS:
1. Provide a direct answer. No filler.
2. Cite documents simply as [Source Name] if known.

### FINAL ANSWER:"""
    
    def _call_ollama(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call Ollama API for generation"""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens,
                "repeat_penalty": 1.3,  # Increased from 1.1 — prevents paragraph-level looping
                "top_k": 20,
                "top_p": 0.6,
                "stop": ["###", "ROLE:", "TASK:", "EVIDENCE:", "In summary:", "In conclusion:", "The review is divided"]
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '').strip()
    
    def _fallback_response(self, query: str, documents: List[Dict[str, str]]) -> str:
        """Generate simple fallback response if Ollama fails"""
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Simple extractive response
        relevant_text = " ".join([doc['text'][:200] for doc in documents[:2]])
        return f"Based on the available documents: {relevant_text}..."
    
    def generate_simple(self, prompt: str) -> str:
        """Simple generation without plan execution"""
        try:
            return self._call_ollama(prompt)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def check_model_available(self) -> bool:
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(m['name'].startswith(self.model) for m in models)
            return False
        except:
            return False


if __name__ == "__main__":
    # Test the executor
    executor = LocalLLMExecutor()
    
    if executor.check_model_available():
        print(f" Model '{executor.model}' is available")
    else:
        print(f"  Model '{executor.model}' not found. Pull it with: ollama pull {executor.model}")
    
    # Test simple generation
    test_response = executor.generate_simple("Say hello in one sentence.")
    print(f"\nTest response: {test_response}")
