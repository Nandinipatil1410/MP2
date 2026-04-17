"""
Cloud Reasoning Planner Module
Uses cloud LLM (Groq) to generate structured reasoning plans and synthesize answers.
Domain-agnostic: no financial/medical/legal/research branching.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import CLOUD_MODEL


class CloudReasoningPlanner:
    """Generate reasoning plans and synthesize answers using the cloud LLM (Groq)."""

    def __init__(self, api_key: str = None, model: str = None):
        env_key = os.getenv("GROQ_API_KEY")
        self.api_key = api_key or env_key
        if self.api_key and self.api_key.strip().lower() in {
            "your_groq_api_key_here", "your_api_key_here", "changeme"
        }:
            self.api_key = None
        self.model = model or CLOUD_MODEL

        if not self.api_key:
            print("  Warning: No Groq API key found. Cloud features will use fallback mode.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def generate_structured_plan(
        self,
        query: str,
        metadata: Dict[str, Any],
        expert_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a structured step-by-step plan using document metadata.

        Args:
            query: The (abstracted) user query.
            metadata: Structured metadata dict from metadata_extractor
                      (document_type, sections, key_fields, data_types).
            expert_mode: Request a more detailed plan.

        Returns:
            {"steps": [{"id": int, "action": str, "target": str}, ...]}
        """
        if not self.client:
            return self._fallback_plan(query)

        num_steps = "5-7" if expert_mode else "3-5"

        system_prompt = (
            "You are a Reasoning Architect. "
            "Design a step-by-step investigation plan based on the query and document metadata. "
            "Each step must specify a concrete action and a specific target to investigate. "
            "Actions must be one of: extract, compare, infer, summarize, validate, calculate, evaluate, synthesize_arguments, identify_limitations."
        )

        user_prompt = f"""Query: "{query}"

Document Metadata:
- Type: {metadata.get("document_type", "unknown")}
- Sections: {metadata.get("sections", [])}
- Key Fields: {metadata.get("key_fields", [])}
- Data Types: {metadata.get("data_types", ["text"])}

Design {num_steps} concrete steps to answer the query.

Return ONLY a JSON object in this exact format:
{{
  "steps": [
    {{"id": 1, "action": "extract", "target": "specific thing to extract"}},
    {{"id": 2, "action": "compare", "target": "what to compare"}},
    {{"id": 3, "action": "infer", "target": "what to infer from evidence"}}
  ]
}}

Rules:
- No vague steps like "analyze the document"
- Each target must be specific and directly relate to the query
- Actions must be from: extract, compare, infer, summarize, validate, calculate, evaluate, synthesize_arguments, identify_limitations"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(response.choices[0].message.content)
            steps = data.get("steps", [])
            if not steps:
                return self._fallback_plan(query)
            print(f"  [Planner] Generated {len(steps)}-step plan for: '{query[:60]}'")
            return {"steps": steps}
        except Exception as e:
            print(f"  [Planner] Plan generation failed ({e}), using fallback.")
            return self._fallback_plan(query)

    def generate_agentic_plan(
        self,
        query: str,
        expert_mode: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Backward-compatible plan generation (used by orchestrator legacy paths).
        Returns list of step dicts with 'intent' and 'search_query' keys.
        """
        if not self.client:
            return self._fallback_agentic_steps(query, expert_mode)

        num_steps = "4-7" if expert_mode else "3-5"
        meta_block = ""
        if metadata:
            meta_block = f"\nDocument context: type={metadata.get('document_type','unknown')}, key fields={metadata.get('key_fields',[])}"

        system_prompt = (
            "You are a Lead Reasoning Architect. Design a step-by-step investigation plan. "
            "For each step provide a targeted search query for a vector database."
        )
        user_prompt = f"""Design a {num_steps}-step reasoning plan for: "{query}"{meta_block}

For each step provide:
- intent: What we are trying to find
- search_query: A specific keyword-rich query for the vector DB

Return as a JSON object with a "steps" key containing a list:
{{
  "steps": [
    {{"intent": "...", "search_query": "..."}},
    ...
  ]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            data = json.loads(response.choices[0].message.content)
            steps = data.get("steps", [])
            if isinstance(steps, list) and steps:
                return steps
            return self._fallback_agentic_steps(query, expert_mode)
        except Exception as e:
            print(f"  [Planner] Agentic plan failed ({e}), using fallback.")
            return self._fallback_agentic_steps(query, expert_mode)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        local_findings: str,
        intent: str = "reasoning",
        expert_mode: bool = False,
    ) -> Optional[str]:
        """
        Synthesize a grounded final answer from local findings using the cloud model.
        Domain-agnostic: no financial/medical/legal/research branching.

        Args:
            query: The original user query.
            local_findings: Structured findings extracted locally.
            expert_mode: Request a more detailed response.

        Returns:
            Answer string, or None if cloud is unavailable.
        """
        if not self.client:
            return None

        detail_level = "detailed, well-structured" if expert_mode else "concise and direct"

        system_prompt = (
            "You are a precise, evidence-based analyst. "
            "Your only job is to synthesize a grounded answer from the provided findings. "
            "You NEVER invent information. If a detail is not in the findings, you omit it."
        )

        if intent == "reasoning" and ("review" in query.lower() or "evaluate" in query.lower() or "critique" in query.lower()):
            structure_rubric = (
                "\n\nStructure your answer with the following sections if applicable:\n"
                "- Overview & Methodology\n"
                "- Key Arguments & Findings\n"
                "- Limitations or Weaknesses\n"
                "- Implications/Conclusion"
            )
        elif intent == "comparison":
            structure_rubric = "\n\nStructure your answer as a systematic comparison, contrasting the entities explicitly."
        else:
            structure_rubric = ""

        user_prompt = f"""Query: "{query}"

--- FINDINGS (only valid source of truth) ---
{local_findings}
--- END FINDINGS ---

Write a {detail_level} answer that:
1. Directly addresses the query
2. Uses ONLY information present in the findings
3. Avoids speculation, filler phrases, or generic statements
4. If a finding says "DATA_ABSENT", acknowledge it as not found{structure_rubric}

If the findings contain insufficient data to answer the query, respond:
"Insufficient data in document to answer: [specific missing information]"

ANSWER:"""

        try:
            print("  [Planner] Cloud synthesis in progress...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Planner] Cloud synthesis failed ({e}).")
            return None

    def synthesize_repair(
        self,
        query: str,
        local_findings: str,
        draft_answer: str,
        critique: Optional[str] = None,
        expert_mode: bool = False,
    ) -> Optional[str]:
        """
        Rewrite a draft answer to be strictly grounded in findings.
        Used when the validator flags potential hallucination.
        """
        if not self.client:
            return None

        critique_block = f"\nVALIDATION CRITIQUE:\n{critique}\n" if critique else ""

        system_prompt = (
            "You are a Grounded Rewriter. "
            "Rewrite the draft using ONLY the provided findings. "
            "If a detail is not explicitly supported, omit it or state it is not in the findings."
        )

        user_prompt = f"""Rewrite the draft answer so it is strictly grounded in the findings.

QUERY: {query}

FINDINGS (only source of truth):
{local_findings}

DRAFT ANSWER (may contain unsupported claims):
{draft_answer}
{critique_block}
STRICT RULES:
1. Use ONLY the findings as evidence.
2. If something is not supported, write "Not stated in the findings."
3. Do not introduce new facts, numbers, entities, or dates.
4. Be concise and direct. No filler.

REPAIRED ANSWER:"""

        try:
            print("  [Planner] Cloud repair in progress...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=900,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Planner] Cloud repair failed ({e}).")
            return None

    # ------------------------------------------------------------------
    # Validation (kept here for backward compat; also lives in pipeline/validator.py)
    # ------------------------------------------------------------------

    def validate_response(
        self, query: str, answer: str, findings: str
    ) -> Dict[str, Any]:
        """Verify answer grounding against findings using the cloud model."""
        if not self.client:
            return {"valid": True, "score": 1.0, "critique": "Validation skipped: No cloud client."}

        system_prompt = (
            "You are a Grounding Auditor. "
            "Flag hallucination ONLY if the answer makes specific numerical or factual claims "
            "that directly contradict or are completely absent from the findings. "
            "Allow natural-language summarisation and logical deductions."
        )

        user_prompt = f"""Query: {query}
Answer: {answer}
Findings: {findings}

Return JSON:
{{
  "valid": boolean,
  "score": float (0-1),
  "hallucination_detected": boolean,
  "critique": "brief analysis"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"valid": True, "score": 0.5, "critique": f"Validation error: {e}"}

    # ------------------------------------------------------------------
    # Direct completion (used by cloud-only benchmark mode)
    # ------------------------------------------------------------------

    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Get a direct completion from the cloud LLM."""
        if not self.client:
            return "Cloud response unavailable: no Groq API key configured."
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to cloud: {e}"

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    def _fallback_plan(self, query: str) -> Dict[str, Any]:
        """Minimal structured plan used when cloud is unavailable."""
        return {
            "steps": [
                {"id": 1, "action": "extract", "target": f"key information about: {query}"},
                {"id": 2, "action": "infer", "target": f"conclusions from extracted data for: {query}"},
            ]
        }

    def _fallback_agentic_steps(self, query: str, expert_mode: bool) -> List[Dict[str, Any]]:
        if expert_mode:
            return [
                {"intent": "Retrieve primary evidence", "search_query": query},
                {"intent": "Retrieve supporting context and caveats", "search_query": query},
                {"intent": "Retrieve any clarifying notes or disclosures", "search_query": query},
            ]
        return [
            {"intent": "Search and retrieve specific evidence", "search_query": query},
            {"intent": "Analyse findings and extract core signals", "search_query": query},
        ]
