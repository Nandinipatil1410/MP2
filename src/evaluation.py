"""
Evaluation Module: LLMJudge
Provides automated evaluation and comparison of LLM answers using a local judge model.
"""
import json
import re
import requests
from typing import Dict, List, Any, Optional
from config.settings import LOCAL_MODEL

class LLMJudge:
    """
    LLM-as-a-Judge: Uses a local LLM to evaluate query answers based on 
    accuracy, completeness, relevance, and groundedness.
    """

    def __init__(self, model: str = None):
        self.judge_model = model or LOCAL_MODEL
        self.ollama_url = "http://localhost:11434"

    def evaluate_answer(self, query: str, answer: str, context: List[Dict]) -> Dict[str, Any]:
        """Evaluate a single answer against provided context."""
        context_text = "\n".join([f"[Doc {i+1}]: {d.get('text', '')}" for i, d in enumerate(context[:10])])
        
        prompt = f"""### SYSTEM:
You are an impartial judge evaluating an LLM response for a RAG system.
Evaluate the ANSWER based on the provided QUERY and CONTEXT (evidence).

### QUERY:
{query}

### CONTEXT:
{context_text}

### ANSWER:
{answer}

### INSTRUCTIONS:
Evaluate the answer on 5 criteria (0-10 scale):
1. accuracy: Is it factually correct according to the context?
2. completeness: Does it answer all parts of the query?
3. relevance: Is it focused on the query without filler?
4. coherence: Is the response logical and well-structured?
5. groundedness: Does it strictly use the context (no hallucinations)?

Output ONLY a JSON object with this structure:
{{
    "accuracy": {{"score": 0-10, "reason": "..."}},
    "completeness": {{"score": 0-10, "reason": "..."}},
    "relevance": {{"score": 0-10, "reason": "..."}},
    "coherence": {{"score": 0-10, "reason": "..."}},
    "groundedness": {{"score": 0-10, "reason": "..."}},
    "overall_critique": "..."
}}
"""
        try:
            response = self._call_ollama(prompt)
            return self._parse_json(response)
        except Exception as e:
            return {
                "error": str(e),
                "accuracy": {"score": 0, "reason": "Eval failed"},
                "completeness": {"score": 0, "reason": "Eval failed"},
                "relevance": {"score": 0, "reason": "Eval failed"},
                "coherence": {"score": 0, "reason": "Eval failed"},
                "groundedness": {"score": 0, "reason": "Eval failed"}
            }

    def compare_answers(self, query: str, answer_a: str, answer_b: str, context: List[Dict]) -> Dict[str, Any]:
        """Side-by-side comparison of two answers."""
        context_text = "\n".join([f"[Doc {i+1}]: {d.get('text', '')}" for i, d in enumerate(context[:10])])

        prompt = f"""### SYSTEM:
You are a senior analyst comparing two LLM responses (A and B) for the same query.
Identify which response is better based on the provided context.

### QUERY:
{query}

### CONTEXT:
{context_text}

### ANSWER A (Local Only):
{answer_a}

### ANSWER B (Hybrid):
{answer_b}

### INSTRUCTIONS:
Determine which answer is superior. Consider accuracy, depth, and reasoning.
Output ONLY a JSON object with this structure:
{{
    "winner": "A" | "B" | "Tie",
    "preference_strength": "Weak" | "Moderate" | "Strong",
    "reasoning": "...",
    "key_advantages_a": ["...", "..."],
    "key_advantages_b": ["...", "..."]
}}
"""
        try:
            response = self._call_ollama(prompt)
            return self._parse_json(response)
        except Exception as e:
            return {
                "winner": "Tie",
                "preference_strength": "Weak",
                "reasoning": f"Comparison failed: {str(e)}",
                "key_advantages_a": [],
                "key_advantages_b": []
            }

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.judge_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "repeat_penalty": 1.15,   # penalise repeated tokens to prevent looping
                "repeat_last_n": 128,      # lookback window for repeat penalty
                "num_predict": 600,        # cap output length so model can't spiral
                "stop": ["###", "---"],
            }
        }
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=90)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            raise Exception(f"Ollama call failed: {e}")

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON block from LLM output."""
        try:
            # Look for JSON block
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception:
            # Fallback if parsing fails
            return {"raw_output": text, "status": "error_parsing_json"}
