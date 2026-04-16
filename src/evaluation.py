"""
Evaluation Module
Metrics and automated LLM-as-a-judge evaluation for comparing approaches
"""
from __future__ import annotations

import json
import os
import re
import time
import random
from statistics import mean
from typing import Any, Dict, List

import numpy as np
import requests
from rouge_score import rouge_scorer

try:
    from groq import Groq  # type: ignore
except ImportError:  # pragma: no cover
    Groq = None  # type: ignore


class EvaluationMetrics:
    """Basic evaluation metrics for the system."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        scores = self.rouge_scorer.score(reference, hypothesis)

        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def calculate_privacy_score(self, query: str, abstracted_query: str) -> float:
        """
        Calculate privacy preservation score.
        Higher score = better privacy.
        """
        removed_info_ratio = 1.0 - (len(abstracted_query) / max(len(query), 1))
        sensitive_patterns = ["@", "phone", "ssn", "account", "password"]
        sensitivity_penalty = sum(1 for pattern in sensitive_patterns if pattern in abstracted_query.lower())

        score = max(0.0, 1.0 - (sensitivity_penalty * 0.1))
        return max(score, removed_info_ratio * 0.5 + score * 0.5)

    def calculate_latency_score(self, latency: float, baseline: float = 5.0) -> float:
        """Calculate latency score in [0, 1]."""
        if latency <= baseline:
            return 1.0
        return max(0.0, float(np.exp(-(latency - baseline) / baseline)))

    def calculate_accuracy_heuristic(self, answer: str, query: str) -> float:
        """Heuristic for answer quality that penalizes repetition and rewards relevance."""
        if not answer:
            return 0.0

        # 1. Length Score (up to 100 words, but not too short)
        words = answer.split()
        length_score = min(1.0, len(words) / 100.0)

        # 2. Relevance Score (word overlap with query)
        query_words = set(query.lower().split())
        answer_words_set = set(answer.lower().split())
        overlap = len(query_words & answer_words_set)
        relevance_score = min(1.0, overlap / max(len(query_words), 1))

        # 3. Repetition Penalty (detect duplicate paragraphs or heavy word-set vs list mismatch)
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        unique_paragraphs = set(paragraphs)
        repetition_multiplier = 1.0
        if len(paragraphs) > 1:
            repetition_multiplier = len(unique_paragraphs) / len(paragraphs)
        
        # Also check word variety
        word_variety = len(answer_words_set) / max(len(words), 1)
        if word_variety < 0.4: # Very low variety usually means loops
            repetition_multiplier *= (word_variety / 0.4)

        score = (length_score * 0.3 + relevance_score * 0.7) * repetition_multiplier
        return min(1.0, score)

    def compare_approaches(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare approaches using lightweight heuristics."""
        comparison = {
            "latency": {},
            "privacy": {},
            "answers": {},
        }

        for approach, result in results.items():
            comparison["latency"][approach] = result.get("latency", 0)
            comparison["privacy"][approach] = result.get("privacy_score", 0)
            comparison["answers"][approach] = result.get("answer", "")

        comparison["fastest"] = min(comparison["latency"], key=comparison["latency"].get)
        comparison["most_private"] = max(comparison["privacy"], key=comparison["privacy"].get)

        hybrid_score = (
            (1 - comparison["latency"]["hybrid"] / max(comparison["latency"].values())) * 0.4
            + comparison["privacy"]["hybrid"] * 0.6
        )
        local_score = comparison["privacy"]["local_only"]
        comparison["recommended"] = "hybrid" if hybrid_score >= local_score * 0.9 else "local_only"

        return comparison

    def generate_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Generate a plain-text report of evaluation metrics."""
        comparison = self.compare_approaches(results)

        report = "=" * 60 + "\n"
        report += "EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"

        report += "LATENCY:\n"
        for approach, latency in comparison["latency"].items():
            report += f"  {approach:15s}: {latency:.2f}s\n"
        report += f"  Fastest: {comparison['fastest']}\n\n"

        report += "PRIVACY SCORE:\n"
        for approach, privacy in comparison["privacy"].items():
            report += f"  {approach:15s}: {privacy:.2f}\n"
        report += f"  Most Private: {comparison['most_private']}\n\n"

        report += f"RECOMMENDATION: {comparison['recommended']}\n"
        report += "=" * 60 + "\n"
        return report


class LLMJudge:
    """Judge LLM that evaluates answers and compares approaches using local models."""

    def __init__(self, judge_model: str = None):
        """
        Initialize the judge with local Ollama.
        Defaults to OLLAMA_JUDGE_MODEL from .env or 'mistral:latest'.
        """
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
        self.judge_model = (judge_model or os.getenv("OLLAMA_JUDGE_MODEL") or "mistral:latest").strip()
        self.metrics = EvaluationMetrics()
        
        print(f"  Judging via Local Ollama ({self.judge_model})")

    def _check_ollama(self) -> bool:
        """Check if local Ollama is available."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def evaluate_answer(self, question: str, answer: str, reference_docs: str | List[Dict[str, Any]] | List[str]) -> Dict[str, Any]:
        """Evaluate one answer on multiple criteria."""
        reference_text = self._format_reference_docs(reference_docs)

        # No fallback logic needed; if Ollama is down, it should raise an error for the user to fix

        has_context = bool(reference_text.strip())
        context_block = reference_text if has_context else "[No specific reference documents — evaluate on internal quality]"
        groundedness_rule = (
            "GROUNDEDNESS: Score how well claims are supported by the reference documents above."
            if has_context else
            "GROUNDEDNESS: No reference documents were provided. Score based on internal consistency and factual plausibility."
        )

        prompt = f"""### LLM JUDGE — ANSWER EVALUATION

You are a CALIBRATED EVALUATOR. You MUST use the full scoring range — do NOT give 1.0 to a brief or generic answer.

### SCORING RUBRIC (apply to ALL criteria):
- 0.0–0.3: Very poor — missing, wrong, or completely off-topic
- 0.4–0.6: Basic — answers superficially or in 1-2 sentences with minimal detail
- 0.7–0.8: Good — reasonably complete, specific, and accurate
- 0.9–1.0: Exceptional — very thorough, detailed, well-cited, covers multiple aspects

### SCORING CRITERIA:
1. ACCURACY: Are the specific facts, entities, and claims correct?
2. COMPLETENESS: Does the answer cover ALL key aspects, or only a superficial overview? (A 2-sentence answer should score MAX 0.6)
3. RELEVANCE: Is the answer directly addressing the specific question asked?
4. COHERENCE: Is the answer well-structured, specific, and non-repetitive?
5. {groundedness_rule}

### CRITICAL RULE: A short/brief answer that covers the topic only at a surface level must score 0.4-0.6 on COMPLETENESS and COHERENCE, not 0.9 or 1.0. Reserve 0.9+ for answers that are detailed and thorough.

### PENALTIES (set score to 0.0 for that criterion only):
- Deduct severely for obvious hallucinations
- Deduct for heavy repetition (same paragraph repeated 2+ times)

### REFERENCE DOCUMENTS:
{context_block}

### QUESTION:
{question}

### CANDIDATE ANSWER:
{answer}

### RESPONSE (JSON ONLY, no other text):
{{
  "accuracy": {{ "score": <0.0-1.0>, "reasoning": "<brief>" }},
  "completeness": {{ "score": <0.0-1.0>, "reasoning": "<brief>" }},
  "relevance": {{ "score": <0.0-1.0>, "reasoning": "<brief>" }},
  "coherence": {{ "score": <0.0-1.0>, "reasoning": "<brief>" }},
  "groundedness": {{ "score": <0.0-1.0>, "reasoning": "<brief>" }},
  "overall_score": <0.0-1.0>,
  "summary": "<one sentence verdict>"
}}"""
        response_text = self.call_judge_llm(prompt)
        parsed = self._parse_json_response(response_text, default={})
        return self._normalize_answer_evaluation(parsed)

    def compare_answers(
        self,
        question: str,
        answer_local: str,
        answer_hybrid: str,
        reference_docs: str | List[Dict[str, Any]] | List[str],
    ) -> Dict[str, Any]:
        """Directly compare local-only vs hybrid answers."""
        reference_text = self._format_reference_docs(reference_docs)

        # No fallback in local-only mode

        prompt = f"""### LLM JUDGE — COMPARISON EVALUATION

You are comparing two AI-generated answers to determine which is BETTER. Both answers are responses to the same user task/question.

> IMPORTANT: The QUESTION below is always a user's task or query, even if it seems short or directive (e.g. "Generate Expert Review" or "Summarize"). Your job is to compare which answer fulfills that task better.

### TASK/QUESTION asked by the user:
{question}

### CANDIDATE A (Local-Only Model):
{answer_local}

### CANDIDATE B (Hybrid Model):
{answer_hybrid}

### REFERENCE DOCUMENTS (for context):
{reference_text if reference_text.strip() else "[No specific documents — compare on general quality]"}

### EVALUATION CRITERIA (in order of importance):
1. **Depth & Technical Detail** — Which answer provides more substantive, specific information?
2. **Structure & Clarity** — Which is better organized and easier to read?
3. **Completeness** — Which covers more aspects of the task?
4. **Non-repetition** — Penalize answers that repeat the same sentences or paragraphs.

### OUTPUT (JSON ONLY):
{{
  "winner": "<A or B or Tie — ONLY 'Tie' if genuinely equivalent>",
  "answer_a_score": <0.0-1.0>,
  "answer_b_score": <0.0-1.0>,
  "key_differences": ["<specific difference 1>", "<specific difference 2>"],
  "reasoning": "<which answer is better and why, referencing specific content>",
  "preference_strength": "<Weak|Moderate|Strong>"
}}"""
        response_text = self.call_judge_llm(prompt)
        parsed = self._parse_json_response(response_text, default={})
        return self._normalize_comparison(parsed)

    def evaluate_systems(self, orchestrator, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Run local-only and hybrid, then evaluate both answers automatically.
        Returns scores plus a comparison explaining why one is better.
        """
        reference_docs = orchestrator.vector_store.search(question, top_k=top_k) if orchestrator.documents_loaded else []
        local_result = orchestrator.process_query_local_only(question, top_k=top_k)
        hybrid_result = orchestrator.process_query_hybrid(question, top_k=top_k)

        local_eval = self.evaluate_answer(question, local_result.get("answer", ""), reference_docs)
        hybrid_eval = self.evaluate_answer(question, hybrid_result.get("answer", ""), reference_docs)
        comparison = self.compare_answers(
            question,
            local_result.get("answer", ""),
            hybrid_result.get("answer", ""),
            reference_docs,
        )

        return {
            "question": question,
            "reference_docs": self._format_reference_docs(reference_docs),
            "local_only": {
                "result": local_result,
                "evaluation": local_eval,
            },
            "hybrid": {
                "result": hybrid_result,
                "evaluation": hybrid_eval,
            },
            "comparison": comparison,
        }

    def evaluate_batch(self, orchestrator, questions: List[str], top_k: int = 3) -> Dict[str, Any]:
        """Evaluate multiple questions and compute aggregate results."""
        batch_results = [self.evaluate_systems(orchestrator, question, top_k=top_k) for question in questions]

        local_scores = [item["local_only"]["evaluation"]["overall_score"] for item in batch_results]
        hybrid_scores = [item["hybrid"]["evaluation"]["overall_score"] for item in batch_results]
        hybrid_wins = sum(1 for item in batch_results if item["comparison"]["winner"] == "B")
        local_wins = sum(1 for item in batch_results if item["comparison"]["winner"] == "A")
        ties = sum(1 for item in batch_results if item["comparison"]["winner"] == "Tie")

        return {
            "per_question": batch_results,
            "summary": {
                "questions_evaluated": len(questions),
                "average_local_score": round(mean(local_scores), 2) if local_scores else 0.0,
                "average_hybrid_score": round(mean(hybrid_scores), 2) if hybrid_scores else 0.0,
                "hybrid_wins": hybrid_wins,
                "local_wins": local_wins,
                "ties": ties,
                "overall_winner": "Hybrid" if hybrid_wins > local_wins else ("Local" if local_wins > hybrid_wins else "Tie"),
            },
        }

    def call_judge_llm(self, prompt: str) -> str:
        """Call the local Ollama judge and return raw text."""
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API for evaluation."""
        try:
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.judge_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 1400  # Increased for more detailed reasoning
                }
            }
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            print(f"Error calling Ollama judge: {e}")
            raise RuntimeError(f"Ollama judge failed: {e}")

    def _format_reference_docs(self, reference_docs: str | List[Dict[str, Any]] | List[str]) -> str:
        """Normalize reference docs into judge-friendly text."""
        if isinstance(reference_docs, str):
            return reference_docs

        formatted = []
        for index, doc in enumerate(reference_docs, start=1):
            if isinstance(doc, dict):
                source = doc.get("source", f"Document {index}")
                text = doc.get("text", "")
            else:
                source = f"Document {index}"
                text = str(doc)
            formatted.append(f"[{source}]\n{text}")
        return "\n\n".join(formatted)

    def _parse_json_response(self, response_text: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw JSON or recover JSON embedded in text."""
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        if not response_text:
            return default

        # Try to extract JSON from text if it's not a pure JSON response
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        return default

    def _normalize_answer_evaluation(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure answer evaluation fields exist and are well-typed."""
        criteria = ["accuracy", "completeness", "relevance", "coherence", "groundedness"]
        normalized = {}

        if not isinstance(parsed, dict):
            parsed = {}

        for criterion in criteria:
            item = parsed.get(criterion, {})
            # Handle both {"score": 10} and raw 10 formats
            if isinstance(item, (int, float)):
                score = float(item)
                reasoning = ""
            elif isinstance(item, dict):
                score = float(item.get("score", 0))
                reasoning = str(item.get("reasoning", ""))
            else:
                score = 0.0
                reasoning = ""

            normalized[criterion] = {
                "score": round(score, 2),
                "reasoning": reasoning,
            }

        overall = parsed.get("overall_score")
        if overall is None or not isinstance(overall, (int, float)):
            # If no overall score, calculate average of non-zero criterion scores
            valid_scores = [normalized[c]["score"] for c in criteria if normalized[c]["score"] > 0]
            overall = mean(valid_scores) if valid_scores else 0.0

        normalized["overall_score"] = round(float(overall), 2)
        normalized["summary"] = str(parsed.get("summary", ""))
        return normalized

    def _normalize_comparison(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure comparison fields exist and are well-typed."""
        if not isinstance(parsed, dict):
            parsed = {}

        winner = str(parsed.get("winner", "Tie"))
        if winner not in {"A", "B", "Tie"}:
            winner = "Tie"

        differences = parsed.get("key_differences", [])
        if not isinstance(differences, list):
            differences = [str(differences)]

        return {
            "winner": winner,
            "answer_a_score": round(float(parsed.get("answer_a_score", 0)), 2),
            "answer_b_score": round(float(parsed.get("answer_b_score", 0)), 2),
            "key_differences": [str(item) for item in differences],
            "reasoning": str(parsed.get("reasoning", "")),
            "preference_strength": str(parsed.get("preference_strength", "Weak")),
        }


if __name__ == "__main__":
    evaluator = EvaluationMetrics()
    ref = "The cat sat on the mat"
    hyp = "The cat is on the mat"
    rouge_scores = evaluator.calculate_rouge(ref, hyp)
    print(f"ROUGE scores: {rouge_scores}")
