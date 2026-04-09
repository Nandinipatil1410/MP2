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
        """Simple heuristic for answer quality when no judge model is available."""
        length_score = min(1.0, len(answer.split()) / 50.0) if answer else 0.0

        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(query_words & answer_words)
        relevance_score = min(1.0, overlap / max(len(query_words), 1))

        return (length_score + relevance_score) / 2.0

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
    """
    LLM-as-a-judge evaluator for comparing local vs hybrid answers.
    Uses Groq when an API key is available and falls back to heuristics otherwise.
    """

    def __init__(
        self,
        api_key: str | None = None,
        judge_model: str | None = None,
        allow_groq_fallback: bool = False,
    ):
        # Prefer an independent judge provider when possible:
        # 1) Gemini (if GEMINI_API_KEY/GOOGLE_API_KEY is set)
        # 2) Groq (optional fallback if allow_groq_fallback is True)
        # 3) Fallback heuristic judge
        self.gemini_api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        self.groq_api_key = (api_key or os.getenv("GROQ_API_KEY") or "").strip()
        if self.gemini_api_key.lower() in {"your_gemini_api_key_here", "your_api_key_here", "changeme"}:
            self.gemini_api_key = ""
        if self.groq_api_key.lower() in {"your_groq_api_key_here", "your_api_key_here", "changeme"}:
            self.groq_api_key = ""

        self.allow_groq_fallback = allow_groq_fallback
        if self.gemini_api_key:
            self.judge_provider = "gemini"
        elif allow_groq_fallback and Groq and self.groq_api_key:
            self.judge_provider = "groq"
        else:
            self.judge_provider = "heuristic"
        if judge_model:
            self.judge_model = judge_model.strip()
        elif self.judge_provider == "gemini":
            # Model names can change over time; "gemini-flash-latest" is a safer default.
            self.judge_model = (os.getenv("GEMINI_JUDGE_MODEL") or "gemini-flash-latest").strip()
        else:
            self.judge_model = "llama-3.3-70b-versatile"

        self.client = Groq(api_key=self.groq_api_key) if (self.judge_provider == "groq" and Groq) else None
        self.metrics = EvaluationMetrics()

    def evaluate_answer(self, question: str, answer: str, reference_docs: str | List[Dict[str, Any]] | List[str]) -> Dict[str, Any]:
        """Evaluate one answer on multiple criteria."""
        reference_text = self._format_reference_docs(reference_docs)

        if self.judge_provider == "heuristic":
            return self._fallback_answer_evaluation(question, answer, reference_text)

        prompt = f"""You are an expert evaluator for document-grounded QA systems.

Question:
{question}

Answer to Evaluate:
{answer}

Reference Documents:
{reference_text}

Evaluate the answer on these criteria from 1 to 10:
1. accuracy
2. completeness
3. relevance
4. coherence
5. groundedness

Return valid JSON only with this structure:
{{
  "accuracy": {{"score": 0, "reasoning": ""}},
  "completeness": {{"score": 0, "reasoning": ""}},
  "relevance": {{"score": 0, "reasoning": ""}},
  "coherence": {{"score": 0, "reasoning": ""}},
  "groundedness": {{"score": 0, "reasoning": ""}},
  "overall_score": 0,
  "summary": ""
}}
"""
        response_text = self.call_judge_llm(prompt)
        parsed = self._parse_json_response(response_text, default=self._fallback_answer_evaluation(question, answer, reference_text))
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

        if self.judge_provider == "heuristic":
            return self._fallback_answer_comparison(question, answer_local, answer_hybrid, reference_text)

        prompt = f"""You are an expert evaluator comparing two AI-generated answers grounded in reference documents.

Question:
{question}

Answer A (Local-Only):
{answer_local}

Answer B (Hybrid):
{answer_hybrid}

Reference Documents:
{reference_text}

Decide which answer is better. Focus on:
- accuracy
- completeness
- coherence
- detail level
- reasoning quality
- groundedness in the documents

Return valid JSON only with this structure:
{{
  "winner": "A",
  "answer_a_score": 0,
  "answer_b_score": 0,
  "key_differences": ["", ""],
  "reasoning": "",
  "preference_strength": "Weak"
}}
"""
        response_text = self.call_judge_llm(prompt)
        parsed = self._parse_json_response(
            response_text,
            default=self._fallback_answer_comparison(question, answer_local, answer_hybrid, reference_text),
        )
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
        """Call the judge LLM and return raw text."""
        if self.judge_provider == "gemini":
            return self._call_gemini(prompt)
        if self.judge_provider != "groq" or not self.client:
            raise RuntimeError("Judge client is not configured.")

        # Prefer requesting structured JSON if supported by the SDK/model,
        # but keep parsing robust if it's not.
        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict evaluator. Return only valid JSON with no markdown fencing.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=1400,
                response_format={"type": "json_object"},
            )
        except TypeError:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict evaluator. Return only valid JSON with no markdown fencing.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_tokens=1400,
            )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini via REST (Google AI Studio / Generative Language API).
        Requires GEMINI_API_KEY or GOOGLE_API_KEY.
        """
        if not self.gemini_api_key:
            raise RuntimeError("Gemini judge is not configured.")

        model = self.judge_model.strip()
        # Accept either "gemini-flash-latest" or "models/gemini-flash-latest"
        if model.startswith("models/"):
            model = model[len("models/") :]

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        params = {"key": self.gemini_api_key}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 1400,
            },
        }

        max_attempts = int(os.getenv("GEMINI_JUDGE_RETRIES", "3"))
        timeout_s = int(os.getenv("GEMINI_JUDGE_TIMEOUT", "60"))

        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = requests.post(url, params=params, json=payload, timeout=timeout_s)
                if response.status_code >= 400:
                    # Common failure mode: model renamed / not supported for generateContent -> 404
                    if response.status_code == 404 and model != "gemini-flash-latest":
                        self.judge_model = "gemini-flash-latest"
                        return self._call_gemini(prompt)

                    # Retry on transient errors.
                    if response.status_code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                        backoff = (0.8 * (2 ** attempt)) + random.uniform(0, 0.35)
                        time.sleep(backoff)
                        continue

                    if response.status_code == 404:
                        available = self._list_gemini_models(max_models=20)
                        hint = ""
                        if available:
                            hint = " Available models (subset): " + ", ".join(available)
                        raise RuntimeError(
                            f"Gemini model '{model}' was not found or not supported for generateContent.{hint}"
                        )

                    raise RuntimeError(
                        f"Gemini judge request failed (HTTP {response.status_code}). Response: {response.text[:400]}"
                    )

                data = response.json()
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
                if attempt < max_attempts - 1:
                    backoff = (0.8 * (2 ** attempt)) + random.uniform(0, 0.35)
                    time.sleep(backoff)
                    continue
                raise RuntimeError("Gemini judge request failed due to a network timeout/connection error.") from exc
            except Exception as exc:
                last_error = exc
                break
        else:
            raise RuntimeError("Gemini judge request failed after retries.")

        if last_error and "data" not in locals():
            raise RuntimeError("Gemini judge request failed.") from last_error

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("Gemini judge returned no candidates.")

        content = (candidates[0].get("content") or {})
        parts = content.get("parts") or []
        text_parts = [part.get("text", "") for part in parts if isinstance(part, dict)]
        return "\n".join([part for part in text_parts if part]).strip()

    def _list_gemini_models(self, max_models: int = 50) -> List[str]:
        """List available Gemini models for the provided API key (best-effort)."""
        if not self.gemini_api_key:
            return []
        try:
            response = requests.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": self.gemini_api_key, "pageSize": max_models},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            models = data.get("models") or []
            names: List[str] = []
            for item in models:
                name = item.get("name")
                if not name:
                    continue
                # Prefer models that can generate content.
                methods = item.get("supportedGenerationMethods") or []
                if "generateContent" in methods:
                    # Strip "models/" prefix for display consistency.
                    names.append(name.replace("models/", ""))
            return names[:max_models]
        except Exception:
            return []

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
            return json.loads(response_text)
        except (json.JSONDecodeError, TypeError):
            pass

        if not response_text:
            return default

        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            return default

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return default

    def _fallback_answer_evaluation(self, question: str, answer: str, reference_text: str) -> Dict[str, Any]:
        """Fallback evaluator when no judge LLM is configured."""
        base_score = round(self.metrics.calculate_accuracy_heuristic(answer, question) * 10, 2)
        groundedness = 8.0 if answer and reference_text else 4.0

        return {
            "accuracy": {"score": base_score, "reasoning": "Heuristic score based on relevance and answer length."},
            "completeness": {"score": base_score, "reasoning": "Longer, more query-aligned answers score higher heuristically."},
            "relevance": {"score": base_score, "reasoning": "Keyword overlap suggests how directly the answer addresses the query."},
            "coherence": {"score": min(10.0, base_score + 0.5), "reasoning": "Fallback assumes readable answers unless empty."},
            "groundedness": {"score": groundedness, "reasoning": "Groundedness is approximated because no judge LLM is available."},
            "overall_score": round(mean([base_score, base_score, base_score, min(10.0, base_score + 0.5), groundedness]), 2),
            "summary": "Fallback heuristic evaluation used because no judge API key was provided.",
        }

    def _fallback_answer_comparison(
        self,
        question: str,
        answer_local: str,
        answer_hybrid: str,
        reference_text: str,
    ) -> Dict[str, Any]:
        """Fallback comparison when no judge LLM is available."""
        local_eval = self._fallback_answer_evaluation(question, answer_local, reference_text)
        hybrid_eval = self._fallback_answer_evaluation(question, answer_hybrid, reference_text)

        local_score = local_eval["overall_score"]
        hybrid_score = hybrid_eval["overall_score"]

        if abs(hybrid_score - local_score) < 0.3:
            winner = "Tie"
            strength = "Weak"
        elif hybrid_score > local_score:
            winner = "B"
            strength = "Moderate" if hybrid_score - local_score < 1.0 else "Strong"
        else:
            winner = "A"
            strength = "Moderate" if local_score - hybrid_score < 1.0 else "Strong"

        return {
            "winner": winner,
            "answer_a_score": round(local_score, 2),
            "answer_b_score": round(hybrid_score, 2),
            "key_differences": [
                "Hybrid answers may include more structure when planning helps, but that depends on the query and docs.",
                "Local-only answers can be equally strong when retrieval captures the right context.",
            ],
            "reasoning": "Fallback heuristic comparison used because no judge API key was provided.",
            "preference_strength": strength,
        }

    def _normalize_answer_evaluation(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure answer evaluation fields exist and are well-typed."""
        criteria = ["accuracy", "completeness", "relevance", "coherence", "groundedness"]
        normalized = {}

        for criterion in criteria:
            item = parsed.get(criterion, {})
            score = float(item.get("score", 0))
            normalized[criterion] = {
                "score": round(score, 2),
                "reasoning": str(item.get("reasoning", "")),
            }

        overall = parsed.get("overall_score")
        if overall is None:
            overall = mean(normalized[criterion]["score"] for criterion in criteria)

        normalized["overall_score"] = round(float(overall), 2)
        normalized["summary"] = str(parsed.get("summary", ""))
        return normalized

    def _normalize_comparison(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure comparison fields exist and are well-typed."""
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
