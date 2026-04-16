"""
Cloud Reasoning Planner Module
Uses cloud LLM (Groq) to generate reasoning plans without accessing private data
"""
import os
import sys
from pathlib import Path
from typing import List, Dict
import requests
import json
from groq import Groq

# Add root to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import CLOUD_MODEL


class CloudReasoningPlanner:
    """Generate reasoning plans using cloud LLM"""
    
    def __init__(self, api_key: str = None, model: str = None):
        env_key = os.getenv("GROQ_API_KEY")
        self.api_key = api_key or env_key
        # Treat common placeholder values as "not configured" so we can fall back gracefully.
        if self.api_key and self.api_key.strip().lower() in {"your_groq_api_key_here", "your_api_key_here", "changeme"}:
            self.api_key = None
        self.model = model or CLOUD_MODEL
        
        if not self.api_key:
            print("  Warning: No Groq API key found. Using fallback mode.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
    
    def generate_reasoning_plan(self, abstract_query: str, expert_mode: bool = False) -> Dict[str, any]:
        """
        Generate a step-by-step reasoning plan for the query
        Returns structured reasoning steps
        """
        if not self.client:
            # Fallback: Generate simple plan without API
            return {
                'success': True,
                'plan': self._generate_fallback_plan(abstract_query),
                'model': 'fallback'
            }
        
        system_prompt = (
            "You are a strategic reasoning planner for a Privacy-Preserving Hybrid LLM system. "
            "Your role is to generate a structured execution plan that a local agent will follow to process sensitive data. "
            "STRICT RULES:\n"
            "1. You NEVER have access to the actual document content.\n"
            "2. NEVER ask for more information, filenames, or specific data.\n"
            "3. ALWAYS generate a plan based on the logical intent of the query.\n"
            "4. Even if you don't know the document, plan the steps an expert would take to analyze it."
        )
        
        prompt = self._create_planning_prompt(abstract_query, expert_mode)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            plan_text = response.choices[0].message.content
            reasoning_plan = self._parse_plan(plan_text)
            
            return {
                'success': True,
                'plan': reasoning_plan,
                'raw_plan': plan_text,
                'model': self.model
            }
            
        except Exception as e:
            print(f"Error generating plan: {e}")
            return {
                'success': False,
                'error': str(e),
                'plan': self._generate_fallback_plan(abstract_query)
            }

    def synthesize(self, query: str, local_findings: str, expert_mode: bool = False) -> str:
        """
        Use the cloud model to synthesize a rich final answer from local extracted findings.
        This is the key hybrid advantage: local model extracts private data safely,
        cloud model reasons over the sanitized findings to produce a superior response.
        Returns None if cloud is unavailable (caller should fall back to local synthesis).
        """
        if not self.client:
            return None

        if expert_mode:
            system_prompt = (
                "You are a senior academic peer reviewer with expertise in writing detailed, "
                "critical, and well-structured expert reviews. Your reviews are precise, "
                "evidence-grounded, and provide genuine critical insight."
            )
            user_prompt = f"""Below are findings extracted from a private document about: "{query}"

--- LOCAL FINDINGS ---
{local_findings}
--- END FINDINGS ---

Write a comprehensive expert review (3-5 paragraphs, no bullet points) that:
1. Summarizes the document's scope and core contribution
2. Highlights specific strengths with evidence from the findings
3. Identifies concrete gaps, limitations, or weaknesses
4. Gives an overall scholarly assessment

Do NOT invent information not present in the findings. Be analytical, not descriptive."""
        else:
            system_prompt = "You are a helpful, precise assistant that synthesizes research findings into clear, direct answers."
            user_prompt = f"""Answer this question: "{query}"

Using ONLY the following extracted findings from a private document:
--- FINDINGS ---
{local_findings}
--- END ---

Provide a clear, concise, well-structured answer. Do not invent facts not in the findings."""

        try:
            print("     -> Phase 3 [Cloud Synthesis]: Using cloud model for richer response...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.4,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"     -> Cloud synthesis failed ({e}), falling back to local...")
            return None

    def _create_planning_prompt(self, query: str, expert_mode: bool = False) -> str:
        """Create a prompt for plan generation"""
        if expert_mode:
            return f"""Given this query: "{query}"

As an elite research strategist, generate a comprehensive 6-step reasoning plan for a professional academic review and deep analysis.
The plan must cover:
1. Contextual Introduction and Objective Mapping.
2. Structural Deconstruction of the source material.
3. Technical Rigor and Methodology Validation.
4. Identification of Novelty and Significant Contributions.
5. Critical Evaluation of Limitations and Clarity.
6. Synthesis of Executive Findings.

Format your response as numbered steps only. Do not provide any conversational filler."""
        
        return f"""Given this query: "{query}"

Generate a highly efficient, concise step-by-step reasoning plan (MAX 5 steps) to answer this query.
Focus only on the reasoning structure required to extract the answer from private documents.

Format your response as numbered steps:
1. [First reasoning step]
2. [Second reasoning step]
3. [Synthesize final answer]

Strict Requirements:
- NO MORE THAN 5 STEPS.
- Focus only on the reasoning structure, not on specific data."""
    
    def _parse_plan(self, plan_text: str) -> List[str]:
        """Parse plan text into structured steps"""
        steps = []
        lines = plan_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Match numbered steps
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('')):
                # Remove numbering
                step = line.lstrip('0123456789.-) ').strip()
                if step:
                    steps.append(step)
        
        if not steps:
            # If parsing failed, treat each non-empty line as a step
            steps = [line.strip() for line in lines if line.strip()]
        
        return steps
    
    def _generate_fallback_plan(self, query: str) -> List[str]:
        """Generate a simple fallback plan without API"""
        query_lower = query.lower()
        
        # Simple heuristic-based planning
        if 'find' in query_lower or 'search' in query_lower:
            return [
                "Search for relevant documents in the database",
                "Extract key information from found documents",
                "Summarize and present the findings"
            ]
        elif 'compare' in query_lower:
            return [
                "Identify the items to compare",
                "Extract relevant attributes for each item",
                "Compare attributes and highlight differences",
                "Provide a comparative summary"
            ]
        elif 'summarize' in query_lower or 'summary' in query_lower:
            return [
                "Retrieve all relevant documents",
                "Extract main points from each document",
                "Synthesize information into a coherent summary"
            ]
        else:
            return [
                "Understand the query intent",
                "Retrieve relevant information from documents",
                "Analyze and process the information",
                "Generate a comprehensive answer"
            ]
    
    def generate_with_context(self, query: str, context: str = "") -> Dict[str, any]:
        """Generate plan with additional context (still no private data)"""
        enhanced_query = f"{query}\n\nContext: {context}" if context else query
        return self.generate_reasoning_plan(enhanced_query)

    def get_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Get a direct completion from the cloud LLM"""
        if not self.client:
            return "Cloud Only response is simulated because no Groq API Key was provided during initialization."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to cloud: {e}"


if __name__ == "__main__":
    # Test the planner
    planner = CloudReasoningPlanner()
    
    test_query = "Find documents related to machine learning and summarize key findings"
    result = planner.generate_reasoning_plan(test_query)
    
    print("Reasoning Plan Generated:")
    print("=" * 50)
    if result['success']:
        for i, step in enumerate(result['plan'], 1):
            print(f"{i}. {step}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
