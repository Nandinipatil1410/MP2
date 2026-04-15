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
    
    def generate_reasoning_plan(self, abstract_query: str) -> Dict[str, any]:
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
        
        prompt = self._create_planning_prompt(abstract_query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a reasoning planner. Generate step-by-step reasoning plans without accessing any private data. Focus only on the logical structure."
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
    
    def _create_planning_prompt(self, query: str) -> str:
        """Create a prompt for plan generation"""
        return f"""Given this query: "{query}"

Generate a highly efficient, concise step-by-step reasoning plan to answer this query.
The plan should be the MINIMAL number of logical steps (MAXIMUM 5) required to extract the answer from private documents.

Format your response as numbered steps:
1. [First reasoning step]
2. [Second reasoning step]
3. [Synthesize final answer]

Strict Requirements:
- NO MORE THAN 5 STEPS.
- Avoid redundant steps or minor details.
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
