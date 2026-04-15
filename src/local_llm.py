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
                print(f"✓ Ollama is running")
                return True
        except:
            print("⚠️  Warning: Ollama is not running or not accessible")
            print("   Start Ollama with: ollama serve")
            return False
    
    def execute_plan(self, 
                    reasoning_plan: List[str], 
                    retrieved_docs: List[Dict[str, str]], 
                    original_query: str) -> Dict[str, any]:
        """
        Execute reasoning plan on private documents
        """
        # Prepare context from retrieved documents
        context = self._prepare_context(retrieved_docs)
        
        # Create execution prompt
        prompt = self._create_execution_prompt(
            original_query, 
            reasoning_plan, 
            context
        )
        
        # Generate response using Ollama
        try:
            response = self._call_ollama(prompt)
            
            return {
                'success': True,
                'answer': response,
                'model': self.model,
                'docs_used': len(retrieved_docs)
            }
            
        except Exception as e:
            print(f"Error executing plan: {e}")
            return {
                'success': False,
                'error': str(e),
                'answer': self._fallback_response(original_query, retrieved_docs)
            }
    
    def _prepare_context(self, documents: List[Dict[str, str]]) -> str:
        """Prepare context from retrieved documents"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc['text']}\n")
        
        return "\n".join(context_parts)
    
    def _create_execution_prompt(self, 
                                 query: str, 
                                 plan: List[str], 
                                 context: str) -> str:
        """Create prompt for local LLM execution"""
        plan_text = "\n".join([f"{i}. {step}" for i, step in enumerate(plan, 1)])
        
        prompt = f"""### SYSTEM: You are a senior expert analyst. Provide a professional expert review based on the following document context.
        
### USER QUESTION: {query}

### REASONING PLAN:
{plan_text}

### CONTEXT:
{context}

### STRICT INSTRUCTIONS:
1. Provide ONLY your synthesized expert review. 
2. Do NOT mention "Document 1", "According to the context", or any other source markers.
3. Do NOT repeat the input text or context markers.
4. Follow the reasoning plan to form a comprehensive, polished summary.
5. If the context is technical or bibliographic, extract the underlying themes and synthesize them into a readable expert perspective.
6. START YOUR ANSWER DIRECTLY WITH THE EXPERT REVIEW.

### EXPERT REVIEW:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str, max_tokens: int = 1024) -> str:
        """Call Ollama API for generation"""
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": max_tokens,
                "repeat_penalty": 1.2,
                "top_p": 0.9,
                "stop": ["###", "Context:", "User Question:"]
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
        print(f"✓ Model '{executor.model}' is available")
    else:
        print(f"⚠️  Model '{executor.model}' not found. Pull it with: ollama pull {executor.model}")
    
    # Test simple generation
    test_response = executor.generate_simple("Say hello in one sentence.")
    print(f"\nTest response: {test_response}")
