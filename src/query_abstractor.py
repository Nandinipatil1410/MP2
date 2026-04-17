"""
Query Abstraction Module
Removes sensitive information from queries before sending to cloud
"""
import re
from typing import Dict, List, Tuple


class QueryAbstractor:
    """Abstract queries to remove sensitive information"""
    
    def __init__(self):
        # Patterns for detecting sensitive information
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'money': r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?',
        }
        
        self.sensitive_keywords = [
            'name', 'address', 'salary', 'password', 'account number',
            'diagnosis', 'patient', 'medical record', 'confidential',
            'private', 'secret', 'personal', 'ssn', 'social security'
        ]
        
        self.replacements = {}
    
    def detect_sensitive_info(self, query: str) -> Dict[str, List[str]]:
        """Detect sensitive information in query"""
        detected = {}
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                detected[pattern_name] = matches
        
        # Check for sensitive keywords
        found_keywords = []
        query_lower = query.lower()
        for keyword in self.sensitive_keywords:
            if keyword in query_lower:
                found_keywords.append(keyword)
        
        if found_keywords:
            detected['keywords'] = found_keywords
        
        return detected
    
    def abstract_query(self, query: str) -> Tuple[str, Dict[str, any]]:
        """
        Abstract query by removing/replacing sensitive information
        Returns: (abstracted_query, metadata)
        """
        abstracted = query
        self.replacements = {}
        
        # Replace patterns with placeholders
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, abstracted, re.IGNORECASE)
            for i, match in enumerate(matches):
                placeholder = f"[{pattern_name.upper()}_{i}]"
                self.replacements[placeholder] = match.group()
                abstracted = abstracted.replace(match.group(), placeholder)
        
        # Extract intent while removing specific details
        abstracted = self._generalize_query(abstracted)
        
        metadata = {
            'original_length': len(query),
            'abstracted_length': len(abstracted),
            'sensitive_items_removed': len(self.replacements),
            'replacements': self.replacements
        }
        
        return abstracted, metadata
    
    def _generalize_query(self, query: str) -> str:
        """Generalize query to focus on intent while preserving instructions"""
        # Whitelist of terms that should NEVER be masked as [PERSON]
        whitelist = ['Expert', 'Review', 'Generate', 'Analysis', 'Report', 'Summary']
        
        def preserve_whitelist(text):
            # Temporarily hide whitelisted words
            preserved = {}
            for i, word in enumerate(whitelist):
                marker = f"__PRESERVED_{i}__"
                if word in text:
                    preserved[marker] = word
                    text = text.replace(word, marker)
            return text, preserved

        def restore_whitelist(text, preserved):
            for marker, word in preserved.items():
                text = text.replace(marker, word)
            return text

        # 1. Temporarily hide whitelisted instructions
        temp_query, preserved_map = preserve_whitelist(query)

        # 2. Mask obvious names (Two capitalized words, but avoid sentence-initial verbs)
        # We only mask if it's NOT at the very start of the query (likely a verb) 
        # OR if it's clearly a name pattern.
        temp_query = re.sub(r'(?<!^)\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[PERSON]', temp_query)
        
        # 3. Handle sentence-initial if it really looks like a name (e.g. John Doe starts...)
        # but avoid instructions like "Generate Expert"
        # We'll stick to a safer exclusion loop for now.

        # 4. Generalize specific numbers
        temp_query = re.sub(r'\b\d{4,}\b', '[NUMBER]', temp_query)
        
        # 5. Restore whitelisted terms
        final_query = restore_whitelist(temp_query, preserved_map)
        
        return final_query
    
    def reconstruct_response(self, response: str) -> str:
        """Reconstruct response by replacing placeholders with original values"""
        reconstructed = response
        
        for placeholder, original in self.replacements.items():
            reconstructed = reconstructed.replace(placeholder, original)
        
        return reconstructed
    
    def calculate_privacy_score(self, query: str) -> float:
        """
        Calculate privacy score (0-1, higher is better)
        Based on absence of sensitive information
        """
        detected = self.detect_sensitive_info(query)
        
        if not detected:
            return 1.0
        
        # Calculate score based on types and count of sensitive info
        total_sensitive = sum(len(v) if isinstance(v, list) else 1 
                            for v in detected.values())
        
        # Penalize more for more sensitive information
        score = max(0.0, 1.0 - (total_sensitive * 0.1))
        
        return score
    
    def classify_query_sensitivity(self, query: str) -> str:
        """Classify query as LOW, MEDIUM, or HIGH sensitivity"""
        score = self.calculate_privacy_score(query)
        
        if score >= 0.9:
            return "LOW"
        elif score >= 0.7:
            return "MEDIUM"
        else:
            return "HIGH"

    def classify_query_type_keywords(self, query: str) -> Dict[str, any]:
        """
        KEYWORD-BASED FALLBACK classifier (used internally by pipeline/intent_classifier.py).
        Primary intent classification is now LLM-based (via intent_classifier.py).
        Kept here for reference and testing.
        """
        """
        Lightweight local query-type classifier used for routing.

        Returns:
          {
            "query_type": "summary" | "extraction" | "reasoning" | "comparison" | "hypothetical",
            "confidence": float (0-1),
            "should_use_agentic": bool,
            "reason": str
          }
        """
        q = (query or "").strip()
        ql = re.sub(r"\s+", " ", q.lower())

        if not ql:
            return {
                "query_type": "extraction",
                "confidence": 0.2,
                "should_use_agentic": False,
                "reason": "Empty query treated as direct lookup.",
            }

        # --- Summary / description (must override other patterns like "what is") ---
        summary_phrases = [
            "what is this document about",
            "what is the document about",
            "what is this paper about",
            "what is this report about",
            "what does this document talk about",
            "what does this paper talk about",
            "give me an overview",
            "high level overview",
            "summarize",
            "summary of",
            "describe this document",
            "describe this paper",
            "document overview",
            "paper overview",
            "main idea",
            "what is this about",
        ]
        if any(p in ql for p in summary_phrases):
            return {
                "query_type": "summary",
                "confidence": 0.95,
                "should_use_agentic": False,
                "reason": "Detected summary/overview intent.",
            }

        # --- Comparison ---
        comparison_markers = ["compare", "difference between", "differences between", "vs ", " vs.", "versus"]
        if any(m in ql for m in comparison_markers):
            return {
                "query_type": "comparison",
                "confidence": 0.8,
                "should_use_agentic": True,
                "reason": "Detected comparison intent.",
            }

        # --- Hypothetical / counterfactual ---
        hypothetical_markers = ["what if", "suppose", "imagine", "if we ", "if the "]
        if any(m in ql for m in hypothetical_markers):
            return {
                "query_type": "hypothetical",
                "confidence": 0.75,
                "should_use_agentic": True,
                "reason": "Detected hypothetical intent.",
            }

        # --- Reasoning / explanation ---
        reasoning_markers = ["why", "how", "explain", "reason", "cause", "despite", "impact", "effect", "drivers"]
        if any(re.search(rf"\b{re.escape(m)}\b", ql) for m in reasoning_markers):
            return {
                "query_type": "reasoning",
                "confidence": 0.7,
                "should_use_agentic": True,
                "reason": "Detected explanatory/causal intent.",
            }

        # --- Extraction / lookup (default) ---
        extraction_markers = [
            "what is",
            "who is",
            "when",
            "where",
            "how many",
            "list",
            "extract",
            "show",
            "give me",
            "find",
        ]
        if any(ql.startswith(m) or f" {m} " in ql for m in extraction_markers):
            return {
                "query_type": "extraction",
                "confidence": 0.65,
                "should_use_agentic": False,
                "reason": "Detected direct lookup intent.",
            }

        return {
            "query_type": "extraction",
            "confidence": 0.4,
            "should_use_agentic": False,
            "reason": "Defaulted to extraction (no strong markers).",
        }


if __name__ == "__main__":
    # Test the abstractor
    abstractor = QueryAbstractor()
    
    test_queries = [
        "What is the salary of John Doe?",
        "Find all documents about machine learning",
        "Show me emails from john@example.com about the project",
        "What is my account balance for account 123456789?"
    ]
    
    for query in test_queries:
        print(f"\nOriginal: {query}")
        abstracted, metadata = abstractor.abstract_query(query)
        print(f"Abstracted: {abstracted}")
        print(f"Sensitivity: {abstractor.classify_query_sensitivity(query)}")
        print(f"Privacy Score: {abstractor.calculate_privacy_score(query):.2f}")
