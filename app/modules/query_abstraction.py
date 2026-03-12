import re

class QueryAbstrator:
    def __init__(self):
        # Basic patterns for PII detection
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "name_placeholder": r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Z][a-z]+\b',
            "id_num": r'\b\d{10,12}\b' # Generic ID number pattern
        }

    def mask_pii(self, text):
        """Masks sensitive information in the query."""
        masked_text = text
        for label, pattern in self.pii_patterns.items():
            masked_text = re.sub(pattern, f"<{label.upper()}>", masked_text)
        return masked_text

    def abstract_query(self, query):
        """
        Converts a specific query into an abstract reasoning task.
        In a real scenario, this might use a small local LLM or NLP rules.
        For this prototype, we'll perform PII masking and structured transformation.
        """
        masked_query = self.mask_pii(query)
        
        # Simple heuristic to extract intent
        # In a more advanced version, we'd use intent classification
        abstract_prompt = f"Perform a reasoning task based on this intent: {masked_query}. " \
                          f"Focus on the logical steps required to answer, without knowing the specific details inside the <...> placeholders."
        
        return abstract_prompt, masked_query
