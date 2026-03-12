import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class LocalExecutor:
    def __init__(self, model_name=None):
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("LOCAL_MODEL", "mistral")
        
        self.llm = Ollama(
            base_url=self.base_url,
            model=self.model_name,
            temperature=0
        )

    def execute_plan(self, query, reasoning_plan, context_docs):
        """Executes the reasoning plan using local documents as context."""
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        system_prompt = (
            "You are a secure local execution engine. You have access to private document context. "
            "A high-level reasoning plan has been generated for you. Your task is to answer the user's original query "
            "by following the reasoning plan and using ONLY the provided document context. "
            "Keep the response professional and strictly based on the context."
        )
        
        prompt_template = (
            "USER QUERY: {query}\n\n"
            "REASONING PLAN:\n{plan}\n\n"
            "PRIVATE CONTEXT:\n{context}\n\n"
            "Please provide the final answer below:"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", prompt_template)
        ])
        
        # Since Ollama in LangChain is a bit different, we can use a simpler approach or the template
        formatted_prompt = prompt.format(query=query, plan=reasoning_plan, context=context_text)
        
        # In actual implementation, we'd use the chain, but let's be direct for reliability with Ollama
        response = self.llm.invoke(formatted_prompt)
        
        return response
