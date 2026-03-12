import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class CloudPlanner:
    def __init__(self, model_name=None):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model_name = model_name or os.getenv("CLOUD_MODEL", "llama3-70b-8192")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables.")
            
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=self.api_key,
            model_name=self.model_name
        )

    def generate_plan(self, abstract_query):
        """Generates a step-by-step reasoning plan."""
        system_prompt = (
            "You are a high-level reasoning planner. You will be given an abstract query where sensitive details are masked with placeholders like <EMAIL>, <PHONE>, etc. "
            "Your task is NOT to answer the query, but to provide a detailed, step-by-step reasoning plan that a local executor should follow to find the answer using private documents. "
            "Break down the reasoning into logical steps. Do not invent any facts."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"query": abstract_query})
        
        return response.content
