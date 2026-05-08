"""
Configuration settings for Privacy-Preserving Hybrid LLM System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vector_db"

# Create directories if they don't exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "mistral")
CLOUD_MODEL = os.getenv("CLOUD_MODEL", "llama-3.3-70b-versatile")  # Groq model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Vector Database Settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 20

# Query Abstraction Settings
SENSITIVE_KEYWORDS = [
    "name", "email", "phone", "address", "ssn", "credit card",
    "password", "account", "salary", "diagnosis", "patient",
    "confidential", "private", "secret"
]

# Evaluation Settings
EVAL_METRICS = ["rouge", "bertscore", "latency", "privacy_score"]

# UI Settings
APP_TITLE = "Privacy-Preserving Hybrid LLM System"
APP_ICON = None
MAX_FILE_SIZE_MB = 10
