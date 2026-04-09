# Privacy-Preserving Hybrid LLM System

## Project Overview
A privacy-preserving AI system that improves reasoning quality of local models without transmitting sensitive data to the cloud.

## Architecture
```
User Query + Documents
    ↓
Local Processing (Encryption + Vector Indexing)
    ↓
Query Abstraction (Privacy Filter)
    ↓
Cloud Reasoning Planner (Logic Only - No Data)
    ↓
Local Execution Engine (LLM + Documents)
    ↓
Secure Response
```

## Directory Structure
```
privacy_llm_project/
├── requirements.txt
├── README.md
├── .env.example
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── student1_rag/          # Student 1: Local LLM + RAG + Vector DB
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── vector_store.py
│   │   └── local_llm.py
│   ├── student2_cloud/         # Student 2: Cloud Integration + Planning
│   │   ├── __init__.py
│   │   ├── cloud_planner.py
│   │   └── reasoning_executor.py
│   ├── student3_ui/            # Student 3: Query Abstraction + UI + Evaluation
│   │   ├── __init__.py
│   │   ├── query_abstractor.py
│   │   ├── evaluation.py
│   │   └── streamlit_app.py
│   └── orchestrator/
│       ├── __init__.py
│       └── langchain_orchestrator.py
├── data/
│   ├── documents/              # User uploaded documents
│   └── vector_db/              # FAISS index storage
└── tests/
    ├── test_rag.py
    ├── test_planner.py
    └── test_abstraction.py
```

## Setup Instructions

### 1. Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull mistral
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
LOCAL_MODEL=mistral
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_DB_PATH=./data/vector_db
```

### 4. Run the Application
```bash
python src/app.py
```

## Work Distribution

### Student 1 (Nandini Patil): Local LLM + RAG Pipeline + Vector DB
- Document parsing (PDF, DOCX, TXT)
- FAISS vector database setup
- Embedding generation
- Document retrieval system
- Local LLM integration with Ollama

### Student 2 (Bhumika Mane): Cloud Integration + Planning Module
- Groq API integration
- Reasoning plan generation
- Plan execution engine
- API error handling and retry logic

### Student 3 (Trusha Kulkarni): Query Abstraction + UI + Evaluation
- Query classification
- Sensitive information removal
- Streamlit web interface
- Performance metrics (ROUGE, BERT-Score)
- Comparative evaluation framework

## Testing

```bash
# Test RAG pipeline
python -m pytest tests/test_rag.py

# Test cloud planner
python -m pytest tests/test_planner.py

# Test query abstraction
python -m pytest tests/test_abstraction.py
```

## Automated Evaluation (LLM-as-Judge)

```bash
# Uses GEMINI_API_KEY/GOOGLE_API_KEY (recommended and used for dashboard judge verdict).
# If Gemini is not configured, the CLI demo falls back to a heuristic judge.
python test_demo.py --judge
```

## Evaluation Metrics

- **Accuracy**: Answer correctness
- **Privacy Score**: Sensitive information leakage
- **Latency**: Response time
- **ROUGE Score**: Answer quality
- **Comparison**: Local-only vs Hybrid vs Cloud-only

## Features

✅ Local document processing
✅ Privacy-preserving query abstraction
✅ Cloud-based reasoning planning
✅ Hybrid execution pipeline
✅ Performance comparison dashboard
✅ Streamlit web interface

## License
MIT License - Educational Project
