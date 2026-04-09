# Installation and Setup Guide
# Privacy-Preserving Hybrid LLM System

## Prerequisites

### 1. Python 3.8 or higher
```bash
python --version  # Should be 3.8+
```

### 2. Install Ollama (for local LLM)

**Linux/Mac:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from https://ollama.com/download

**Verify installation:**
```bash
ollama --version
```

### 3. Pull a local model
```bash
ollama pull mistral
# or
ollama pull llama3
```

## Installation Steps

### Step 1: Clone/Download the Project
```bash
cd privacy_llm_project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate it:
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables
```bash
# Copy the example env file
cp .env.example .env

# Edit .env file and add your Groq API key (optional)
# You can get a free key from: https://console.groq.com
```

### Step 5: Create Data Directories
```bash
mkdir -p data/documents
mkdir -p data/vector_db
```

## Running the Application

### Option 1: Web Interface (Recommended)
```bash
python src/app.py
```
Then open browser to: http://localhost:8501

### Option 2: Quick Test
```bash
python test_demo.py --quick
```

### Option 3: Full Demo with Sample Documents
```bash
python test_demo.py --full
```

### Option 4: Automated Evaluation (LLM-as-Judge)
```bash
# Uses GEMINI_API_KEY/GOOGLE_API_KEY (recommended and used for the dashboard verdict).
# If Gemini is not configured, the CLI demo falls back to heuristic judging.
python test_demo.py --judge
```

## Usage Guide

### 1. Using the Web Interface

**Step 1:** Initialize the system
- Enter the Groq API key if needed
- Click "Initialize Pipeline"

**Step 2:** Upload documents
- Select PDF, TXT, or DOCX files
- Click "Load Documents"

**Step 3:** Ask questions
- Go to "Query Workspace"
- Enter your question
- Select mode (Hybrid recommended)
- Click "Run Analysis"

**Step 4:** Compare approaches
- Go to "Model Comparison"
- Enter a question
- Click "Generate Comparative Report"
- View the comparison results

### 2. Using Python API

```python
from src.orchestrator import HybridLLMOrchestrator

# Initialize
orchestrator = HybridLLMOrchestrator(groq_api_key="your_key")

# Load documents
orchestrator.load_documents(["path/to/doc1.pdf", "path/to/doc2.txt"])

# Ask question - Hybrid mode
result = orchestrator.process_query_hybrid("What is the main topic?")
print(result['answer'])

# Ask question - Local only mode
result = orchestrator.process_query_local_only("Summarize the document")
print(result['answer'])

# Compare all approaches
results = orchestrator.compare_approaches("What are the key points?")
for mode, result in results.items():
    print(f"{mode}: {result['answer']}")
```

## Architecture Overview

```
┌─────────────┐
│    User     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────┐
│   Flask Web UI          │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   Orchestrator          │
│  (Main Controller)      │
└──┬──┬──┬──┬────────────┘
   │  │  │  │
   │  │  │  └──────────────────┐
   │  │  │                     │
   ▼  ▼  ▼                     ▼
┌──────┐ ┌──────────┐   ┌────────────┐
│ Docs │ │  Vector  │   │   Query    │
│ Proc │ │  Store   │   │ Abstractor │
└──────┘ └──────────┘   └────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │Cloud Planner │
                        │(Groq API)    │
                        └──────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │Local Executor│
                        │(Ollama)      │
                        └──────────────┘
```

## Troubleshooting

### Issue: "Ollama is not running"
**Solution:**
```bash
# Start Ollama
ollama serve

# In another terminal, pull a model
ollama pull mistral
```

### Issue: "No module named 'groq'"
**Solution:**
```bash
pip install groq
```

### Issue: "FAISS installation failed"
**Solution:**
```bash
# Try CPU version
pip install faiss-cpu

# Or if you have GPU
pip install faiss-gpu
```

### Issue: "Model not found in Ollama"
**Solution:**
```bash
# List available models
ollama list

# Pull the required model
ollama pull mistral
```

### Issue: "Groq API key error"
**Solution:**
- Get free API key from https://console.groq.com
- Add to .env file or enter in web interface
- System works without it (uses fallback mode)

## System Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.10+
- 8GB RAM
- 5GB free disk space
- GPU (optional, for faster local inference)

## Features

✅ Local document processing (PDF, DOCX, TXT)
✅ Privacy-preserving query abstraction  
✅ Cloud-based reasoning planning (optional)
✅ Local execution on private documents
✅ FAISS vector database for retrieval
✅ Comparison of different approaches
✅ Web-based user interface
✅ Evaluation metrics

## Support

For issues, please check:
1. All dependencies installed correctly
2. Ollama is running (ollama serve)
3. Model is downloaded (ollama pull mistral)
4. Python version is 3.8+

## Next Steps

After setup:
1. Run the demo: `python test_demo.py --full`
2. Try the web interface: `python src/app.py`
3. Upload your own documents
4. Experiment with different queries
5. Compare the three modes

## Notes

- The system works WITHOUT a Groq API key (uses fallback planning)
- All documents stay local - never sent to cloud
- Only abstracted queries (no private data) sent to cloud
- Hybrid mode recommended for best balance
