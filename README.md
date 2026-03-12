# Privacy-Preserving Hybrid LLM System for Secure Document Reasoning

This project implements a secure AI architecture that leverages the high-level reasoning capabilities of cloud models (like Llama 3 via Groq) while keeping sensitive data strictly on local infrastructure (via Ollama).

## 🚀 Overview

The system separates **Reasoning Planning** from **Task Execution**:
1.  **Query Abstraction**: Sensitive data is masked from the user query.
2.  **Cloud Planning**: The masked query is sent to a cloud model to generate a reasoning plan.
3.  **Local Execution**: The reasoning plan is executed locally using sensitive document context retrieved via RAG.

## 🛠️ Tech Stack

| Component | Technology | Reason |
| --- | --- | --- |
| **Frontend** | Streamlit | Rapid development of interactive web UI. |
| **Orchestrator** | LangChain | Industry standard for LLM workflow management. |
| **Local LLM** | Ollama (Mistral/Llama3) | High-performance local inference engine. |
| **Cloud LLM** | Groq (Llama 3 70B) | High-speed cloud model for complex reasoning tasks. |
| **Vector DB** | FAISS | Efficient local similarity search for RAG. |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Small and fast local embeddings for indexing. |

## 📂 Folder Structure

```text
Hybrid-LLM/
├── app/
│   ├── main.py                 # Streamlit Web Entry point
│   └── modules/
│       ├── cloud_planner.py     # Groq API integration
│       ├── local_executor.py    # Ollama integration
│       ├── orchestrator.py      # Main workflow logic
│       ├── query_abstraction.py # PII masking & abstraction
│       └── vector_store.py      # FAISS & Document processing
├── data/
│   ├── documents/               # Temporary storage for PDFs
│   └── vectors/                 # Local FAISS index storage
├── .env.example                 # Environment variable template
├── requirements.txt             # Python dependencies
└── README.md                    # Setup and usage guide
```

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running.
- [Groq API Key](https://console.groq.com/).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Local LLM
```bash
ollama serve
ollama pull mistral
```

### 4. Configure Environment
Copy `.env.example` to `.env` and add your keys:
```bash
cp .env.example .env
# Edit .env and replace YOUR_API_KEY_HERE
```

### 5. Run the Application
From the project root directory, run:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/app
streamlit run app/main.py
```
*Note: Adding the `app` directory to your PYTHONPATH ensures that the modules are correctly discovered.*

## 🧪 Testing Instructions

1.  **Upload**: Provide a PDF containing some private information (e.g., a dummy medical case).
2.  **Query**: Ask "What are the key patient risks and what plan should be followed?".
3.  **Security Check**: Open the "Show Privacy & Reasoning Details" expander to verify that your query was masked before being sent to the cloud.
4.  **Baseline Check**: Compare the results in the "Local-Only Baseline" tab to see how the hybrid approach provides better structured reasoning.

## 🔮 Future Improvements
- [ ] **Advanced PII Masking**: Use Spacy or Presidio for better name/entity detection.
- [ ] **Multi-Agent Collaboration**: Use LangGraph for more complex reasoning loops.
- [ ] **Data Encryption**: Encrypt the local FAISS index on disk.
