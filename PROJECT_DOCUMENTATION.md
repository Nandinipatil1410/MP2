# Privacy-Preserving Hybrid LLM System
## Mini Project - Final Documentation

### Team Information
- **Nandini Patil** (23510001): Local LLM + RAG + Vector DB
- **Bhumika Mane** (23510002): Cloud Integration + Planning Module
- **Trusha Kulkarni** (23510025): Query Abstraction + UI + Evaluation

### Guide
- **Dr. N. L. Gavankar**

---

## Project Overview

This project implements a **Privacy-Preserving Hybrid LLM Architecture** that enables secure document reasoning without exposing sensitive data to cloud services.

### Problem Statement
Organizations handling sensitive documents (healthcare, banking, legal) cannot use cloud-based LLMs due to privacy concerns, but local models lack strong reasoning capabilities.

### Solution
A hybrid architecture that:
1. Keeps all sensitive documents local
2. Sends only abstracted queries (no private data) to cloud
3. Gets reasoning plans from powerful cloud LLMs
4. Executes plans locally on private documents

---

## System Architecture

### Components

1. **Document Processor** (`document_processor.py`)
   - Handles PDF, DOCX, TXT files
   - Chunks documents for efficient retrieval
   - Processes and cleans text

2. **Vector Store** (`vector_store.py`)
   - FAISS-based vector database
   - Sentence-BERT embeddings
   - Semantic search capability

3. **Query Abstractor** (`query_abstractor.py`)
   - Removes sensitive information (PII, credentials)
   - Maintains query intent
   - Privacy scoring system

4. **Cloud Planner** (`cloud_planner.py`)
   - Groq API integration
   - Generates step-by-step reasoning plans
   - No access to private data

5. **Local Executor** (`local_llm.py`)
   - Ollama integration
   - Executes plans on private documents
   - Local-only processing

6. **Orchestrator** (`orchestrator.py`)
   - Main controller
   - Coordinates all components
   - Manages three modes: Hybrid, Local-only, Cloud-only

7. **Web Interface** (`app.py`)
   - Streamlit-based UI
   - Document upload
   - Query interface
   - Comparison dashboard

8. **Evaluation** (`evaluation.py`)
   - ROUGE scores
   - Privacy metrics
   - Latency comparison
   - Performance analysis

---

## Key Features

### 1. Privacy Protection
- ✅ Documents never leave local machine
- ✅ Automatic PII detection and removal
- ✅ Query abstraction before cloud communication
- ✅ Privacy scoring system

### 2. Three Operating Modes

**Hybrid Mode (Recommended)**
- Cloud generates reasoning plan (no data)
- Local LLM executes on private documents
- Best balance of privacy and performance

**Local-Only Mode**
- Everything runs locally
- Maximum privacy
- No cloud dependency

**Cloud-Only Mode**
- For comparison purposes
- Shows privacy risk of full cloud approach

### 3. RAG (Retrieval-Augmented Generation)
- Semantic search using FAISS
- Sentence-BERT embeddings
- Top-K document retrieval

---

## Technical Stack

### Core Technologies
- **Language:** Python 3.8+
- **LLM Framework:** LangChain
- **Local LLM:** Ollama (Mistral/Llama)
- **Cloud LLM:** Groq API (llama-3.3-70b-versatile)
- **Vector DB:** FAISS
- **Embeddings:** Sentence-Transformers
- **UI:** Streamlit

### Key Libraries
```
langchain
ollama
groq
faiss-cpu
sentence-transformers
pypdf
python-docx
streamlit
rouge-score
```

---

## Implementation Details

### Document Processing Pipeline
```python
1. Upload document (PDF/DOCX/TXT)
2. Extract text
3. Chunk into 500-word segments with 50-word overlap
4. Generate embeddings (384-dimensional vectors)
5. Store in FAISS index
```

### Query Processing Pipeline (Hybrid Mode)
```python
1. User enters query
2. Query abstraction (remove PII)
3. Send abstracted query to cloud
4. Cloud generates reasoning plan
5. Retrieve relevant documents locally
6. Execute plan on private documents
7. Return answer to user
```

### Privacy Abstraction Example
```
Original:  "What is John Doe's salary at john@company.com?"
Abstracted: "What is the salary information for [PERSON] at [EMAIL]?"
Sent to cloud: "What is the salary information?"
```

---

## Evaluation Metrics

### 1. Privacy Score (0-1)
- 1.0 = Perfect privacy (local-only)
- 0.8-0.9 = High privacy (hybrid with abstraction)
- 0.0 = No privacy (cloud-only)

### 2. Latency
- Time to process query and generate answer
- Compared across three modes

### 3. Answer Quality
- ROUGE scores (when reference answers available)
- Relevance heuristics
- Document coverage

### 4. Comparison Metrics
| Metric | Local-Only | Hybrid | Cloud-Only |
|--------|-----------|--------|------------|
| Privacy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Speed | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Quality | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Cost | Free | Low | High |

---

## Usage Examples

### Example 1: Healthcare Document
```python
# Upload: patient_records.pdf
# Query: "What treatments were prescribed for diabetes patients?"

# Hybrid Mode:
# - Abstracts: "What treatments for medical condition?"
# - Cloud: Generates reasoning plan
# - Local: Executes on actual patient records
# - Answer: Detailed treatment list (stays local)
```

### Example 2: Financial Documents
```python
# Upload: financial_reports.pdf
# Query: "Compare Q1 vs Q2 revenue"

# Hybrid Mode:
# - Abstracts: "Compare quarterly revenue data"
# - Cloud: Reasoning plan for comparison
# - Local: Calculates from actual numbers
# - Answer: Comparison with specific figures
```

---

## Results & Achievements

### What We Built
✅ Complete working system with all components
✅ Web interface for easy interaction
✅ Support for multiple document formats
✅ Three operating modes for comparison
✅ Privacy-preserving architecture
✅ Evaluation framework

### Performance
- **Privacy Protection:** Successful PII removal
- **Query Accuracy:** Maintains answer quality
- **Latency:** Sub-10 second responses (depends on doc size)
- **Scalability:** Handles 100+ documents

### Key Insights
1. Hybrid approach viable for privacy-sensitive domains
2. Query abstraction effective for PII protection
3. Local LLMs sufficient with good reasoning plans
4. RAG essential for document grounding

---

## Future Enhancements

### Short-term
- [ ] Add more document formats (Excel, PPT)
- [ ] Implement caching for faster responses
- [ ] Add user authentication
- [ ] Support for multi-language documents

### Long-term
- [ ] Homomorphic encryption for enhanced privacy
- [ ] Fine-tune local models for specific domains
- [ ] Multi-user support with access control
- [ ] Integration with enterprise systems
- [ ] Advanced evaluation metrics

---

## Challenges Faced & Solutions

### Challenge 1: Ollama Installation
**Problem:** Students had issues installing Ollama
**Solution:** Created detailed setup guide with alternatives

### Challenge 2: API Rate Limits
**Problem:** Groq free tier has limits
**Solution:** Implemented fallback planning mode

### Challenge 3: Large Documents
**Problem:** PDF processing slow for large files
**Solution:** Implemented chunking and parallel processing

### Challenge 4: Privacy Leakage
**Problem:** Some PII in abstracted queries
**Solution:** Enhanced pattern matching and keyword detection

---

## Deployment

### Local Deployment
```bash
1. Install dependencies: pip install -r requirements.txt
2. Start Ollama: ollama serve
3. Run app: streamlit run src/app.py
4. Access: http://localhost:8501
```

### Cloud Deployment (Optional)
- Can deploy Streamlit app to Streamlit Cloud
- Ollama needs to run on server with model files
- Requires proper security configuration

---

## Conclusion

This project successfully demonstrates a **privacy-preserving approach to LLM-based document reasoning**. By separating the reasoning planning (cloud) from execution (local), we achieve a practical balance between:

- **Privacy:** Sensitive data never leaves local machine
- **Performance:** Leverages powerful cloud models for planning
- **Practicality:** Easy to use with web interface

The system is ready for:
- ✅ Educational demonstrations
- ✅ Prototype deployments
- ✅ Research publications
- ✅ Startup development

### SDG Alignment
- **SDG 4 (Education):** Secure AI for learning
- **SDG 9 (Innovation):** Privacy-preserving infrastructure

---

## References

1. Bae et al. (2025). PPMI: Privacy-Preserving LLM Interaction. arXiv:2506.17336
2. Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP
3. LangChain Documentation: https://docs.langchain.com
4. Ollama Documentation: https://ollama.com/docs

---

## Project Files Structure
```
privacy_llm_project/
├── README.md                 # Project overview
├── SETUP.md                  # Installation guide
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
├── test_demo.py              # Demo script
├── src/
│   ├── document_processor.py # PDF/DOCX/TXT processing
│   ├── vector_store.py       # FAISS vector database
│   ├── query_abstractor.py   # Privacy protection
│   ├── cloud_planner.py      # Cloud reasoning
│   ├── local_llm.py          # Local execution
│   ├── orchestrator.py       # Main controller
│   ├── app.py                # Streamlit UI
│   └── evaluation.py         # Metrics
├── data/
│   ├── documents/            # Uploaded documents
│   └── vector_db/            # FAISS indices
└── config/
    └── settings.py           # Configuration
```

---

**Project Status:** ✅ Complete & Working
**Demonstration Ready:** Yes
**Publication Ready:** Yes
**Deployment Ready:** Yes (with proper setup)

---

*Developed by Team from Walchand College of Engineering, Sangli*
*Under the guidance of Dr. N. L. Gavankar*
*Academic Year 2025-26*
