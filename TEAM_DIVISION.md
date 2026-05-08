# Team Division & Responsibility Plan

This document outlines the division of the **Privacy-Preserving Hybrid LLM System** into 3 distinct parts for the team members. It is updated to reflect the latest codebase structure, including the Flask UI and the new `pipeline` architecture.

---

## Part 1: Local LLM, RAG Pipeline & Vector DB
**Assigned to:** Nandini Patil (23510001)

### Core Responsibilities
- **Data Ingestion:** Parsing and processing uploaded documents (PDF, DOCX, TXT).
- **Vector Database:** Managing the FAISS index and generating embeddings for semantic search.
- **Local Execution:** Managing the local Ollama instance (e.g., `mistral`, `phi3:mini`) and ensuring that queries are executed safely on local private data.

### Relevant Files & Folders
- `src/document_processor.py`: Handles file uploading, chunking, and text extraction.
- `src/vector_store.py`: FAISS-based vector database and Sentence-BERT embedding logic.
- `src/local_llm.py`: Ollama API integration and local execution engine.
- `src/pipeline/metadata_extractor.py`: Extracts structured metadata from local chunks to aid the cloud planner.

---

## Part 2: Cloud Integration, Intent Routing & Planning Module
**Assigned to:** Bhumika Mane (23510002)

### Core Responsibilities
- **Cloud Reasoning Planner:** Interfacing with the Groq API (e.g., `llama-3.3-70b-versatile`) to generate step-by-step reasoning plans.
- **Intent-based Routing:** Classifying the user's intent (summary, reasoning, extraction, comparison) and routing it appropriately without sending private data.
- **Orchestration & Validation:** Stitching together the pipeline, executing step-by-step plans, and validating that the final answer is grounded in facts.

### Relevant Files & Folders
- `src/orchestrator.py`: The main controller coordinating all Hybrid modes and the local vs. cloud fallback logic.
- `src/cloud_planner.py`: Generates reasoning plans via the Groq API.
- `src/pipeline/router.py`: Routes queries based on their complexity and intent.
- `src/pipeline/intent_classifier.py`: Determines what kind of operation the query requires.
- `src/pipeline/executor.py`: Executes the reasoning steps.
- `src/pipeline/validator.py`: Validates the final generated answer against the context.

---

## Part 3: Query Abstraction, UI Frontend & Evaluation Framework
**Assigned to:** Trusha Kulkarni (23510025)

### Core Responsibilities
- **Privacy Filter (Abstraction):** Removing sensitive information (PII, credentials) from queries before they are sent to the cloud.
- **Web Interface:** Maintaining the unified Flask + HTML/JS/CSS web dashboard.
- **Evaluation & Metrics:** Running LLM-as-a-judge comparisons, tracking ROUGE scores, privacy scores, and latency.

### Relevant Files & Folders
- `src/app.py`: The Flask application serving the backend API routes and rendering the frontend.
- `src/static/` & `src/templates/`: The frontend UI (HTML, CSS, JS) for the unified dashboard.
- `src/query_abstractor.py`: Uses regex and heuristics to mask PII from user queries.
- `src/evaluation.py`: LLM Judge implementation for comparing local, hybrid, and cloud answers.
- `src/metrics.py`: Tracks performance metrics like latency, latency differences, etc.
- `test_demo.py` & `run_official_eval.py`: Evaluation scripts.

---

## How to Collaborate Effectively
1. **API Contracts:** Since the `orchestrator.py` (Part 2) depends on `local_llm.py` (Part 1) and `query_abstractor.py` (Part 3), ensure the function signatures (inputs/outputs) remain consistent. If you need to change a function's parameters, notify the team.
2. **Independent Testing:** 
   - **Part 1** can test document chunking and local responses offline.
   - **Part 2** can mock local RAG responses and test the cloud reasoning logic.
   - **Part 3** can test the UI by connecting it to mock data endpoints or testing the query abstractor on sample text.
