"""
Flask Web Interface
Privacy-Preserving Hybrid LLM System - Hardened Unified Dashboard
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from orchestrator import HybridLLMOrchestrator
from evaluation import LLMJudge

# Load env before anything else
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

STATE: dict[str, Any] = {
    "orchestrator": None,
}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/state", methods=["GET"])
def get_state():
    orch = STATE["orchestrator"]
    if not orch:
        return jsonify({"initialized": False})
    
    return jsonify({
        "initialized": orch.initialized,
        "documents_loaded": orch.documents_loaded,
        "loaded_files": orch.get_document_list(),
        "stats": {
            "local_model": orch.local_executor.model if hasattr(orch.local_executor, "model") else "phi3:mini",
            "cloud_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "pipeline": "intent-driven"
        }
    })

@app.route("/api/documents", methods=["GET"])
def get_documents():
    orch = STATE["orchestrator"]
    if not orch:
        return jsonify([])
    return jsonify(orch.get_document_list())

@app.route("/api/initialize", methods=["POST"])
def initialize_api():
    payload = request.get_json(silent=True) or {}
    api_key = payload.get("groq_api_key")
    
    try:
        STATE["orchestrator"] = HybridLLMOrchestrator(groq_api_key=api_key)
        return jsonify({"success": True, "message": "System re-initialized."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/load-documents", methods=["POST"])
def load_documents_api():
    orch = STATE["orchestrator"]
    if not orch:
        return jsonify({"success": False, "error": "Orchestrator not initialized."}), 400
        
    if "documents" not in request.files:
        return jsonify({"success": False, "error": "No files uploaded."}), 400
    
    files = request.files.getlist("documents")
    saved_paths = []
    
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = str(DOCUMENTS_DIR / filename)
            file.save(path)
            saved_paths.append(path)
    
    if not saved_paths:
        return jsonify({"success": False, "error": "No valid documents found."}), 400

    success = orch.load_documents(saved_paths)
    if success:
        return jsonify({
            "success": True, 
            "message": f"Loaded {len(saved_paths)} documents.",
            "loaded_files": orch.get_document_list()
        })
    else:
        return jsonify({"success": False, "error": "Failed to process documents."}), 500

@app.route("/api/query", methods=["POST"])
def query_api():
    orch = STATE["orchestrator"]
    if not orch:
         return jsonify({"success": False, "error": "Orchestrator not initialized."}), 400

    data = request.json or {}
    query_text = data.get("query")
    mode = data.get("mode", "hybrid")
    expert_mode = data.get("expert_mode", False)
    selected_document = data.get("selected_document")
    # domain is no longer used — intent is classified automatically

    if selected_document == "all":
        selected_document = None

    if not query_text:
        return jsonify({"success": False, "error": "No query provided."}), 400

    if mode == "hybrid":
        result = orch.process_query_hybrid(
            query_text, expert_mode=expert_mode, selected_document=selected_document
        )
    else:
        result = orch.process_query_local_only(
            query_text, selected_document=selected_document
        )

    return jsonify(result)

@app.route("/api/compare", methods=["POST"])
def compare_api():
    orch = STATE["orchestrator"]
    if not orch:
         return jsonify({"success": False, "error": "Orchestrator not initialized."}), 400

    data = request.json or {}
    query_text = data.get("query")
    expert_mode = data.get("expert_mode", False)
    selected_document = data.get("selected_document")
    # domain is no longer used — intent is classified automatically

    if selected_document == "all":
        selected_document = None

    if not query_text:
        return jsonify({"success": False, "error": "No query provided."}), 400

    judge = LLMJudge()
    results = orch.compare_approaches(
        query_text, expert_mode=expert_mode, selected_document=selected_document
    )
    
    # Check if any modes failed
    if not results.get("hybrid", {}).get("success") or not results.get("local_only", {}).get("success"):
        error_msg = results.get("hybrid", {}).get("error") or results.get("local_only", {}).get("error") or "Analysis failed."
        return jsonify({"success": False, "error": error_msg}), 400

    with ThreadPoolExecutor(max_workers=3) as executor:
        local_answer = results["local_only"].get("answer", "")
        hybrid_answer = results["hybrid"].get("answer", "")
        local_context = results["local_only"].get("retrieved_context", [])
        hybrid_context = results["hybrid"].get("retrieved_context", [])
        
        # For individual evaluations, we use their respective context
        local_eval_future = executor.submit(judge.evaluate_answer, query_text, local_answer, local_context)
        hybrid_eval_future = executor.submit(judge.evaluate_answer, query_text, hybrid_answer, hybrid_context)
        
        # For comparison, we use a combined context to see who retrieved better/more relevant info
        combined_context = local_context + [c for c in hybrid_context if c not in local_context]
        comparison_future = executor.submit(judge.compare_answers, query_text, local_answer, hybrid_answer, combined_context)
        
        judge_payload = {
            "provider": "ollama",
            "model": judge.judge_model,
            "local_only": local_eval_future.result(),
            "hybrid": hybrid_eval_future.result(),
            "comparison": comparison_future.result(),
        }

    # Reconcile scores
    criteria = ["accuracy", "completeness", "relevance", "coherence", "groundedness"]

    def avg_score(evaluation: dict) -> float:
        scores = []
        for criterion in criteria:
            item = evaluation.get(criterion, {})
            try:
                scores.append(float(item.get("score", 0)))
            except (TypeError, ValueError):
                scores.append(0.0)
        return sum(scores) / max(len(scores), 1)

    local_avg = round(avg_score(judge_payload["local_only"]), 2)
    hybrid_avg = round(avg_score(judge_payload["hybrid"]), 2)

    comp = judge_payload.get("comparison")
    if not isinstance(comp, dict):
        comp = {}
        judge_payload["comparison"] = comp

    comp["answer_a_score"] = local_avg
    comp["answer_b_score"] = hybrid_avg

    # --- Deterministic quality score (objective, unaffected by Mistral lenience) ---
    import re as _re

    def quality_score(text: str) -> float:
        """Blend of depth, vocabulary richness, and non-repetition."""
        if not text:
            return 0.0
        words = text.lower().split()
        word_count = len(words)
        # Depth: normalize word count (cap benefit at 600 words)
        depth = min(word_count / 600, 1.0)
        # Vocabulary diversity: unique words / total words
        vocab = len(set(words)) / max(word_count, 1)
        # Non-repetition: deduped sentences / all sentences
        sents = [s.strip() for s in _re.split(r'[.!?]', text) if s.strip()]
        dedup_ratio = len(set(sents)) / max(len(sents), 1)
        return round(0.5 * depth + 0.3 * vocab + 0.2 * dedup_ratio, 3)

    local_q = quality_score(local_answer)
    hybrid_q = quality_score(hybrid_answer)
    comp["answer_a_quality"] = local_q
    comp["answer_b_quality"] = hybrid_q

    # Primary: trust Mistral's clear side-by-side verdict (A or B)
    mistral_winner = str(comp.get("winner", "Tie")).strip()
    mistral_strength = str(comp.get("preference_strength", "Weak")).strip()

    if mistral_winner in ("A", "B"):
        computed_winner = mistral_winner
        strength = mistral_strength if mistral_strength in ("Weak", "Moderate", "Strong") else "Moderate"
    else:
        # Mistral uncertain — use deterministic quality score
        q_diff = hybrid_q - local_q
        if q_diff > 0.08:
            computed_winner = "B"   # Hybrid clearly better
            strength = "Strong" if q_diff > 0.2 else "Moderate"
        elif q_diff < -0.08:
            computed_winner = "A"   # Local clearly better
            strength = "Strong" if q_diff < -0.2 else "Moderate"
        else:
            computed_winner = "Tie"
            strength = "Weak"

    comp["winner"] = computed_winner
    comp["preference_strength"] = strength
    judge_payload["comparison"] = comp

    return jsonify({
        "success": True,
        "results": results,
        "judge": judge_payload
    })

# --- Auto-Initialization Logic ---
def auto_initialize():
    apiKey = os.getenv("GROQ_API_KEY")
    if apiKey:
        print(f"Auto-initializing orchestrator with GROQ_API_KEY from environment...")
        try:
            STATE["orchestrator"] = HybridLLMOrchestrator(groq_api_key=apiKey)
            print("Auto-initialization successful.")
        except Exception as e:
            print(f"Auto-initialization failed: {e}")
    else:
        print("GROQ_API_KEY not found in .env. System waiting for manual key.")

auto_initialize()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
