"""
Flask Web Interface
Privacy-Preserving Hybrid LLM System
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from orchestrator import HybridLLMOrchestrator
from evaluation import LLMJudge

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


STATE: dict[str, Any] = {
    "orchestrator": None,
    "documents_loaded": False,
    "loaded_files": [],
    "groq_api_key": "",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_files(files) -> list[str]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for uploaded_file in files:
        if not uploaded_file or not uploaded_file.filename:
            continue
        if not allowed_file(uploaded_file.filename):
            continue

        filename = secure_filename(uploaded_file.filename)
        file_path = DOCUMENTS_DIR / filename
        uploaded_file.save(file_path)
        saved_paths.append(str(file_path))

    return saved_paths


def require_orchestrator():
    if STATE["orchestrator"] is None:
        return False, jsonify({"success": False, "error": "Initialize the system first."}), 400
    return True, None, None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def get_state():
    stats = None
    if STATE["orchestrator"] is not None:
        try:
            stats = STATE["orchestrator"].get_system_stats()
        except Exception:
            stats = None

    return jsonify(
        {
            "success": True,
            "initialized": STATE["orchestrator"] is not None,
            "documents_loaded": STATE["documents_loaded"],
            "loaded_files": STATE["loaded_files"],
            "stats": stats,
        }
    )


@app.post("/api/initialize")
def initialize():
    payload = request.get_json(silent=True) or {}
    groq_api_key = (payload.get("groq_api_key") or "").strip()
    if groq_api_key.lower() in {"your_groq_api_key_here", "your_api_key_here", "changeme"}:
        groq_api_key = ""

    STATE["groq_api_key"] = groq_api_key
    STATE["orchestrator"] = HybridLLMOrchestrator(groq_api_key=(groq_api_key or None))
    STATE["documents_loaded"] = False
    STATE["loaded_files"] = []

    return jsonify(
        {
            "success": True,
            "message": "System initialized successfully.",
            "initialized": True,
            "documents_loaded": False,
        }
    )


@app.post("/api/load-documents")
def load_documents():
    ok, response, status = require_orchestrator()
    if not ok:
        return response, status

    files = request.files.getlist("documents")
    if not files:
        return jsonify({"success": False, "error": "Upload documents first."}), 400

    saved_paths = save_uploaded_files(files)
    if not saved_paths:
        return jsonify({"success": False, "error": "No valid files were uploaded."}), 400

    try:
        STATE["orchestrator"].load_documents(saved_paths)
        STATE["documents_loaded"] = True
        STATE["loaded_files"] = [Path(path).name for path in saved_paths]
        return jsonify(
            {
                "success": True,
                "message": f"{len(saved_paths)} document(s) loaded into the secure workspace.",
                "documents_loaded": True,
                "loaded_files": STATE["loaded_files"],
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": f"Document loading failed: {exc}"}), 500


@app.post("/api/query")
def run_query():
    ok, response, status = require_orchestrator()
    if not ok:
        return response, status

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    mode = payload.get("mode") or "hybrid"

    if not query:
        return jsonify({"success": False, "error": "Enter a question before running the analysis."}), 400

    try:
        if mode == "local_only":
            result = STATE["orchestrator"].process_query_local_only(query)
        else:
            result = STATE["orchestrator"].process_query_hybrid(query)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Query execution failed: {exc}"}), 500

    return jsonify(result)


@app.post("/api/compare")
def compare():
    ok, response, status = require_orchestrator()
    if not ok:
        return response, status

    if not STATE["documents_loaded"]:
        return jsonify({"success": False, "error": "Load documents before generating a comparison report."}), 400

    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()

    if not query:
        return jsonify({"success": False, "error": "Enter a benchmarking question first."}), 400

    try:
        results = STATE["orchestrator"].compare_approaches(query)
    except Exception as exc:
        return jsonify({"success": False, "error": f"Comparison failed: {exc}"}), 500

    hybrid_score = (
        results["hybrid"].get("privacy_score", 0) * 0.6
        + results["hybrid"].get("completeness_score", 0) * 0.4
    )
    local_score = (
        results["local_only"].get("privacy_score", 0) * 0.8
        + results["local_only"].get("completeness_score", 0) * 0.2
    )

    recommendation = "Hybrid Strategy" if hybrid_score > local_score else "Local-Only Strategy"

    # LLM-as-a-judge (Gemini only). Groq must remain only for hybrid planning/cloud.
    try:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise RuntimeError("Gemini judge is not configured. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")

        reference_docs = STATE["orchestrator"].vector_store.search(query, top_k=3)
        judge = LLMJudge(allow_groq_fallback=False)
        local_answer = results.get("local_only", {}).get("answer", "")
        hybrid_answer = results.get("hybrid", {}).get("answer", "")

        judge_payload = {
            "provider": judge.judge_provider,
            "model": judge.judge_model,
            "local_only": judge.evaluate_answer(query, local_answer, reference_docs),
            "hybrid": judge.evaluate_answer(query, hybrid_answer, reference_docs),
            "comparison": judge.compare_answers(query, local_answer, hybrid_answer, reference_docs),
        }

        # Reconcile any inconsistency between per-criterion scores and the comparison winner/scores.
        # We want the dashboard to be internally consistent and not claim a "strong" win when
        # criterion scores indicate a tie.
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

        comp = judge_payload.get("comparison") or {}
        comp["answer_a_score"] = local_avg
        comp["answer_b_score"] = hybrid_avg

        tolerance = 0.3
        if abs(hybrid_avg - local_avg) <= tolerance:
            computed_winner = "Tie"
            strength = "Weak"
        elif hybrid_avg > local_avg:
            computed_winner = "B"
            strength = "Moderate" if (hybrid_avg - local_avg) < 1.0 else "Strong"
        else:
            computed_winner = "A"
            strength = "Moderate" if (local_avg - hybrid_avg) < 1.0 else "Strong"

        original_winner = comp.get("winner")
        comp["winner"] = computed_winner
        comp["preference_strength"] = strength

        if original_winner and original_winner != computed_winner:
            diffs = comp.get("key_differences")
            if not isinstance(diffs, list):
                diffs = []
            diffs.insert(
                0,
                "The direct-comparison verdict disagreed with the criterion scores; the dashboard uses the criterion-average for consistency.",
            )
            comp["key_differences"] = diffs

        judge_payload["comparison"] = comp
    except Exception as exc:
        judge_payload = {
            "provider": "unavailable",
            "model": None,
            "error": str(exc),
        }

    return jsonify(
        {
            "success": True,
            "results": results,
            "recommendation": recommendation,
            "judge": judge_payload,
        }
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8501)
