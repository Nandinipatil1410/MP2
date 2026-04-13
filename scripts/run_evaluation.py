"""
run_evaluation.py
=================
Runs the full evaluation pipeline:
  1. Loads all 1000 generated documents into vector store
  2. Samples queries across difficulty levels
  3. Runs each query through all 3 pipelines (Hybrid, Local, Cloud)
  4. Scores each answer using FREE Gemini as LLM judge
  5. Saves results + generates final summary report

Usage:
    python run_evaluation.py                        # run full evaluation
    python run_evaluation.py --n_queries 50         # quick test with 50 queries
    python run_evaluation.py --resume               # resume interrupted run
    python run_evaluation.py --report_only          # just regenerate report from saved results

Output:
    results/evaluation_results.json     (full per-query results)
    results/evaluation_summary.json     (aggregated statistics)
    results/evaluation_report.txt       (human readable report)
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Add src/ to path ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# ── Try importing dependencies ────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("ERROR: google-generativeai not installed. Run: pip install google-generativeai")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas")
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────
SYNTHETIC_DIR   = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH   = SYNTHETIC_DIR / "manifest.json"
QUERIES_PATH    = SYNTHETIC_DIR / "queries.json"
RESULTS_DIR     = PROJECT_ROOT / "results"
RESULTS_PATH    = RESULTS_DIR  / "evaluation_results.json"
SUMMARY_PATH    = RESULTS_DIR  / "evaluation_summary.json"
REPORT_PATH     = RESULTS_DIR  / "evaluation_report.txt"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemini judge config ───────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

JUDGE_SLEEP     = 5      # seconds between judge calls (free tier)
PIPELINE_SLEEP  = 2      # seconds between pipeline runs


# ── LLM Judge ─────────────────────────────────────────────────────────────────
class OllamaJudge:
    CRITERIA = ["accuracy", "completeness", "relevance", "coherence", "groundedness"]

    def __init__(self):
        self.model_name = "mistral:latest"
        self.request_count = 0
        print("Using Ollama (mistral) as judge — fully local, no rate limits")

    def _rate_limit(self):
        self.request_count += 1

    def score(self, query, answer, reference="", mode=""):
        prompt = f"""You are a STRICT evaluator. Most answers should score 4-7 out of 10.
    Only give 9-10 for truly excellent answers. Give 1-3 for poor answers.

    Query: {query[:200]}
    Answer being evaluated: {answer[:400]}

    Score strictly on each criterion from 0 to 10:
    - accuracy: Is it factually correct?
    - completeness: Does it fully answer the query?
    - relevance: Is it relevant and focused?
    - coherence: Is it well structured?
    - groundedness: Is it based on real document content?

    Respond with ONLY this JSON and nothing else:
    {{"accuracy":5,"completeness":5,"relevance":5,"coherence":5,"groundedness":5,"reasoning":"one sentence"}}"""

        for attempt in range(3):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 150,
                        }
                    },
                    timeout=120,
                )
                raw = response.json().get("response", "").strip()

                # Clean response
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0]
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0]

                # Find JSON object in response
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if not match:
                    raise ValueError("No JSON found in response")

                scores = json.loads(match.group())

                # Validate scores are in range
                for c in self.CRITERIA:
                    scores[c] = max(0, min(10, int(scores.get(c, 5))))

                scores["overall"] = round(
                    sum(scores.get(c, 0) for c in self.CRITERIA) / len(self.CRITERIA), 2
                )
                return scores

            except Exception as e:
                print(f"    [Judge attempt {attempt+1}] Error: {e}")
                time.sleep(3)

        return {c: 5 for c in self.CRITERIA} | {"overall": 5.0, "error": "judge_failed"}

# ── Pipeline runner ───────────────────────────────────────────────────────────
class PipelineRunner:
    """Loads the orchestrator and runs queries through all three pipelines."""

    def __init__(self):
        print("Initializing orchestrator...")
        try:
            from orchestrator import HybridLLMOrchestrator
            self.orchestrator = HybridLLMOrchestrator(groq_api_key=GROQ_API_KEY)
            print("Orchestrator ready.")
        except Exception as e:
            print(f"ERROR initializing orchestrator: {e}")
            sys.exit(1)

    def load_documents(self, doc_paths: list):
        """Load a list of document paths into the vector store."""
        print(f"\nLoading {len(doc_paths)} documents into vector store...")
        if not doc_paths:
            print("ERROR: No documents found. Did generate_documents.py complete?")
            sys.exit(1)
        self.orchestrator.load_documents(doc_paths)
        print("Vector store ready.\n")

    def run_query(self, query: str) -> dict:
        """Run one query through all 3 pipelines. Returns results dict."""
        try:
            return self.orchestrator.compare_approaches(query)
        except Exception as e:
            print(f"    Pipeline error: {e}")
            empty = {"answer": f"ERROR: {e}", "latency": 0,
                     "privacy_score": 0, "reasoning_plan": []}
            return {
                "hybrid"    : {**empty, "mode": "hybrid"},
                "local_only": {**empty, "mode": "local_only"},
                "cloud_only": {**empty, "mode": "cloud_only"},
            }
        
# ── Query sampler ─────────────────────────────────────────────────────────────
def sample_queries(all_queries: list, n_total: int) -> list:
    """
    Sample n_total queries with balanced distribution across:
    - difficulty: easy / medium / hard
    - type: simple_retrieval / calculation / cross_reference / synthesis
    - domain: medical / legal / finance
    """
    # Group by difficulty
    by_difficulty = defaultdict(list)
    for q in all_queries:
        by_difficulty[q.get("difficulty", "medium")].append(q)

    targets = {
        "easy"  : int(n_total * 0.25),   # 25% easy
        "medium": int(n_total * 0.35),   # 35% medium
        "hard"  : int(n_total * 0.40),   # 40% hard — most interesting
    }

    sampled = []
    for difficulty, count in targets.items():
        pool   = by_difficulty.get(difficulty, [])
        chosen = random.sample(pool, min(count, len(pool)))
        sampled.extend(chosen)

    # Shuffle final list
    random.shuffle(sampled)
    return sampled[:n_total]


# ── Main evaluator ────────────────────────────────────────────────────────────
class Evaluator:
    def __init__(self):
        self.judge  = OllamaJudge()      # was GeminiJudge
        self.runner = PipelineRunner()

    def run(self, n_queries: int = 200, resume: bool = False):

        # ── Load queries ──────────────────────────────────────────────────────
        if not QUERIES_PATH.exists():
            print(f"ERROR: {QUERIES_PATH} not found.")
            print("Run generate_documents.py first.")
            sys.exit(1)

        with open(QUERIES_PATH) as f:
            all_queries = json.load(f)

        with open(MANIFEST_PATH) as f:
            manifest = json.load(f)

        # ADD THIS to debug
        print(f"DEBUG: manifest path = {MANIFEST_PATH}")
        print(f"DEBUG: manifest entries = {len(manifest)}")
        print(f"DEBUG: queries path = {QUERIES_PATH}")

        doc_paths = [m["path"] for m in manifest if Path(m["path"]).exists()]

        # ADD THIS too
        print(f"DEBUG: valid doc paths found = {len(doc_paths)}")

        print(f"\nLoaded {len(all_queries)} queries from "
              f"{len(manifest)} documents.\n")

        # ── Load documents into vector store ──────────────────────────────────
        doc_paths = [m["path"] for m in manifest if Path(m["path"]).exists()]
        self.runner.load_documents(doc_paths)

        # ── Sample queries ────────────────────────────────────────────────────
        test_queries = sample_queries(all_queries, n_queries)
        print(f"Sampled {len(test_queries)} queries for evaluation.\n")

        # ── Load existing results if resuming ─────────────────────────────────
        if resume and RESULTS_PATH.exists():
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done_ids = {r["query_index"] for r in results}
            print(f"Resuming: {len(results)} queries already done.\n")
        else:
            results  = []
            done_ids = set()

        # ── Run evaluation loop ───────────────────────────────────────────────
        print(f"{'='*65}")
        print(f"  Starting evaluation — {len(test_queries)} queries")
        print(f"  Judge    : Gemini 1.5 Flash (free)")
        print(f"  Pipelines: Hybrid | Local-Only | Cloud-Only")
        est_mins = len(test_queries) * (JUDGE_SLEEP * 3 + PIPELINE_SLEEP + 5) / 60
        print(f"  Est. time: ~{est_mins:.0f} minutes")
        print(f"  Results  : {RESULTS_PATH}")
        print(f"{'='*65}\n")

        for i, test_q in enumerate(test_queries):

            if i in done_ids:
                continue

            query      = test_q["query"]
            reference  = test_q.get("answer", "")
            difficulty = test_q.get("difficulty", "unknown")
            q_type     = test_q.get("type", "unknown")
            domain     = test_q.get("domain", "unknown")

            print(f"[{i+1:3d}/{len(test_queries)}] "
                  f"{difficulty:6s} | {q_type:20s} | {domain:8s}")
            print(f"         Q: {query[:70]}...")

            # ── Run all 3 pipelines ───────────────────────────────────────────
            pipeline_results = self.runner.run_query(query)
            time.sleep(PIPELINE_SLEEP)

            # ── Judge each answer ─────────────────────────────────────────────
            judge_scores = {}
            winner_scores = {}

            for mode in ["hybrid", "local_only", "cloud_only"]:
                answer = pipeline_results[mode].get("answer", "")
                scores = self.judge.score(
                    query=query,
                    answer=answer,
                    reference=reference,
                    mode=mode
                )
                judge_scores[mode]  = scores
                winner_scores[mode] = scores["overall"]

                print(f"         {mode:12s}: {scores['overall']:.1f}/10")

            # Determine winner
            winner = max(winner_scores, key=winner_scores.get)
            print(f"         Winner: {winner}\n")

            # ── Store result ──────────────────────────────────────────────────
            record = {
                "query_index" : i,
                "query"       : query,
                "reference"   : reference,
                "difficulty"  : difficulty,
                "query_type"  : q_type,
                "domain"      : domain,
                "doc_id"      : test_q.get("doc_id"),
                "winner"      : winner,
                "judge_scores": judge_scores,
                "latency"     : {
                    mode: pipeline_results[mode].get("latency", 0)
                    for mode in ["hybrid", "local_only", "cloud_only"]
                },
                "privacy_score": {
                    "hybrid"    : pipeline_results["hybrid"].get("privacy_score", 0),
                    "local_only": 1.0,
                    "cloud_only": 0.0,
                },
                "reasoning_steps": len(
                    pipeline_results["hybrid"].get("reasoning_plan", [])
                ),
                "answers": {
                    mode: pipeline_results[mode].get("answer", "")
                    for mode in ["hybrid", "local_only", "cloud_only"]
                },
            }
            results.append(record)

            # Save every 5 queries
            if (i + 1) % 5 == 0:
                with open(RESULTS_PATH, "w") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

        # Final save
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation complete. Generating report...")
        self.generate_report(results)

    def generate_report(self, results: list = None):
        """Generate summary JSON and human-readable report."""

        if results is None:
            if not RESULTS_PATH.exists():
                print("No results found. Run evaluation first.")
                return
            with open(RESULTS_PATH) as f:
                results = json.load(f)

        if not results:
            print("No results to report.")
            return

        modes = ["hybrid", "local_only", "cloud_only"]

        # ── Aggregate stats ───────────────────────────────────────────────────
        summary = {
            "generated_at"    : datetime.now().isoformat(),
            "total_queries"   : len(results),
            "overall"         : {},
            "by_difficulty"   : {},
            "by_query_type"   : {},
            "by_domain"       : {},
            "win_counts"      : {m: 0 for m in modes},
            "avg_latency"     : {m: 0 for m in modes},
            "avg_privacy"     : {m: 0 for m in modes},
            "avg_judge_score" : {m: 0 for m in modes},
        }

        # Win counts
        for r in results:
            summary["win_counts"][r["winner"]] += 1

        # Overall averages
        for mode in modes:
            scores   = [r["judge_scores"][mode]["overall"] for r in results]
            latency  = [r["latency"][mode] for r in results]
            privacy  = [r["privacy_score"][mode] for r in results]
            summary["avg_judge_score"][mode] = round(
                sum(scores) / len(scores), 2)
            summary["avg_latency"][mode] = round(
                sum(latency) / len(latency), 2)
            summary["avg_privacy"][mode] = round(
                sum(privacy) / len(privacy), 2)

        # By difficulty
        for diff in ["easy", "medium", "hard"]:
            subset = [r for r in results if r["difficulty"] == diff]
            if not subset:
                continue
            summary["by_difficulty"][diff] = {
                "count": len(subset),
                **{f"avg_{m}": round(
                    sum(r["judge_scores"][m]["overall"] for r in subset)
                    / len(subset), 2)
                   for m in modes},
                "win_counts": {
                    m: sum(1 for r in subset if r["winner"] == m)
                    for m in modes
                }
            }

        # By query type
        q_types = set(r["query_type"] for r in results)
        for qt in q_types:
            subset = [r for r in results if r["query_type"] == qt]
            summary["by_query_type"][qt] = {
                "count": len(subset),
                **{f"avg_{m}": round(
                    sum(r["judge_scores"][m]["overall"] for r in subset)
                    / len(subset), 2)
                   for m in modes},
                "win_counts": {
                    m: sum(1 for r in subset if r["winner"] == m)
                    for m in modes
                }
            }

        # By domain
        domains = set(r["domain"] for r in results)
        for domain in domains:
            subset = [r for r in results if r["domain"] == domain]
            summary["by_domain"][domain] = {
                "count": len(subset),
                **{f"avg_{m}": round(
                    sum(r["judge_scores"][m]["overall"] for r in subset)
                    / len(subset), 2)
                   for m in modes},
            }

        # Save summary JSON
        with open(SUMMARY_PATH, "w") as f:
            json.dump(summary, f, indent=2)

        # ── Human-readable report ─────────────────────────────────────────────
        lines = []
        def add(line=""): lines.append(line)

        add("=" * 65)
        add("  EVALUATION REPORT — Privacy-Preserving Hybrid LLM System")
        add(f"  Generated: {summary['generated_at'][:19]}")
        add("=" * 65)

        add()
        add(f"  Corpus size     : 1,000 synthetic documents")
        add(f"  Domains         : Medical | Legal | Finance")
        add(f"  Queries tested  : {summary['total_queries']}")
        add(f"  Judge model     : Gemini 1.5 Flash")
        add(f"  Local model     : Phi-3 Mini (via Ollama)")
        add(f"  Cloud planner   : Groq LLaMA-3.3-70B")

        add()
        add("─" * 65)
        add("  OVERALL RESULTS")
        add("─" * 65)
        add(f"  {'Mode':<15} {'Avg Score':>10} {'Avg Latency':>12} "
            f"{'Privacy':>9} {'Wins':>6} {'Win%':>6}")
        add(f"  {'─'*15} {'─'*10} {'─'*12} {'─'*9} {'─'*6} {'─'*6}")

        total = summary["total_queries"]
        for mode in modes:
            wins = summary["win_counts"][mode]
            add(f"  {mode.replace('_',' ').title():<15} "
                f"{summary['avg_judge_score'][mode]:>10.2f} "
                f"{summary['avg_latency'][mode]:>11.2f}s "
                f"{summary['avg_privacy'][mode]:>9.2f} "
                f"{wins:>6} "
                f"{wins/total*100:>5.1f}%")

        add()
        add("─" * 65)
        add("  RESULTS BY DIFFICULTY")
        add("─" * 65)
        add(f"  {'Difficulty':<10} {'Count':>6} "
            f"{'Hybrid':>8} {'Local':>8} {'Cloud':>8}  Winner")
        add(f"  {'─'*10} {'─'*6} {'─'*8} {'─'*8} {'─'*8}  {'─'*12}")

        for diff in ["easy", "medium", "hard"]:
            stats = summary["by_difficulty"].get(diff, {})
            if not stats:
                continue
            wc    = stats["win_counts"]
            top   = max(wc, key=wc.get).replace("_", " ").title()
            add(f"  {diff.title():<10} {stats['count']:>6} "
                f"{stats['avg_hybrid']:>8.2f} "
                f"{stats['avg_local_only']:>8.2f} "
                f"{stats['avg_cloud_only']:>8.2f}  {top}")

        add()
        add("─" * 65)
        add("  RESULTS BY QUERY TYPE")
        add("─" * 65)
        add(f"  {'Type':<22} {'Count':>6} "
            f"{'Hybrid':>8} {'Local':>8} {'Cloud':>8}")
        add(f"  {'─'*22} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

        for qt, stats in sorted(summary["by_query_type"].items()):
            add(f"  {qt.replace('_',' ').title():<22} {stats['count']:>6} "
                f"{stats['avg_hybrid']:>8.2f} "
                f"{stats['avg_local_only']:>8.2f} "
                f"{stats['avg_cloud_only']:>8.2f}")

        add()
        add("─" * 65)
        add("  RESULTS BY DOMAIN")
        add("─" * 65)
        add(f"  {'Domain':<12} {'Count':>6} "
            f"{'Hybrid':>8} {'Local':>8} {'Cloud':>8}")
        add(f"  {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*8}")

        for domain, stats in sorted(summary["by_domain"].items()):
            add(f"  {domain.title():<12} {stats['count']:>6} "
                f"{stats['avg_hybrid']:>8.2f} "
                f"{stats['avg_local_only']:>8.2f} "
                f"{stats['avg_cloud_only']:>8.2f}")

        add()
        add("─" * 65)
        add("  KEY FINDINGS")
        add("─" * 65)

        # Auto-generate key findings
        hard = summary["by_difficulty"].get("hard", {})
        easy = summary["by_difficulty"].get("easy", {})
        if hard:
            gap = hard["avg_hybrid"] - hard["avg_local_only"]
            add(f"  1. On HARD queries, Hybrid outperforms Local-Only by "
                f"{gap:+.2f} points")
        if easy:
            gap = easy["avg_hybrid"] - easy["avg_local_only"]
            add(f"  2. On EASY queries, gap is only {gap:+.2f} points "
                f"(local handles simple retrieval well)")

        add(f"  3. Hybrid privacy score: "
            f"{summary['avg_privacy']['hybrid']:.2f} "
            f"vs Cloud-Only: 0.00 (100% data exposure)")

        top_mode = max(modes,
                       key=lambda m: summary["avg_judge_score"][m])
        add(f"  4. Best overall performer: "
            f"{top_mode.replace('_',' ').title()} "
            f"({summary['avg_judge_score'][top_mode]:.2f}/10)")

        add()
        add("=" * 65)

        report_text = "\n".join(lines)

        # Save report
        with open(REPORT_PATH, "w") as f:
            f.write(report_text)

        # Print to console
        print("\n" + report_text)
        print(f"\nReport saved to: {REPORT_PATH}")
        print(f"Summary JSON  : {SUMMARY_PATH}")
        print(f"Full results  : {RESULTS_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run full evaluation of Hybrid vs Local vs Cloud LLM pipelines"
    )
    parser.add_argument(
        "--n_queries", type=int, default=200,
        help="Number of queries to evaluate (default: 200)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved results"
    )
    parser.add_argument(
        "--report_only", action="store_true",
        help="Skip evaluation, just regenerate report from saved results"
    )
    args = parser.parse_args()

    evaluator = Evaluator()

    if args.report_only:
        evaluator.generate_report()
    else:
        evaluator.run(
            n_queries=args.n_queries,
            resume=args.resume
        )

