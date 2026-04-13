"""
generate_documents.py
=====================
Generates 1000 synthetic privacy-sensitive documents using FREE Gemini API.
Each document comes with auto-generated test queries and reference answers.

Usage:
    python generate_documents.py                    # fresh start
    python generate_documents.py --resume           # resume if interrupted
    python generate_documents.py --n_docs 100       # generate fewer for testing

Output:
    data/synthetic/documents/doc_0001.txt  ... doc_1000.txt
    data/synthetic/manifest.json           (document metadata)
    data/synthetic/queries.json            (all test queries)
"""

import os
import sys
import json
import time
import random
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

# ── Add project root to path ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ── Try importing Gemini ──────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("ERROR: google-generativeai not installed.")
    print("Run: pip install google-generativeai")
    sys.exit(1)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ── Configuration ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OUTPUT_DIR     = PROJECT_ROOT / "data" / "synthetic"
MANIFEST_PATH  = OUTPUT_DIR / "manifest.json"
QUERIES_PATH   = OUTPUT_DIR / "queries.json"
DOCS_DIR       = OUTPUT_DIR / "documents"
SAVE_EVERY     = 10          # save progress every N docs
RATE_LIMIT_RPM = 14          # stay under 15 req/min free limit
SLEEP_BETWEEN  = 60 / RATE_LIMIT_RPM   # ~4.3 seconds between calls


# ── Document domains and types ────────────────────────────────────────────────
DOMAINS = {
    "medical": [
        "patient discharge summary",
        "clinical trial report",
        "lab test results report",
        "radiology report",
        "medication review record",
        "pathology report",
        "surgical operation note",
        "psychiatric assessment",
    ],
    "legal": [
        "employment contract",
        "non-disclosure agreement",
        "GDPR compliance assessment",
        "data processing agreement",
        "privacy audit report",
        "intellectual property agreement",
        "service level agreement",
    ],
    "finance": [
        "quarterly earnings report",
        "risk assessment document",
        "loan application summary",
        "internal audit report",
        "investment portfolio review",
        "insurance claim document",
        "tax assessment report",
    ],
}

# ── Prompt template ───────────────────────────────────────────────────────────
GENERATION_PROMPT = """Generate a realistic {doc_type} in the {domain} domain.

Requirements:
- 350-500 words of actual document content
- Include realistic but COMPLETELY FAKE personal information (fake names, fake IDs, fake dates, fake amounts)
- Include domain-specific numbers and terminology
- Include at least 3 numerical values that require reasoning or calculation to interpret
- Include at least 1 subtle nuance or apparent tension in the data
- Write it like a real professional document with sections

Return ONLY valid JSON with no extra text, no markdown, no code blocks:
{{
  "title": "document title",
  "content": "full document text here, properly formatted",
  "key_facts": ["important fact 1", "important fact 2", "important fact 3"],
  "pii_present": ["types of PII in this document e.g. name, DOB, account number"],
  "queries": [
    {{
      "query": "a simple question whose answer is directly stated in the document",
      "answer": "the correct answer",
      "difficulty": "easy",
      "type": "simple_retrieval"
    }},
    {{
      "query": "a question requiring arithmetic or unit conversion using values from the document",
      "answer": "the correct calculated answer with working",
      "difficulty": "medium",
      "type": "calculation"
    }},
    {{
      "query": "a complex multi-step question requiring reasoning across multiple parts of the document",
      "answer": "the correct reasoned answer",
      "difficulty": "hard",
      "type": "cross_reference"
    }},
    {{
      "query": "a question about a nuance, contradiction, or implication in the document",
      "answer": "the correct analytical answer",
      "difficulty": "hard",
      "type": "synthesis"
    }}
  ]
}}"""


class DocumentGenerator:
    def __init__(self):
        # Check ollama is running
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = [m['name'] for m in r.json()['models']]
            print(f"✓ Ollama running. Available models: {models}")
        except Exception as e:
            print(f"ERROR: Ollama not running. Start it with: ollama serve")
            print(f"Details: {e}")
            sys.exit(1)

        self.model_name = "mistral:latest"
        self.request_count = 0
        self.error_count = 0

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """No rate limit needed — fully local."""
        self.request_count += 1

    def _clean_json(self, text: str) -> str:
        """Strip markdown fences and fix control characters."""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()

        import re
        text = re.sub(r'(?<=[^\\])\n', '\\n', text)
        text = re.sub(r'\r', '', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

        return text

    def generate_one(self, domain: str, doc_type: str, doc_id: int):
        """Generate a single document using local Ollama. Returns dict or None on failure."""
        prompt = GENERATION_PROMPT.format(domain=domain, doc_type=doc_type)
        self._rate_limit()

        for attempt in range(3):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"You are a document generator. Respond with valid JSON only. No markdown, no explanation, no code fences. Escape all newlines inside string values as \\n.\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.8,
                            "num_predict": 2048,
                        }
                    },
                    timeout=180,   # 3 min timeout — mistral can be slow
                )

                if response.status_code != 200:
                    raise Exception(f"Ollama HTTP {response.status_code}: {response.text[:200]}")

                raw = response.json().get("response", "")
                text = self._clean_json(raw)
                data = json.loads(text)

                # Validate required fields
                required = ["title", "content", "queries"]
                for field in required:
                    if field not in data:
                        raise ValueError(f"Missing field: {field}")

                data["doc_id"]   = doc_id
                data["domain"]   = domain
                data["doc_type"] = doc_type
                return data

            except json.JSONDecodeError as e:
                print(f"    [Attempt {attempt+1}] JSON parse error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"    [Attempt {attempt+1}] Error: {e}")
                time.sleep(5)

        self.error_count += 1
        return None

    def save_document(self, doc: dict) -> str:
        """Save document text to file, return file path."""
        doc_id   = doc["doc_id"]
        doc_path = str(DOCS_DIR / f"doc_{doc_id:04d}.txt")

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(f"DOMAIN: {doc['domain'].upper()}\n")
            f.write(f"TYPE: {doc['doc_type'].title()}\n")
            f.write(f"TITLE: {doc['title']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(doc["content"])

        return doc_path

    def _save_progress(self, manifest, queries):
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(QUERIES_PATH, "w") as f:
            json.dump(queries, f, indent=2, ensure_ascii=False)
    def run(self, n_docs: int = 1000, resume: bool = False):
            """Main generation loop."""

            if resume and MANIFEST_PATH.exists() and QUERIES_PATH.exists():
                with open(MANIFEST_PATH) as f:
                    manifest = json.load(f)
                with open(QUERIES_PATH) as f:
                    all_queries = json.load(f)
                start_id = len(manifest)
                print(f"Resuming from document {start_id} ({len(manifest)} already done).")
            else:
                manifest    = []
                all_queries = []
                start_id    = 0

            if start_id >= n_docs:
                print(f"All {n_docs} documents already generated!")
                return

            domain_list = list(DOMAINS.keys())
            remaining   = n_docs - start_id

            print(f"\n{'='*60}")
            print(f"  Document Generator — Mistral (Local Ollama)")
            print(f"{'='*60}")
            print(f"  Target     : {n_docs} documents")
            print(f"  Remaining  : {remaining} documents")
            print(f"  Output dir : {OUTPUT_DIR}")
            print(f"  Tip        : Safe to Ctrl+C and resume later with --resume")
            print(f"{'='*60}\n")

            try:
                for doc_id in range(start_id, n_docs):
                    domain   = random.choice(domain_list)
                    doc_type = random.choice(DOMAINS[domain])

                    print(f"[{doc_id+1:4d}/{n_docs}] {domain:8s} | {doc_type}")

                    doc = self.generate_one(domain, doc_type, doc_id)

                    if doc is None:
                        print(f"         SKIPPED (generation failed after 3 attempts)")
                        continue

                    doc_path = self.save_document(doc)

                    for q in doc.get("queries", []):
                        q["doc_id"]   = doc_id
                        q["doc_path"] = doc_path
                        q["domain"]   = domain
                        q["doc_type"] = doc_type
                        all_queries.append(q)

                    manifest.append({
                        "doc_id"     : doc_id,
                        "path"       : doc_path,
                        "domain"     : domain,
                        "doc_type"   : doc_type,
                        "title"      : doc.get("title", ""),
                        "pii_present": doc.get("pii_present", []),
                        "n_queries"  : len(doc.get("queries", [])),
                    })

                    # Save after every single document
                    self._save_progress(manifest, all_queries)
                    print(f"         Saved. Docs: {len(manifest)}, Queries: {len(all_queries)}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Saving progress...")

            self._save_progress(manifest, all_queries)

            print(f"\n{'='*60}")
            print(f"  Done! Documents: {len(manifest)}, Queries: {len(all_queries)}, Errors: {self.error_count}")
            print(f"{'='*60}\n")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic documents for LLM evaluation"
    )
    parser.add_argument(
        "--n_docs", type=int, default=1000,
        help="Number of documents to generate (default: 1000)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved progress"
    )
    args = parser.parse_args()

    generator = DocumentGenerator()
    generator.run(n_docs=args.n_docs, resume=args.resume)