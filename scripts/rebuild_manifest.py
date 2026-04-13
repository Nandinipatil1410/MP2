"""
rebuild_manifest.py
====================
Rebuilds manifest.json and queries.json from existing document files.
Run this once if manifest.json is empty but documents exist.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
DOCS_DIR      = SYNTHETIC_DIR / "documents"
MANIFEST_PATH = SYNTHETIC_DIR / "manifest.json"
QUERIES_PATH  = SYNTHETIC_DIR / "queries.json"

manifest    = []
all_queries = []

doc_files = sorted(DOCS_DIR.glob("doc_*.txt"))
print(f"Found {len(doc_files)} document files. Rebuilding manifest...")

for doc_file in doc_files:
    text = doc_file.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Parse header lines written by save_document()
    domain   = ""
    doc_type = ""
    title    = ""
    for line in lines[:4]:
        if line.startswith("DOMAIN:"):
            domain = line.replace("DOMAIN:", "").strip().lower()
        elif line.startswith("TYPE:"):
            doc_type = line.replace("TYPE:", "").strip().lower()
        elif line.startswith("TITLE:"):
            title = line.replace("TITLE:", "").strip()

    # Extract doc_id from filename
    doc_id = int(doc_file.stem.replace("doc_", ""))

    manifest.append({
        "doc_id"     : doc_id,
        "path"       : str(doc_file),
        "domain"     : domain,
        "doc_type"   : doc_type,
        "title"      : title,
        "pii_present": [],
        "n_queries"  : 0,
    })

    print(f"  [{doc_id:4d}] {domain:10s} | {title[:50]}")

# Save manifest
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"\nManifest rebuilt: {len(manifest)} documents")
print(f"NOTE: queries.json will be empty — evaluation will run without reference answers.")
print(f"      This is fine; the judge will score based on query + answer alone.")

# Save empty queries list so run_evaluation.py doesn't crash
# We'll generate queries on the fly from document content
queries = []
for entry in manifest:
    doc_file = Path(entry["path"])
    text = doc_file.read_text(encoding="utf-8")
    # Skip header (first 4 lines)
    content = "\n".join(text.splitlines()[4:]).strip()

    # Create one generic query per document per difficulty
    base = f"Based on the document titled '{entry['title']}'"
    queries += [
        {
            "doc_id"   : entry["doc_id"],
            "doc_path" : entry["path"],
            "domain"   : entry["domain"],
            "doc_type" : entry["doc_type"],
            "query"    : f"What are the key facts and findings in this {entry['doc_type']}?",
            "answer"   : "",
            "difficulty": "easy",
            "type"     : "simple_retrieval",
        },
        {
            "doc_id"   : entry["doc_id"],
            "doc_path" : entry["path"],
            "domain"   : entry["domain"],
            "doc_type" : entry["doc_type"],
            "query"    : f"What numerical values or calculations are present in this {entry['doc_type']} and what do they indicate?",
            "answer"   : "",
            "difficulty": "medium",
            "type"     : "calculation",
        },
        {
            "doc_id"   : entry["doc_id"],
            "doc_path" : entry["path"],
            "domain"   : entry["domain"],
            "doc_type" : entry["doc_type"],
            "query"    : f"What are the risks, contradictions, or implications present in this {entry['doc_type']}?",
            "answer"   : "",
            "difficulty": "hard",
            "type"     : "synthesis",
        },
    ]

with open(QUERIES_PATH, "w") as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)

print(f"Queries generated : {len(queries)} ({len(manifest)} docs × 3 queries each)")
print(f"\nNext step: python run_evaluation.py --n_queries 20")