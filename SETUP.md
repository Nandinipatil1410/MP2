Ran command: `python src/app.py`
Ran command: `clear`
Viewed eval_benchmarks.json:63-63

This is a great question to prepare for. Here's how to think about it and what to say:

---

## The Honest, Confident Answer

### "Yes, AI tools helped us. Here's what WE did:"

The distinction judges look for is **understanding vs. copy-paste**. You can say:

> *"We used AI as a coding assistant — the same way engineers use Stack Overflow or documentation. But the system design, the problem identification, the architectural decisions, and all the debugging were done by us."*

Then point to **specific decisions you made:**

---

## What You Actually Designed & Decided

**1. The core problem you solved:**
> We identified that sending private documents (patient records, financial reports) to cloud LLMs is a privacy risk. We designed a system where sensitive data never leaves the local machine — only anonymized, abstracted queries go to the cloud.

**2. The architecture you chose:**
> We chose a **hybrid approach** — local LLM for retrieval and execution, cloud LLM only for high-level reasoning on masked data. We designed the intent classification router to decide which path a query takes (summary → extraction → reasoning → comparison).

**3. The bugs you found and fixed:**
> We discovered and fixed a **placeholder namespace collision bug** — the query abstractor used `[PERSON_0]` but the context masker independently used `[NAME_1]` for the same person, causing the cloud to think they were two different people. We traced this through 3 layers of code and fixed it.

**4. The privacy-preserving pipeline:**
> We built the PII masking system — entities like names, dates, IDs are replaced with placeholders before any data touches the cloud API. The cloud never sees real names.

**5. The trade-off you evaluated and chose:**
> We consciously chose **not to delete documents on restart** (Option 1) and instead persist embeddings to disk (Option 2) — we reasoned through the privacy vs. UX trade-off.

---

## If They Push: "But the LLM wrote the code"

> *"The LLM wrote syntax. We wrote the system. The same way a civil engineer uses AutoCAD — the software draws the lines, but the engineer decides what to build and why."*

> *"We debugged every component. We identified that the FAISS index wasn't being saved. We found the masking inconsistency. We decided the routing logic. If we didn't understand it, it wouldn't work."*

---

## What to Be Ready to Explain (Know These Cold)

| Topic | Be ready to explain |
|-------|-------------------|
| **RAG** | Why retrieval-augmented generation? What's the alternative? |
| **FAISS** | What is it, why not a database? What does `IndexFlatL2` do? |
| **Intent classification** | How does the router decide between summary/extraction/reasoning? |
| **PII masking** | What patterns are detected? What's a placeholder? |
| **Hybrid vs Local-only** | What's the latency/privacy tradeoff? |
| **Ollama** | Why local? What model? How is it different from the cloud model? |

If you can explain those confidently, no judge will question your ownership of the project.