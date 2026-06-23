# SCOTUS Legal Aid

**AI-powered retrieval and synthesis over landmark U.S. Supreme Court opinions**

![Python](https://img.shields.io/badge/python-3.12-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.58-FF4B4B)
![Anthropic](https://img.shields.io/badge/Claude-Haiku-D97757)
![License](https://img.shields.io/badge/license-MIT-green)

**Live demo:** [YOUR_STREAMLIT_URL](YOUR_STREAMLIT_URL)
**Repo:** [github.com/scdubay/ai-SCOTUS](https://github.com/scdubay/ai-SCOTUS)

## What it does

SCOTUS Legal Aid lets you ask plain-English questions about six landmark U.S. Supreme Court cases and get back precise, citation-grounded answers — not generic legal advice, but answers synthesized directly from the actual opinion text. It's built for anyone curious about how these cases were actually decided: students, writers, or anyone who wants a faster way into dense legal opinions without wading through the full text themselves.

It is **not** a comprehensive legal database, does not cover current or pending cases, and is not a substitute for legal advice.

## Architecture

```
User Query
  ↓
Question Classifier (Haiku) → META / OVERVIEW / RESEARCH
  ↓
Case Router (lexical + semantic hybrid)
  ↓
FAISS Vector Search (BGE-small-en-v1.5, 384-dim)
  ↓
Case-Scoped Retrieval (read_whole or within_case)
  ↓
Answer Generation (Claude Haiku)
  ↓
Faithfulness Guard
  ↓
Response
```

META and OVERVIEW questions (about the app itself, or general case summaries) short-circuit straight to answer generation — they skip case routing, retrieval, and the faithfulness guard entirely, since they're not answered from retrieved opinion text. Only RESEARCH questions go through the full pipeline above. A topic/prompt-injection filter also runs before classification (not pictured), rejecting off-topic or adversarial input before any model call is made.

## Key engineering decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | `BAAI/bge-small-en-v1.5` (384-dim) | Chosen over larger 768-dim options (mpnet, bge-base) specifically so the FAISS index stays within Streamlit Cloud's free-tier memory budget as the corpus scales toward 100–500 cases / tens of thousands of chunks — at that scale, a 768-dim index alone would consume most of the ~1GB ceiling before the model or app even loads. |
| LLM for generation | Claude Haiku (`claude-haiku-4-5`), via an `BACKEND` switch | Low cost and fast latency for a portfolio-scale demo. The `BACKEND` env var also supports a local Ollama backend for development without incurring API cost. |
| Routing strategy | Hybrid lexical + semantic | Cheap lexical token-overlap against case titles is tried first and only trusted above a confidence threshold; semantic (embedding) similarity is the fallback, and is itself gated by a minimum-distance threshold so an unrelated question doesn't get confidently mis-pinned to whatever case happens to be least-far in the embedding space. |
| Retrieval mode | Case-scoped: read-whole vs. within-case | Once a case is resolved, small cases are handed to the model in full; large cases fall back to in-case (not corpus-wide) chunk retrieval. This was a deliberate fix for cross-case contamination that occurred when retrieval competed across the whole corpus. |
| Faithfulness guard | Heuristic regex-based check, high-precision not high-recall | Flags justices or cases named in the answer that don't appear anywhere in the retrieved context, and catches role misattribution (e.g., calling a dissent a "concurrence"). Deliberately tuned to avoid false alarms over exhaustive coverage — a tripwire alongside the retrieval metrics, not a substitute for a capable model. |
| Question classification | Single LLM call (Haiku), not a keyword list | An earlier keyword-based classifier reliably missed natural phrasings like "what is your purpose?" A single fast classification call generalizes to phrasing the original list never anticipated, at the cost of one extra cheap call per question. |
| Deployment architecture | Standalone Streamlit app | `app.py` calls the retrieval/generation pipeline directly (cached via `st.cache_resource`) so it runs as a single process with no separate backend — required for Streamlit Cloud, which can't host a second FastAPI process alongside it. `api.py` remains in the repo as a FastAPI reference implementation of the same pipeline, but isn't used by the deployed app. |

## Evaluation results

| Metric | Score | Notes |
|---|---|---|
| Routing accuracy | 18/18 (100%) | 18-question expanded eval set; hybrid lexical+semantic routing |
| Mean MRR | 0.670 | Same 18-question set; improved from 0.632 measured under the prior embedding model |
| Context term coverage | 85% | Same 18-question set; down from a measured 90% under the prior embedding model, within the accepted 10% tolerance for the embedding-model switch |
| Faithfulness flags | 1/12 | 12-question batch eval (`query_demo_clean.py --batch`); one pre-existing, known edge case |
| Answer term rate | 85.42% | Same 12-question batch eval |

## Tech stack

- **Frontend:** Streamlit
- **LLM:** Anthropic Claude Haiku
- **Embeddings:** BAAI/bge-small-en-v1.5
- **Vector store:** FAISS
- **Pipeline:** LangChain Community
- **Data source:** CourtListener API
- **Deployment:** Streamlit Cloud

## Local setup

```bash
git clone https://github.com/scdubay/ai-SCOTUS.git
cd ai-SCOTUS
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
streamlit run app.py
```

## Project structure

| File | Description |
|---|---|
| `app.py` | Standalone Streamlit app. Calls the pipeline directly — no separate backend process required. The deployed entry point. |
| `api.py` | FastAPI wrapper around the same pipeline (rate limiting, topic filtering, cost cap). Reference architecture; not used by the deployed app. |
| `query_demo_clean.py` | Core retrieval/generation utilities: vector store config, embeddings, reranking, prompt construction, and the Ollama/Anthropic backend switch. |
| `case_store.py` | Case-scoped retrieval: resolves which case a question is about (lexical + semantic hybrid), then reads that case whole or retrieves within it. |
| `faithfulness.py` | Heuristic post-generation check for hallucinated authorities and opinion-role misattribution. |

## Roadmap

- Expand corpus to 100+ cases
- Thematic cross-case queries
- Orchestration layer with intent detection
- Self-hosted open source LLM on Azure VM

## About the builder

**Stephen Dubay** — Azure cloud infrastructure engineer repositioning toward AI/ML engineering.

Built to demonstrate practical AI engineering skills: RAG pipeline design, vector search, LLM integration, faithfulness checking, and cloud deployment.

[LinkedIn](https://www.linkedin.com/in/stephen-dubay-8a0ba5202) | [GitHub](https://github.com/scdubay/ai-SCOTUS)
