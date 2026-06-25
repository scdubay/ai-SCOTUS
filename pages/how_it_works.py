"""
pages/how_it_works.py

Engineering retrospective for SCOTUS Legal Aid. Pure content — no pipeline
imports, no Anthropic/FAISS calls. Structured as a portfolio piece showing
engineering judgment, not just documentation.
"""

import streamlit as st

st.set_page_config(page_title="How it works", page_icon="📘")

# ---------------------------------------------------------------------------
# 1. Header
# ---------------------------------------------------------------------------

st.title("How it works — engineering notes")
st.subheader("The decisions behind this system, and why they were made")
st.write(
    "Built as a portfolio piece to demonstrate practical AI engineering "
    "judgment, not just working code."
)
st.divider()

# ---------------------------------------------------------------------------
# 2. The problem
# ---------------------------------------------------------------------------

st.markdown("## The problem")
st.write(
    "Legal research is structurally different from general Q&A. "
    "Supreme Court opinions run 50–200+ pages with dense citation networks, "
    "and keyword search fails completely — the words that matter most "
    "(the constitutional test applied, the reasoning, the holding) rarely "
    "match the words in the question. "
    "The same case name can refer to different decisions across different years: "
    "Brown v. Board of Education appears twice in this corpus — the 1954 merits "
    "decision and the 1955 implementation decision — and a system that conflates "
    "them gives wrong answers confidently. "
    "A hallucinated legal authority isn't just wrong: citing a case that doesn't "
    "exist or misattributing a holding to the wrong justice is specifically harmful "
    "in a legal context in ways that a hallucinated restaurant review is not. "
    "And scale compounds all of this — a system that works for 6 cases must be "
    "architected to work for 500."
)
st.divider()

# ---------------------------------------------------------------------------
# 3. Architecture
# ---------------------------------------------------------------------------

st.markdown("## Architecture")

left, center, right = st.columns([1, 3, 1])
with center:
    _STAGES = [
        ("User Query", None),
        ("Topic Filter", "decision: why heuristic not ML"),
        ("Question Classifier (Haiku)", "META / OVERVIEW / COMPARATIVE / RESEARCH\ndecision: why LLM-native not keyword lists"),
        ("Case Router", "lexical + semantic hybrid\ndecision: why hybrid, why not pure embedding"),
        ("FAISS Retrieval", "BGE-small-en-v1.5, case-scoped\ndecision: why BGE-small, why case-scoped not global"),
        ("Answer Generation", "Claude Haiku\ndecision: why Haiku not Opus"),
        ("Faithfulness Guard", "flags hallucinated authorities\ndecision: why this exists in legal context"),
        ("Response", None),
    ]

    for i, (stage_name, stage_desc) in enumerate(_STAGES):
        with st.container(border=True):
            st.markdown(f"**{stage_name}**")
            if stage_desc:
                st.caption(stage_desc)
        if i < len(_STAGES) - 1:
            st.markdown("↓")

    st.info(
        "META and OVERVIEW questions short-circuit straight to generation, "
        "skipping case routing, retrieval, and the faithfulness guard — "
        "they are not answered from retrieved opinion text. "
        "COMPARATIVE queries use FAISS-first retrieval across the full corpus "
        "rather than a single-case scope."
    )

st.divider()

# ---------------------------------------------------------------------------
# 4. Six engineering decisions
# ---------------------------------------------------------------------------

st.markdown("## Six engineering decisions")
st.caption(
    "Each decision is framed as: PROBLEM → OPTIONS CONSIDERED → CHOICE → OUTCOME"
)

with st.expander("Decision 1: The same-name routing problem"):
    st.markdown("**PROBLEM**")
    st.write(
        "Brown v. Board of Education appears twice in this corpus — the 1954 "
        "merits decision (Brown I) and the 1955 implementation decision (Brown II). "
        "A lexical scorer that counts title token matches will always route "
        "\"What did Brown I hold?\" to Brown II — because Brown II's title has "
        "fewer tokens, so the same hit count produces a higher proportional score. "
        "This is a structural failure, not an edge case."
    )
    st.markdown("**OPTIONS CONSIDERED**")
    st.markdown(
        "- Hardcode a special case for Brown\n"
        "- Add a year-suffix penalty to the lexical scorer\n"
        "- Use only embedding similarity for disambiguation"
    )
    st.markdown("**CHOICE**")
    st.write(
        "Built `_ordinal_ranks()` using real `date_filed` metadata from each chunk "
        "to assign chronological ordinals (I, II, III...) to any same-named sibling "
        "decisions. When a question mentions \"Brown I\" or \"Brown II,\" the ordinal "
        "resolves against actual decision dates, not title string parsing."
    )
    st.markdown("**OUTCOME**")
    st.write(
        "Generalizes to any future same-name sibling pair automatically — no "
        "hardcoding required. Routing accuracy: 18/18 on the eval set, including "
        "Brown I/II disambiguation."
    )

with st.expander("Decision 2: Embedding model selection for scale"):
    st.markdown("**PROBLEM**")
    st.write(
        "Choosing an embedding model for a 6-case corpus is easy. Choosing one that "
        "still fits Streamlit Cloud's ~1GB free-tier memory at 500 cases / 200,000 "
        "chunks is a different constraint entirely."
    )
    st.markdown("**OPTIONS CONSIDERED**")
    st.markdown(
        "- `nomic-embed-text` (existing, Ollama-based, 768-dim — can't run on cloud)\n"
        "- `all-MiniLM-L6-v2` (384-dim, ~80MB, fast but weakest retrieval quality)\n"
        "- `all-mpnet-base-v2` (768-dim, ~420MB — good quality but consumes most of "
        "the memory budget at scale)\n"
        "- `BAAI/bge-small-en-v1.5` (384-dim, ~130MB, retrieval-specific training)"
    )
    st.markdown("**CHOICE**")
    st.write(
        "BGE-small. At 200K chunks, a 768-dim FAISS index alone consumes ~614MB before "
        "the model or app loads. 384-dim cuts that to ~307MB, leaving headroom for the "
        "model and application. BGE-small benchmarks above MiniLM on retrieval tasks "
        "despite the same dimension count because it was trained with contrastive "
        "retrieval objectives, not general sentence similarity."
    )
    st.markdown("**OUTCOME**")
    st.write(
        "MRR improved from 0.632 (nomic baseline) to 0.670 after the switch. "
        "Index fits comfortably at current scale with headroom for 10x growth."
    )

with st.expander("Decision 3: Case-scoped retrieval vs. global search"):
    st.markdown("**PROBLEM**")
    st.write(
        "Early testing used global retrieval — similarity search across all chunks "
        "in the corpus. Miranda chunks would surface in answers about Marbury. "
        "Gideon context would bleed into Meyer answers. The more cases added, "
        "the worse this got."
    )
    st.markdown("**OPTIONS CONSIDERED**")
    st.markdown(
        "- Global retrieval with reranking\n"
        "- Metadata filtering by case at search time\n"
        "- Case-scoped retrieval with a separate routing step"
    )
    st.markdown("**CHOICE**")
    st.write(
        "Two-stage retrieval. First, route the question to the most relevant case "
        "using a hybrid lexical + semantic scorer. Then search only within that case's "
        "chunks. Lexical matching handles exact citation strings (\"Miranda v. Arizona\") "
        "that embedding similarity would treat as just another phrase. Semantic "
        "similarity handles paraphrases and conceptual questions."
    )
    st.markdown("**OUTCOME**")
    st.write(
        "Cross-case contamination eliminated. Context term coverage: 85–94% depending "
        "on case size. Routing accuracy held at 100% through 15 cases."
    )

with st.expander("Decision 4: Why Haiku and not a larger model"):
    st.markdown("**PROBLEM**")
    st.write(
        "Larger models produce better answers on open-ended tasks. But this is a "
        "grounded RAG system — the retrieval pipeline delivers the relevant text, "
        "and the model synthesizes it. The question is whether a larger model "
        "actually improves synthesis quality when the context is already well-retrieved."
    )
    st.markdown("**OPTIONS CONSIDERED**")
    st.markdown(
        "- Claude Haiku (fast, cheap, good instruction following)\n"
        "- Claude Sonnet (better reasoning, 4x cost)\n"
        "- Claude Opus (frontier quality, high latency, 15x cost)"
    )
    st.markdown("**CHOICE**")
    st.write(
        "Haiku for retrieval-grounded queries. Sonnet reserved for COMPARATIVE "
        "queries that require synthesis across multiple cases simultaneously — "
        "a genuinely harder reasoning task where the additional capability justifies "
        "the cost."
    )
    st.markdown("**OUTCOME**")
    st.write(
        "Haiku achieves 85%+ answer term rate on the eval set. Faithfulness flags "
        "are driven by retrieval gaps, not model quality — switching to Opus would "
        "not reduce them. Average response time: 2–4 seconds."
    )

with st.expander("Decision 5: LLM-native classification over keyword lists"):
    st.markdown("**PROBLEM**")
    st.write(
        "The first classifier used keyword lists to route questions to META, OVERVIEW, "
        "or RESEARCH handlers. It reliably missed natural phrasing variations: "
        "\"what is your purpose?\" routed to RESEARCH instead of META. Every missed "
        "case required adding more keywords — an unbounded maintenance problem."
    )
    st.markdown("**OPTIONS CONSIDERED**")
    st.markdown(
        "- Expand the keyword list\n"
        "- Add regex patterns\n"
        "- Use a small embedding classifier\n"
        "- Single LLM call for classification"
    )
    st.markdown("**CHOICE**")
    st.write(
        "Single fast Haiku call with a carefully engineered system prompt. The prompt "
        "defines category boundaries with positive and negative examples and explicit "
        "tie-break rules. Tested against 12 boundary cases including known failure modes."
    )
    st.markdown("**OUTCOME**")
    st.write(
        "12/12 on the boundary test set. Handles arbitrary phrasing variations without "
        "code changes. Adds ~200ms and ~$0.0001 per query — acceptable cost for "
        "eliminating a maintenance burden."
    )

with st.expander("Decision 6: FAISS-first comparative retrieval"):
    st.markdown("**PROBLEM**")
    st.write(
        "The first implementation of cross-case synthesis asked Haiku to identify "
        "relevant cases from a list, then retrieved chunks from those cases. This "
        "worked at 15 cases but would degrade at 200+ — asking a model to pick "
        "relevant cases from a 200-item list is a harder task than it was designed "
        "for, and longer opinions had a structural advantage because more chunks per "
        "case meant higher raw hit counts in semantic search results."
    )
    st.markdown("**OPTIONS CONSIDERED**")
    st.markdown(
        "- Keep the Haiku selector, add more context\n"
        "- Use metadata filtering (era, topics) to pre-filter\n"
        "- Switch to direct FAISS search"
    )
    st.markdown("**CHOICE**")
    st.write(
        "FAISS-first retrieval. Search the full corpus semantically (k=50), group "
        "results by case, normalize hit counts by each case's total chunk count "
        "(removing size bias), add an explicit title-match bonus for cases named in "
        "the question. No model call in the selection step."
    )
    st.markdown("**OUTCOME**")
    st.write(
        "Weeks v. United States (34 chunks) now surfaces alongside Mapp v. Ohio "
        "(larger opinion) when both are relevant. Scales to 200+ cases with no code "
        "changes. Removes one API call per comparative query."
    )

st.divider()

# ---------------------------------------------------------------------------
# 5. Evaluation methodology
# ---------------------------------------------------------------------------

st.markdown("## Evaluation methodology")
st.write(
    "Eval questions are human-written, not auto-generated. "
    "Auto-generated questions tend to paraphrase the source text directly, "
    "which artificially inflates retrieval scores — they measure how well "
    "the system retrieves text it has already seen, not whether it can handle "
    "the phrasing a real user would use. "
    "Questions were written after reading the opinions but without looking at "
    "the chunks, specifically to surface routing and retrieval failure modes."
)
st.write(
    "The eval is organized in batches (batch_001, batch_002, batch_003), each "
    "corresponding to a new set of ingested cases. New batches are gated on a "
    "minimum acceptance threshold: routing accuracy must be 100% and MRR must "
    "not regress from the prior batch before a batch is accepted into the "
    "cumulative eval set. This prevents the eval set from expanding faster "
    "than the pipeline can be validated."
)
st.write(
    "Metrics tracked: routing accuracy (did the question reach the right case), "
    "MRR (mean reciprocal rank of the first relevant chunk in retrieval results), "
    "context term coverage (fraction of answer-relevant terms present in the "
    "retrieved context), answer term rate (fraction of expected answer terms "
    "present in the generated answer), and faithfulness flags "
    "(answers that name a justice or case not present in the retrieved context)."
)
st.markdown(
    """
| Metric | Score | Notes |
|---|---|---|
| Routing accuracy | 100% | 18 routing questions across 15 cases |
| Mean MRR | 0.670 | Up from 0.632 on the prior embedding model |
| Context term coverage | 85–94% | Varies by case size |
| Answer term rate | 85.42% | 12-question batch eval |
| Faithfulness flags | 1/12 | One pre-existing edge case |
| Total eval questions | 73 | Across 3 batches, 15 cases |
"""
)
st.divider()

# ---------------------------------------------------------------------------
# 6. What's next
# ---------------------------------------------------------------------------

st.markdown("## What's next")
st.write(
    "These are framed as engineering problems to solve, "
    "not just features to add."
)
st.markdown(
    "**Corpus expansion to 100+ cases.** "
    "The ingestion pipeline, manifest system, and eval structure are already "
    "built for this — it's an execution problem, not an architecture problem. "
    "CourtListener rate limits (125 requests/day) make it a multi-week effort "
    "at current batch sizes.\n\n"
    "**Topic-aware comparative scoring.** "
    "Add a topic overlap bonus to FAISS-first retrieval to improve precision "
    "when the corpus grows and more cases share surface-level semantic similarity.\n\n"
    "**Faithfulness checking for comparative queries.** "
    "Currently skipped — the faithfulness guard checks a single case's context. "
    "Cross-case comparative answers need a version that validates against "
    "multi-case context, which requires a different check structure.\n\n"
    "**Self-hosted LLM on Azure VM.** "
    "Swap the Anthropic API for an open-source model hosted on Azure "
    "infrastructure, directly applying existing cloud engineering background "
    "to the AI infrastructure layer."
)
st.divider()

# ---------------------------------------------------------------------------
# 7. About
# ---------------------------------------------------------------------------

st.markdown("## About")
st.write(
    "**Stephen Dubay** — Azure cloud infrastructure engineer "
    "repositioning toward AI/ML engineering."
)
st.write("This project demonstrates:")
st.markdown(
    "- RAG pipeline design and evaluation\n"
    "- Vector search architecture decisions\n"
    "- LLM integration and model selection\n"
    "- Faithfulness and hallucination mitigation\n"
    "- Production deployment considerations\n"
    "- Engineering judgment under real constraints"
)
st.markdown(
    "[LinkedIn](https://www.linkedin.com/in/stephen-dubay-8a0ba5202) | "
    "[GitHub](https://github.com/scdubay/ai-SCOTUS)"
)
