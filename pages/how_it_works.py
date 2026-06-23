"""
pages/how_it_works.py

Static "How it works" page for the SCOTUS Legal Aid Streamlit app. Pure
content -- no pipeline imports, no Anthropic/FAISS calls. Streamlit picks
this up automatically as a second navigation page because it lives under
pages/.
"""

import streamlit as st

st.set_page_config(page_title="How it works", page_icon="📘")

# ---------------------------------------------------------------------------
# 1. Header
# ---------------------------------------------------------------------------

st.title("How it works")
st.caption("Architecture, decisions, and engineering rationale behind SCOTUS Legal Aid")
st.divider()

# ---------------------------------------------------------------------------
# 2. The problem
# ---------------------------------------------------------------------------

st.markdown("## The problem")
st.write(
    "Supreme Court opinions are long, dense, and citation-heavy -- "
    "exactly the kind of text where simple keyword search falls apart, "
    "because the words that matter most (the test applied, the reasoning, "
    "the holding) rarely match the words in the question. "
    "Retrieval-augmented generation (RAG) solves this by retrieving the "
    "relevant passages from the actual opinion text first, then asking an "
    "LLM to synthesize an answer grounded in only that retrieved material."
)
st.divider()

# ---------------------------------------------------------------------------
# 3. Pipeline architecture
# ---------------------------------------------------------------------------

st.markdown("## Pipeline architecture")

_STAGES = [
    ("User Query", None),
    ("Topic Filter", "blocks off-topic and injection attempts"),
    ("Question Classifier", "META / OVERVIEW / RESEARCH (Haiku)"),
    ("Case Router", "lexical match first, semantic fallback — RESEARCH path only"),
    ("FAISS Retrieval", "case-scoped, BGE-small-en-v1.5"),
    ("Answer Generation", "Claude Haiku"),
    ("Faithfulness Guard", "flags hallucinated authorities"),
    ("Response", None),
]

left, center, right = st.columns([1, 3, 1])
with center:
    for i, (stage_name, stage_desc) in enumerate(_STAGES):
        with st.container(border=True):
            st.markdown(f"**{stage_name}**")
            if stage_desc:
                st.caption(stage_desc)
        if i < len(_STAGES) - 1:
            st.markdown("↓")

    st.info(
        "META and OVERVIEW questions short-circuit straight to generation, "
        "skipping case routing, retrieval, and the faithfulness guard "
        "entirely -- they aren't answered from retrieved opinion text."
    )
st.divider()

# ---------------------------------------------------------------------------
# 4. Key engineering decisions
# ---------------------------------------------------------------------------

st.markdown("## Key engineering decisions")

with st.expander("Why BGE-small for embeddings?"):
    st.write(
        "The corpus will grow from 6 to 100-500 cases. 768-dim embedding "
        "models blow through Streamlit Cloud's free-tier memory at that "
        "scale -- the FAISS index alone would consume most of the ~1GB "
        "budget before the model or app even loads. BGE-small keeps a "
        "384-dim footprint while outperforming MiniLM, a similarly-sized "
        "model, on retrieval benchmarks."
    )

with st.expander("Why Claude Haiku for generation?"):
    st.write(
        "Retrieval does the heavy lifting here -- by the time the LLM is "
        "called, the relevant case text has already been found. The "
        "model's job is to synthesize an answer from material that's "
        "already grounded, not to reason from scratch. Haiku is fast and "
        "cheap for that job; a larger model like Opus adds latency and "
        "cost with no meaningful quality gain when the context is already "
        "retrieved for it."
    )

with st.expander("Why hybrid lexical + semantic routing?"):
    st.write(
        "Legal citations are exact-match strings -- \"Miranda v. Arizona\" "
        "should match lexically before semantic similarity is even tried. "
        "The router checks lexical token overlap against case titles "
        "first, and only falls back to embedding similarity when lexical "
        "matching isn't confident. This hybrid approach gets 100% routing "
        "accuracy on the eval set."
    )

with st.expander("Why case-scoped retrieval?"):
    st.write(
        "Early testing retrieved across the whole corpus at once, and "
        "answers ended up contaminated with material from the wrong "
        "case. Resolving which case a question is about first, then "
        "retrieving only within that case, eliminated the cross-case "
        "contamination and improved precision."
    )

with st.expander("Why a faithfulness guard?"):
    st.write(
        "Hallucination is particularly harmful in a legal context, where "
        "an invented justice or case can look completely plausible. The "
        "guard flags any authority -- a justice or a case -- named in the "
        "answer that doesn't actually appear anywhere in the retrieved "
        "context, plus cases where an opinion's role (majority, "
        "concurrence, dissent) is misattributed."
    )

with st.expander("Why LLM-native classification over keyword lists?"):
    st.write(
        "An earlier keyword-list classifier reliably missed natural "
        "phrasing variations. A single fast Haiku classification call "
        "handles phrasing the keyword list never anticipated. Tested "
        "directly: \"what is your purpose?\" now correctly routes to META, "
        "where the keyword approach had missed it."
    )

st.divider()

# ---------------------------------------------------------------------------
# 5. Evaluation results
# ---------------------------------------------------------------------------

st.markdown("## Evaluation results")
st.markdown(
    """
| Metric | Score | Notes |
|---|---|---|
| Routing accuracy | 18/18 (100%) | 18-question expanded eval set |
| Mean MRR | 0.670 | Improved from 0.632 measured under the prior embedding model |
| Context term coverage | 85% | Down from a measured 90%, within accepted tolerance |
| Faithfulness flags | 1/12 | 12-question batch eval (`query_demo_clean.py --batch`); one pre-existing edge case |
| Answer term rate | 85.42% | Same 12-question batch eval |
"""
)
st.divider()

# ---------------------------------------------------------------------------
# 6. What's next
# ---------------------------------------------------------------------------

st.markdown("## What's next")
st.markdown(
    """
- Expand corpus to 100+ cases from CourtListener
- Thematic cross-case queries
- Orchestration layer with intent detection
- Self-hosted open source LLM on Azure VM
"""
)
st.divider()

# ---------------------------------------------------------------------------
# 7. About the builder
# ---------------------------------------------------------------------------

st.markdown("## About the builder")
st.write(
    "**Stephen Dubay** — Azure cloud infrastructure engineer "
    "repositioning toward AI/ML engineering."
)
st.markdown(
    """
This project demonstrates:
- RAG pipeline design
- Vector search
- LLM integration
- Evaluation methodology
- Faithfulness checking
- API design
- Cloud deployment
"""
)
st.markdown(
    "[LinkedIn](https://www.linkedin.com/in/stephen-dubay-8a0ba5202) | "
    "[GitHub](https://github.com/scdubay/ai-SCOTUS)"
)
