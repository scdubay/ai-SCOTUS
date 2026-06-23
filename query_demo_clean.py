"""fetch_opinion_textfetch_opinion_text
query_demo_clean.py

Purpose:
  Query the CourtListener demo FAISS vector store and generate grounded answers with source previews.

Required local models:
  ollama pull nomic-embed-text
  ollama pull llama3.2
"""

import json
import os
import re
import string
import sys
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

VECTORSTORE_PATH = os.getenv(
    "SCOTUS_VECTORSTORE_PATH",
    "data/vectors/rag_vectorstore_courtlistener_demo",
)
# Query-time/index-time embedding model. Switched from Ollama's nomic-embed-text
# (required a local Ollama server reachable at OLLAMA_EMBED_ENDPOINT, which does
# not exist on Streamlit Cloud) to a HuggingFace model that runs in-process on
# CPU. BAAI/bge-small-en-v1.5: 384 dims, ~130MB, chosen over larger 768-dim
# options (mpnet, bge-base) specifically so the FAISS index stays small as the
# corpus grows toward 100-500 cases / tens of thousands of chunks on a
# memory-constrained free-tier deploy.
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3.2")
OLLAMA_EMBED_ENDPOINT = os.getenv("OLLAMA_EMBED_ENDPOINT", "http://localhost:11434/api/embed")
OLLAMA_GENERATE_ENDPOINT = os.getenv("OLLAMA_GENERATE_ENDPOINT", "http://localhost:11434/api/generate")

# Generation backend switch: "ollama" (default, existing behavior) or "anthropic".
BACKEND = os.getenv("BACKEND", "ollama").lower()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096"))

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    return _anthropic_client


def anthropic_generate(prompt: str) -> str:
    """Send the same single-prompt structure used for Ollama to the Anthropic Messages API."""
    response = _get_anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, endpoint: str = OLLAMA_EMBED_ENDPOINT):
        self.model = model
        self.endpoint = endpoint

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts: List[str]):
        return [self._embed(t) for t in texts]

    def _embed(self, text: str):
        response = requests.post(
            self.endpoint,
            json={"model": self.model, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]
        if "embedding" in data:
            return data["embedding"]
        raise RuntimeError(f"No embedding returned: {data.keys()}")


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("–", "-").replace("—", "-")
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()

def infer_role_from_doc(doc: Document) -> str:
    meta = doc.metadata

    role = (
        meta.get("opinion_role")
        or meta.get("document_type")
        or meta.get("section")
        or ""
    ).lower()

    text = normalize_text(doc.page_content[:1200])

    # FIX (#5): a "concurring in part and dissenting in part" opinion was being
    # collapsed to pure "dissent" because the bare "dissent" substring check ran
    # first. Catch the combined role explicitly, before dissent/concur checks.
    if "concurrence_dissent" in role or (
        "concur" in role and "dissent" in role
    ):
        return "concurrence_dissent"

    if "dissent" in role:
        return "dissent"

    if "concur" in role:
        return "concurrence"

    if "syllabus" in role:
        return "syllabus"

    dissent_markers = [
        "concurring in part and dissenting in part",
        "i respectfully dissent",
        "i dissent",
        "dissenting",
        "dissent from",
    ]

    concurrence_markers = [
        "concurring",
        "concurring in judgment",
        "concurring in part",
    ]

    # Check the combined text marker before the pure-dissent markers too.
    if "concurring in part and dissenting in part" in text:
        return "concurrence_dissent"

    if any(marker in text for marker in dissent_markers):
        return "dissent"

    if any(marker in text for marker in concurrence_markers):
        return "concurrence"

    if "lead" in role:
        return "majority"

    if "court_opinion" in role or "opinion" in role:
        return "court_opinion"

    return role or "court_opinion"

def rerank_docs(question: str, docs_with_scores: List[Tuple[Document, float]]) -> List[Document]:
    """
    Generic legal-document reranker.

    This intentionally avoids case-specific doctrine boosts. It blends vector distance with
    light structural preferences: substantive chunks, court/majority material, and lower
    priority for dissent/concurrence unless explicitly requested.
    """
    normalized_question = normalize_text(question)
    question_terms = [t for t in normalized_question.split() if len(t) > 3]

    asks_about_dissent = any(term in normalized_question for term in ["dissent", "dissenting", "minority"])
    asks_about_concurrence = any(term in normalized_question for term in ["concurrence", "concurring"])

    scored_raw = []

    for doc, distance in docs_with_scores:
        text = normalize_text(doc.page_content)
        meta = doc.metadata

        # FAISS lower distance is better. Convert to a bounded-ish positive signal.
        vector_score = 1 / (1 + max(float(distance), 0.0))
        heuristic = 0.0

        # Basic lexical relevance.
        heuristic += sum(1 for term in question_terms if term in text) * 0.5

        # Prefer substantive chunks over headers/captions/fragments.
        raw_len = len(doc.page_content)
        if raw_len > 800:
            heuristic += 2.0
        if raw_len < 300:
            heuristic -= 3.0

        role = infer_role_from_doc(doc)
        meta["effective_opinion_role"] = role

        if asks_about_dissent:
            if role in ["dissent", "concurrence_dissent"]:
                heuristic += 4.0
            elif role == "concurrence":
                heuristic += 1.0
            elif role in ["court_opinion", "majority"]:
                heuristic -= 0.5

        elif asks_about_concurrence:
            if role in ["concurrence", "concurrence_dissent"]:
                heuristic += 4.0
            elif role == "dissent":
                heuristic += 1.0
            elif role in ["court_opinion", "majority"]:
                heuristic -= 0.5

        else:
            if role in ["court_opinion", "majority"]:
                heuristic += 2.5
            elif role == "syllabus":
                heuristic += 0.5
            elif role == "concurrence":
                heuristic -= 1.0
            elif role in ["dissent", "concurrence_dissent"]:
                heuristic -= 2.5

        # Penalize pure caption/header chunks.
        caption_phrases = [
            "supreme court of the united states",
            "on writ of certiorari",
            "argued",
            "decided",
        ]
        caption_hits = sum(1 for phrase in caption_phrases if phrase in text[:600])
        if caption_hits >= 3:
            heuristic -= 4.0

        scored_raw.append((vector_score, heuristic, doc))

    # FIX (#3): previously we blended 0.65 * vector_score + 0.35 * heuristic, but
    # vector_score sits in (0, 1] while heuristic routinely reached 7-10, so the
    # heuristic dominated ~5:1 and the vector weight was effectively cosmetic.
    # Min-max normalize BOTH signals across the candidate set so the blend weights
    # express their intended relative importance. Degenerate (all-equal) ranges
    # collapse to 0.5 to avoid divide-by-zero and to stay neutral.
    def _normalize(values):
        lo = min(values)
        hi = max(values)
        span = hi - lo
        if span <= 1e-9:
            return [0.5 for _ in values]
        return [(v - lo) / span for v in values]

    vec_norm = _normalize([v for v, _, _ in scored_raw])
    heur_norm = _normalize([h for _, h, _ in scored_raw])

    scored = []
    for (_, _, doc), v, h in zip(scored_raw, vec_norm, heur_norm):
        final_score = (0.65 * v) + (0.35 * h)
        scored.append((final_score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


def format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        # FIX (#6): prefer the corrected role computed during reranking so the
        # model's grounding matches what the eval sees, instead of the raw
        # stored opinion_role (which may be "unknown").
        effective_role = meta.get("effective_opinion_role") or meta.get("opinion_role")
        parts.append(
            f"""
SOURCE {i}
Case: {meta.get('case_title')}
Citation: {meta.get('citation')}
Document Type: {meta.get('document_type')}
Opinion Role: {effective_role}
Chunk: {meta.get('chunk_id')} of {meta.get('total_chunks')}
Source URL: {meta.get('source')}

Text:
{doc.page_content}
""".strip()
        )
    return "\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a legal research assistant analyzing U.S. Supreme Court materials.

Answer the question using only the provided context.

Rules:
- Identify the case or cases used.
- Distinguish majority/court reasoning from dissenting or concurring reasoning when the source context supports that distinction.
- Do not rely on general legal knowledge unless it is supported by the provided context.
- If the retrieved context is incomplete or mixed, say so.
- Cite source numbers like [Source 1], [Source 2].
- Keep the answer concise but legally precise.
- For evaluation questions, answer directly and include the named doctrine, test, holding, or rule when the context supports it.
- If the question asks for a test, list the elements of the test.
- If the question asks why the Court ruled a certain way, give the Court's main reasons, not only the conclusion.
- Do not reverse the holding. If the Court struck down a law, do not describe it as valid.
- Do not mention any Justice, separate opinion, concurrence, or dissent unless that Justice or opinion appears in the provided case material.
- If a question asks about precedents, list only the precedents appearing in the provided case material.
- Do not speculate about missing opinions, omitted dissents, omitted concurrences, or Justices who may have joined them.
- If the provided material does not include a separate opinion, simply omit it rather than commenting on its absence.

Question:
{question}

Context:
{context}

Answer:
""".strip()

    if BACKEND == "anthropic":
        return anthropic_generate(prompt)

    response = requests.post(
        OLLAMA_GENERATE_ENDPOINT,
        json={
            "model": GEN_MODEL,
            "prompt": prompt,
            "stream": False,
            # Ollama defaults num_ctx to a small window (commonly 2048) and
            # silently truncates anything longer, which quietly drops retrieved
            # context. Set it explicitly.
            "options": {"num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def print_sources(docs: List[Document]) -> None:
    print("\nSOURCES")
    print("=" * 80)
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        print(f"[{i}] {meta.get('case_title')} | {meta.get('citation')}")
        print(f"    Type: {meta.get('document_type')} | Role: {meta.get('opinion_role')}")
        print(f"    Chunk: {meta.get('chunk_id')} of {meta.get('total_chunks')}")
        print(f"    URL: {meta.get('source')}")
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"    Preview: {preview}...")
        print()


def main() -> None:
    print("🔎 Loading vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("✅ Vector store loaded.")
    print("Ask a question about the indexed Supreme Court cases.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        print("\n🔍 Retrieving relevant chunks...")
        raw_docs = vectorstore.similarity_search_with_score(question, k=30)
        docs = rerank_docs(question, raw_docs)[:8]
        context = format_context(docs)

        print("🤖 Generating answer...\n")
        answer = generate_answer(question, context)

        print("ANSWER")
        print("=" * 80)
        print(answer)
        print_sources(docs)


def run_batch(
    test_file: str = "data/evals/test_questions.json",
    output_file: str = "data/evals/answer_results.json",
    ) -> None:
    """Iterate the test questions and generate grounded answers, writing a results file.

    ```
    Uses the case-scoped path (resolve case -> read whole / retrieve within case ->
    generate), which is the validated pipeline. Generating through the old global
    retrieval would reintroduce the cross-case contamination we just removed, so we
    deliberately route through case_store here. Imported locally to avoid a circular
    import (case_store imports from this module).
    """
    from case_store import (
        build_case_index,
        resolve_case,
        build_case_scoped_context,
        case_generate_answer,
    )

    tests_path = Path(test_file)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("🔎 Loading vector store...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL),
        allow_dangerous_deserialization=True,
    )
    case_index = build_case_index(vectorstore)
    print(f"✅ Case index: {len(case_index)} cases")

    with tests_path.open("r", encoding="utf-8") as f:
        tests = json.load(f)

    results = []

    for n, test in enumerate(tests, start=1):
        question = test["question"]
        expected_terms = test.get("expected_terms", [])
        forbidden_terms = test.get("forbidden_terms", [])
        gold_case = test.get("case")

        print("\n" + "=" * 80)
        print(f"[{n}/{len(tests)}] {question}")

        case_title, method, _ = resolve_case(question, case_index, vectorstore)

        if not case_title:
            print("❌ Unresolved — recording empty answer.")
            results.append({
                "question": question,
                "gold_case": gold_case,
                "resolved_case": None,
                "routing_method": method,
                "routed_correct": False,
                "mode": "unresolved",
                "answer": "",
                "expected_terms": expected_terms,
                "context_terms_present": 0,
                "context_term_rate": 0.0 if expected_terms else None,
                "answer_terms_present": 0,
                "answer_term_rate": 0.0 if expected_terms else None,
                "forbidden_terms": forbidden_terms,
                "forbidden_hits": [],
                "faithfulness": {
                    "flagged": False,
                    "unsupported_justices": [],
                    "unsupported_cases": [],
                    "role_conflicts": [],
                },
                "sources": [],
            })
            continue

        context, segments, mode = build_case_scoped_context(
            case_title,
            case_index,
            question,
            vectorstore,
        )

        print(f"📂 {case_title} ({method}) | mode={mode} | {len(segments)} segment(s)")

        print("🤖 Generating answer...")
        answer = case_generate_answer(question, case_title, context)

        # Faithfulness guard: catch hallucinated authorities / role misattribution
        # that term-coverage can't see.
        import faithfulness

        faith = faithfulness.check_answer(
            answer=answer,
            context=context,
            segments=segments,
            case_title=case_title,
        )

        # Light grounding indicators:
        # - context_terms_present: expected terms found in retrieved context
        # - answer_terms_present: expected terms found in generated answer
        # These are simple substring checks, not semantic grading.
        ctx_lower = context.lower()
        answer_lower = answer.lower()

        context_terms_present = sum(
            1 for t in expected_terms
            if t.lower() in ctx_lower
        )

        answer_terms_present = sum(
            1 for t in expected_terms
            if t.lower() in answer_lower
        )

        context_term_rate = (
            context_terms_present / len(expected_terms)
            if expected_terms else None
        )

        answer_term_rate = (
            answer_terms_present / len(expected_terms)
            if expected_terms else None
        )

        # Negative-control / contradiction check.
        # Used for trap questions where certain phrases indicate a bad answer,
        # such as saying Meyer was a "valid exercise" of police power.
        forbidden_hits = [
            t for t in forbidden_terms
            if t.lower() in answer_lower
        ]

        sources = []

        for i, seg in enumerate(segments, start=1):
            m = seg.metadata
            sources.append({
                "segment": i,
                "role": m.get("effective_opinion_role") or m.get("opinion_role"),
                "author": m.get("opinion_author"),
                "chunk_id": m.get("chunk_id"),
                "citation": m.get("citation"),
                "source": m.get("source"),
                "preview": seg.page_content[:300].replace("\n", " "),
            })

        results.append({
            "question": question,
            "gold_case": gold_case,
            "resolved_case": case_title,
            "routing_method": method,
            "routed_correct": case_title == gold_case,
            "mode": mode,
            "answer": answer,
            "expected_terms": expected_terms,
            "context_terms_present": context_terms_present,
            "context_term_rate": context_term_rate,
            "answer_terms_present": answer_terms_present,
            "answer_term_rate": answer_term_rate,
            "forbidden_terms": forbidden_terms,
            "forbidden_hits": forbidden_hits,
            "faithfulness": faith,
            "sources": sources,
        })

        print("ANSWER")
        print("-" * 80)
        print(answer)
        print(faithfulness.summarize(faith))

        if forbidden_hits:
            print(
                "⚠️ forbidden term(s): "
                + ", ".join(forbidden_hits)
            )

    out_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    routed = sum(1 for r in results if r["routed_correct"])
    flagged = sum(
        1 for r in results
        if r.get("faithfulness", {}).get("flagged")
    )
    forbidden_count = sum(
        1 for r in results
        if r.get("forbidden_hits")
    )

    answer_rates = [
        r["answer_term_rate"]
        for r in results
        if r.get("answer_term_rate") is not None
    ]

    context_rates = [
        r["context_term_rate"]
        for r in results
        if r.get("context_term_rate") is not None
    ]

    avg_answer_rate = (
        sum(answer_rates) / len(answer_rates)
        if answer_rates else None
    )

    avg_context_rate = (
        sum(context_rates) / len(context_rates)
        if context_rates else None
    )

    print("\n" + "=" * 80)
    print(f"✅ Wrote {len(results)} answers to {out_path}")
    print(f"   Routing correct:      {routed}/{len(results)}")
    print(
        f"   Faithfulness flagged: {flagged}/{len(results)} "
        f"(answers with unsupported authorities or role conflicts)"
    )
    print(f"   Forbidden-term hits:  {forbidden_count}/{len(results)}")

    if avg_context_rate is not None:
        print(f"   Avg context term rate: {avg_context_rate:.2%}")

    if avg_answer_rate is not None:
        print(f"   Avg answer term rate:  {avg_answer_rate:.2%}")

    if avg_answer_rate is not None and avg_answer_rate < 0.70:
        print("⚠️ ANSWER QUALITY REVIEW NEEDED: average answer term rate is below 70%.")

    if routed < len(results):
        print("❌ ROUTING REVIEW NEEDED: at least one question routed to the wrong case.")

    if flagged > 0:
        print("⚠️ FAITHFULNESS REVIEW NEEDED: one or more answers were flagged.")

    if forbidden_count > 0:
        print("⚠️ ANSWER CONTRADICTION REVIEW NEEDED: forbidden terms appeared in one or more answers.")

    # Optional strict regression gate for the known-good smoke test.
    # Normal batch mode reports flags but does not crash.
    strict_smoke = "--strict-smoke" in sys.argv

    if strict_smoke and str(tests_path).replace("\\", "/").endswith("data/evals/test_questions.json"):
        assert routed == len(results), "Smoke test routing regression"
        assert flagged == 0, "Smoke test faithfulness regression"
        assert forbidden_count == 0, "Smoke test forbidden-term regression"
        assert avg_answer_rate is not None and avg_answer_rate >= 0.70, "Smoke test answer-quality regression"



if __name__ == "__main__":
    if "--batch" in sys.argv:
        test_file = "data/evals/test_questions.json"
        output_file = "data/evals/answer_results.json"

        if "--expanded" in sys.argv:
            test_file = "data/evals/test_questions_expanded.json"
            output_file = "data/evals/answer_results_expanded.json"

        if "--test-file" in sys.argv:
            idx = sys.argv.index("--test-file")
            if idx + 1 >= len(sys.argv):
                raise SystemExit("Missing value after --test-file")
            test_file = sys.argv[idx + 1]

        if "--output-file" in sys.argv:
            idx = sys.argv.index("--output-file")
            if idx + 1 >= len(sys.argv):
                raise SystemExit("Missing value after --output-file")
            output_file = sys.argv[idx + 1]

        run_batch(test_file=test_file, output_file=output_file)
    else:
        main()