import json
from pathlib import Path
import re

from query_demo_clean import (
    VECTORSTORE_PATH,
    OllamaEmbeddings,
    rerank_docs,
)
from langchain_community.vectorstores import FAISS


TEST_FILE = Path("data/evals/test_questions.json")
OUTPUT_DIR = Path("data/evals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_RETRIEVAL_K = 50
TOP_N_BEFORE_EXPAND = 8
FINAL_DOC_LIMIT = 20


def load_tests():
    with TEST_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)
from nltk.stem import SnowballStemmer

STEMMER = SnowballStemmer("english")


def normalize_text(value: str) -> str:
    value = value.lower()
    value = value.replace("’", "'")
    value = value.replace("“", '"').replace("”", '"')
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9\s'-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def tokenize(value: str):
    return re.findall(
        r"[a-z0-9]+(?:'[a-z]+)?",
        normalize_text(value)
    )


def stem_tokens(tokens):
    return [
        STEMMER.stem(token)
        for token in tokens
    ]


def ordered_subsequence_match(term_tokens, text_tokens, max_gap=2) -> bool:
    if not term_tokens:
        return False

    window_size = len(term_tokens) + max_gap

    for i in range(0, len(text_tokens)):
        window = text_tokens[i:i + window_size]

        pos = 0

        for token in term_tokens:
            try:
                pos = window.index(token, pos) + 1
            except ValueError:
                break
        else:
            return True

    return False


def term_matches(term: str, text: str) -> bool:
    term_norm = normalize_text(term)
    text_norm = normalize_text(text)

    # Fast exact normalized phrase match
    if term_norm in text_norm:
        return True

    term_stems = stem_tokens(tokenize(term))
    text_stems = stem_tokens(tokenize(text))

    if not term_stems:
        return False

    # Single-token legal terms / case-name fragments
    if len(term_stems) == 1:
        return term_stems[0] in text_stems

    # Ordered phrase/subsequence match with small gap allowance
    return ordered_subsequence_match(
        term_stems,
        text_stems,
        max_gap=2
    )

def debug_expected_terms(docs, expected_terms):
    print("\nDEBUG TERM SEARCH:")
    for term in expected_terms:
        print(f"\nTERM: {term}")
        for rank, doc in enumerate(docs, start=1):
            text = doc.page_content
            normalized = normalize_text(text)

            if term_matches(term, text):
                print(
                    f"  ✅ matched at rank={rank}, "
                    f"chunk={doc.metadata.get('chunk_id')}"
                )
                break

            # Show likely near-misses for important terms
            simple = normalize_text(term).split()[0]
            if simple in normalized:
                print(
                    f"  ⚠️ near miss rank={rank}, "
                    f"chunk={doc.metadata.get('chunk_id')}"
                )
                print(f"     {normalized[:300]}")

def score_context(context: str, expected_terms):
    missing = [
        term
        for term in expected_terms
        if not term_matches(term, context)
    ]

    return len(expected_terms) - len(missing), missing

def track_expected_terms(docs, expected_terms):
    term_results = []

    for term in expected_terms:
        found = False
        first_rank = None
        first_chunk = None
        first_case = None

        for rank, doc in enumerate(docs, start=1):
            if term_matches(term, doc.page_content):
                found = True
                first_rank = rank
                first_chunk = doc.metadata.get("chunk_id")
                first_case = doc.metadata.get("case_title")
                break

        term_results.append(
            {
                "term": term,
                "found": found,
                "first_rank": first_rank,
                "first_chunk": first_chunk,
                "first_case": first_case,
            }
        )

    return term_results

def calculate_mrr(term_results):
    # FIX (#2): the previous implementation returned 1 / min(rank) — the
    # reciprocal of the single best-ranked term — so a question scored 1.0 if
    # ANY one term landed at rank 1, even with every other term missing.
    # Proper MRR here is the mean reciprocal rank across the expected terms:
    # each term contributes 1/rank if found, 0 otherwise.
    if not term_results:
        return 0.0

    total = 0.0
    for item in term_results:
        if item["found"] and item["first_rank"]:
            total += 1.0 / item["first_rank"]

    return total / len(term_results)

def calculate_hit_at_k(term_results, k: int) -> bool:
    return any(
        item["found"]
        and item["first_rank"] is not None
        and item["first_rank"] <= k
        for item in term_results
    )


def calculate_all_terms_hit_at_k(term_results, k: int) -> bool:
    return all(
        item["found"]
        and item["first_rank"] is not None
        and item["first_rank"] <= k
        for item in term_results
    )

def filter_docs_by_case(docs, case_name):

    return [
        d
        for d in docs
        if d.metadata.get("case_title", "").lower()
        == case_name.lower()
    ]


def build_chunk_lookup(vectorstore):

    lookup = {}

    for doc in vectorstore.docstore._dict.values():

        case_title = doc.metadata.get("case_title")
        chunk_id = doc.metadata.get("chunk_id")

        if case_title is None:
            continue

        if chunk_id is None:
            continue

        lookup[(case_title, chunk_id)] = doc

    return lookup


def expand_neighbors(docs, chunk_lookup):

    expanded = []
    seen = set()

    for doc in docs:

        case_title = doc.metadata.get("case_title")
        chunk_id = doc.metadata.get("chunk_id")

        if chunk_id is None:
            continue

        for offset in [-1, 0, 1]:

            key = (
                case_title,
                chunk_id + offset
            )

            if key in chunk_lookup:

                if key not in seen:

                    expanded.append(
                        chunk_lookup[key]
                    )

                    seen.add(key)

    return expanded


def build_answer_context(reranked_docs, chunk_lookup, top_n=TOP_N_BEFORE_EXPAND, limit=FINAL_DOC_LIMIT):
    """
    Build the answer context the way production would: take the top-N reranked
    docs WITHOUT any knowledge of the gold case, expand to neighboring chunks
    for continuity, then cap. This is the honest, deployable retrieval path.
    """
    seed = reranked_docs[:top_n]
    expanded = expand_neighbors(seed, chunk_lookup)
    return expanded[:limit]


def case_precision(docs, gold_case):
    if not docs:
        return 0, 0.0
    matching = sum(
        1 for d in docs if d.metadata.get("case_title") == gold_case
    )
    return matching, matching / len(docs)


def evaluate_term_block(docs, expected_terms):
    """Compute the metric bundle for a given ordered list of docs."""
    term_results = track_expected_terms(docs, expected_terms)
    combined_context = "\n\n".join(d.page_content for d in docs)
    score, missing = score_context(combined_context, expected_terms)
    return {
        "term_results": term_results,
        "score": score,
        "missing": missing,
        "mrr": calculate_mrr(term_results),
        "hit_at_5": calculate_hit_at_k(term_results, 5),
        "hit_at_10": calculate_hit_at_k(term_results, 10),
        "hit_at_20": calculate_hit_at_k(term_results, 20),
        "all_terms_hit_at_5": calculate_all_terms_hit_at_k(term_results, 5),
        "all_terms_hit_at_10": calculate_all_terms_hit_at_k(term_results, 10),
        "all_terms_hit_at_20": calculate_all_terms_hit_at_k(term_results, 20),
    }


def main():
    print("🔎 Loading vector store...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        OllamaEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    print("✅ Vector store loaded.")

    chunk_lookup = build_chunk_lookup(vectorstore)
    print(f"✅ Built chunk lookup: {len(chunk_lookup)} chunks")

    tests = load_tests()
    results = []

    for test in tests:
        question = test["question"]
        expected_terms = test["expected_terms"]
        gold_case = test["case"]

        print("\n" + "=" * 80)
        print(f"Question: {question}")

        raw_docs = vectorstore.similarity_search_with_score(
            question, k=RAW_RETRIEVAL_K
        )

        # Full reranked list — this is the honest retrieval ranking.
        reranked = rerank_docs(question, raw_docs)

        # -------------------------------------------------------------------
        # PRIMARY (HONEST) METRICS
        # Term positions are measured against the full reranked list, BEFORE
        # neighbor expansion and WITHOUT any oracle case filter. These ranks
        # reflect true retrieval quality (FIX #1, plus pre-expansion ranks).
        # -------------------------------------------------------------------
        retrieval = evaluate_term_block(reranked, expected_terms)

        # Production-style answer context: top-N reranked -> expand -> cap.
        # No knowledge of the gold case is used here.
        answer_docs = build_answer_context(reranked, chunk_lookup)
        context_eval = evaluate_term_block(answer_docs, expected_terms)
        ctx_matching, ctx_precision = case_precision(answer_docs, gold_case)

        # -------------------------------------------------------------------
        # DIAGNOSTIC (ORACLE) METRICS
        # Filter retrieved docs to the gold case using the answer key, then
        # expand. This is an UPPER BOUND / ceiling, not deployable behavior —
        # it tells us how well we'd do if case routing were perfect.
        # -------------------------------------------------------------------
        case_docs = filter_docs_by_case(reranked, gold_case)
        if case_docs:
            oracle_seed = case_docs[:TOP_N_BEFORE_EXPAND]
        else:
            oracle_seed = reranked[:TOP_N_BEFORE_EXPAND]
        oracle_docs = expand_neighbors(oracle_seed, chunk_lookup)[:FINAL_DOC_LIMIT]
        oracle_eval = evaluate_term_block(oracle_docs, expected_terms)
        oracle_matching, oracle_precision = case_precision(oracle_docs, gold_case)

        # ---- Reporting ----
        print("\n-- HONEST retrieval (full reranked list, no oracle) --")
        print(f"Retrieval score: {retrieval['score']}/{len(expected_terms)}")
        print(f"MRR (mean over terms): {retrieval['mrr']:.3f}")
        print(
            f"Hit@5={retrieval['hit_at_5']} | "
            f"Hit@10={retrieval['hit_at_10']} | "
            f"Hit@20={retrieval['hit_at_20']}"
        )
        print(
            f"AllTerms@5={retrieval['all_terms_hit_at_5']} | "
            f"AllTerms@10={retrieval['all_terms_hit_at_10']} | "
            f"AllTerms@20={retrieval['all_terms_hit_at_20']}"
        )

        print("\n-- Production answer context (top-N -> neighbors, no oracle) --")
        print(f"Context term coverage: {context_eval['score']}/{len(expected_terms)}")
        print(
            f"Case precision: {ctx_matching}/{len(answer_docs)} ({ctx_precision:.0%})"
        )

        print("\n-- DIAGNOSTIC oracle (case-filtered by answer key; ceiling only) --")
        print(f"Oracle term coverage: {oracle_eval['score']}/{len(expected_terms)}")
        print(
            f"Oracle case precision: "
            f"{oracle_matching}/{len(oracle_docs)} ({oracle_precision:.0%})"
        )

        print("\nExpected term positions (honest reranked ranks):")
        for item in retrieval["term_results"]:
            status = "✅" if item["found"] else "❌"
            print(
                f"  {status} {item['term']} | "
                f"rank={item['first_rank']} | "
                f"chunk={item['first_chunk']} | "
                f"case={item['first_case']}"
            )

        if retrieval["missing"]:
            print(f"Missing (honest retrieval): {', '.join(retrieval['missing'])}")
            debug_expected_terms(reranked, retrieval["missing"])
        else:
            print("✅ All expected terms found in honest retrieved context.")

        source_rows = []
        for i, doc in enumerate(answer_docs, start=1):
            meta = doc.metadata
            row = {
                "rank": i,
                "case_title": meta.get("case_title"),
                "citation": meta.get("citation"),
                "document_type": meta.get("document_type"),
                "opinion_role": meta.get("effective_opinion_role", meta.get("opinion_role")),
                "section_label": meta.get("section_label"),
                "chunk_id": meta.get("chunk_id"),
                "total_chunks": meta.get("total_chunks"),
                "source": meta.get("source"),
                "preview": doc.page_content[:500].replace("\n", " "),
            }
            source_rows.append(row)

            print(
                f"[{i}] {row['case_title']} | "
                f"role={row['opinion_role']} | "
                f"chunk={row['chunk_id']}/{row['total_chunks']}"
            )
            print(f"    {row['preview']}...")

        results.append(
            {
                "case": gold_case,
                "question": question,
                "expected_terms": expected_terms,
                "max_score": len(expected_terms),

                # Honest retrieval metrics (primary).
                "retrieval_score": retrieval["score"],
                "retrieval_missing_terms": retrieval["missing"],
                "mrr": retrieval["mrr"],
                "term_positions": retrieval["term_results"],
                "hit_at_5": retrieval["hit_at_5"],
                "hit_at_10": retrieval["hit_at_10"],
                "hit_at_20": retrieval["hit_at_20"],
                "all_terms_hit_at_5": retrieval["all_terms_hit_at_5"],
                "all_terms_hit_at_10": retrieval["all_terms_hit_at_10"],
                "all_terms_hit_at_20": retrieval["all_terms_hit_at_20"],

                # Production answer-context metrics.
                "context_score": context_eval["score"],
                "context_missing_terms": context_eval["missing"],
                "context_case_precision": ctx_precision,

                # Diagnostic oracle ceiling.
                "oracle_score": oracle_eval["score"],
                "oracle_case_precision": oracle_precision,

                "sources": source_rows,
            }
        )

    # Corpus-level summary across questions.
    if results:
        n = len(results)
        avg_mrr = sum(r["mrr"] for r in results) / n
        hit10 = sum(1 for r in results if r["hit_at_10"]) / n
        allterms10 = sum(1 for r in results if r["all_terms_hit_at_10"]) / n
        print("\n" + "=" * 80)
        print("CORPUS SUMMARY (honest retrieval)")
        print(f"  Questions:           {n}")
        print(f"  Mean MRR:            {avg_mrr:.3f}")
        print(f"  Hit@10 rate:         {hit10:.0%}")
        print(f"  AllTerms@10 rate:    {allterms10:.0%}")

    output_file = OUTPUT_DIR / "retrieval_eval_results.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✅ Saved retrieval eval results to {output_file}")


if __name__ == "__main__":
    main()
