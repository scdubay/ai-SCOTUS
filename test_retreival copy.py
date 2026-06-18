import json
from pathlib import Path
import re
import nltk

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

def infer_role_from_text(doc) -> str:
    text = normalize_text(doc.page_content[:1000])
    existing_role = normalize_text(str(doc.metadata.get("opinion_role", "")))

    if "dissent" in existing_role:
        return "dissent"

    if "concurr" in existing_role:
        return "concurrence"

    if "syllabus" in existing_role:
        return "syllabus"

    dissent_markers = [
        "dissenting",
        "i respectfully dissent",
        "i dissent",
        "dissent from",
        "concurring in part and dissenting in part",
    ]

    concurrence_markers = [
        "concurring",
        "concurring in judgment",
        "concurring in part",
    ]

    if any(marker in text for marker in dissent_markers):
        return "dissent"

    if any(marker in text for marker in concurrence_markers):
        return "concurrence"

    return existing_role or "court_opinion"


def role_weight_for_question(question: str, role: str) -> float:
    q = normalize_text(question)

    asks_dissent = any(
        term in q
        for term in [
            "dissent",
            "dissenting",
            "criticism",
            "criticize",
            "o'connor",
            "rehnquist",
            "thomas",
        ]
    )

    asks_concurrence = any(
        term in q
        for term in [
            "concurrence",
            "concurring",
            "concurred",
        ]
    )

    if asks_dissent:
        weights = {
            "dissent": 1.00,
            "concurrence": 0.50,
            "court_opinion": 0.00,
            "syllabus": -0.25,
        }
    elif asks_concurrence:
        weights = {
            "concurrence": 1.00,
            "dissent": 0.25,
            "court_opinion": 0.00,
            "syllabus": -0.25,
        }
    else:
        weights = {
            "court_opinion": 1.00,
            "majority": 1.00,
            "lead": 1.00,
            "syllabus": 0.25,
            "concurrence": -0.25,
            "dissent": -0.75,
        }

    return weights.get(role, 0.0)


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
    ranks = [
        item["first_rank"]
        for item in term_results
        if item["found"] and item["first_rank"]
    ]

    if not ranks:
        return 0.0

    best_rank = min(ranks)
    return 1 / best_rank

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

def main():
    print("🔎 Loading vector store...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        OllamaEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    print("✅ Vector store loaded.")
    chunk_lookup = build_chunk_lookup(
    vectorstore
)

    print(
        f"✅ Built chunk lookup: "
        f"{len(chunk_lookup)} chunks"
    )

    tests = load_tests()
    results = []

    for test in tests:
        question = test["question"]
        expected_terms = test["expected_terms"]

        print("\n" + "=" * 80)
        print(f"Question: {question}")

        raw_docs = vectorstore.similarity_search_with_score(
            question,
            k=RAW_RETRIEVAL_K
        )

        docs = rerank_docs(
            question,
            raw_docs
        )

        case_docs = filter_docs_by_case(
            docs,
            test["case"]
        )

        if case_docs:
            docs = case_docs[:8]
        else:
            docs = docs[:8]

        docs = expand_neighbors(
            docs,
            chunk_lookup
        )

        docs = docs[:FINAL_DOC_LIMIT]

        combined_context = "\n\n".join(d.page_content for d in docs)
        score, missing = score_context(combined_context, expected_terms)

        term_results = track_expected_terms(
            docs,
            expected_terms
        )

        mrr = calculate_mrr(term_results)

        hit_at_5 = calculate_hit_at_k(term_results, 5)
        hit_at_10 = calculate_hit_at_k(term_results, 10)
        hit_at_20 = calculate_hit_at_k(term_results, 20)

        all_terms_hit_at_5 = calculate_all_terms_hit_at_k(term_results, 5)
        all_terms_hit_at_10 = calculate_all_terms_hit_at_k(term_results, 10)
        all_terms_hit_at_20 = calculate_all_terms_hit_at_k(term_results, 20)

        matching_case = sum(
            1
            for d in docs
            if d.metadata.get("case_title")
            == test["case"]
        )

        precision = (
            matching_case / len(docs)
            if docs
            else 0
        )

        print(
            f"Retrieval score: "
            f"{score}/{len(expected_terms)}"
        )

        print(
            f"Case precision: "
            f"{matching_case}/{len(docs)} "
            f"({precision:.0%})"
        )

        print(f"MRR: {mrr:.2f}")

        print(
            f"Hit@5={hit_at_5} | "
            f"Hit@10={hit_at_10} | "
            f"Hit@20={hit_at_20}"
        )

        print(
            f"AllTerms@5={all_terms_hit_at_5} | "
            f"AllTerms@10={all_terms_hit_at_10} | "
            f"AllTerms@20={all_terms_hit_at_20}"
        )

        print("Expected term positions:")
        for item in term_results:
            status = "✅" if item["found"] else "❌"
            print(
                f"  {status} {item['term']} | "
                f"rank={item['first_rank']} | "
                f"chunk={item['first_chunk']} | "
                f"case={item['first_case']}"
            )

        if missing:
            print(f"Missing: {', '.join(missing)}")
            debug_expected_terms(docs, missing)
        else:
            print("✅ All expected terms found in retrieved context.")

        source_rows = []
        for i, doc in enumerate(docs, start=1):
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
                "case": test["case"],
                "question": question,
                "expected_terms": expected_terms,
                "score": score,
                "max_score": len(expected_terms),
                "missing_terms": missing,
                "sources": source_rows,
                "mrr": mrr,
                "term_positions": term_results,
                "hit_at_5": hit_at_5,
                "hit_at_10": hit_at_10,
                "hit_at_20": hit_at_20,
                "all_terms_hit_at_5": all_terms_hit_at_5,
                "all_terms_hit_at_10": all_terms_hit_at_10,
                "all_terms_hit_at_20": all_terms_hit_at_20,
                            }
        )

    output_file = OUTPUT_DIR / "retrieval_eval_results.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n✅ Saved retrieval eval results to {output_file}")


if __name__ == "__main__":
    main()