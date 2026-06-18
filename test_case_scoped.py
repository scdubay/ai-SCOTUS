"""
test_case_scoped.py

Evaluates the case-scoped pipeline (resolve case -> read whole / retrieve within
case) and reports it against the same term-coverage metrics as test_retreival.py,
plus a new routing-accuracy number.

Key difference from the prior eval: the case is resolved FROM THE QUESTION TEXT,
not taken from the answer key. So this measures the real deployable path, not the
oracle. (If your product UX lets the user pick the case, routing accuracy becomes
100% by construction and the coverage numbers here are your real numbers.)

Run after `ollama serve` is up and the demo vector store exists.
"""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

from query_demo_clean import VECTORSTORE_PATH, OllamaEmbeddings

# Reuse the EXACT metric definitions from the prior eval for apples-to-apples.
from test_retreival import (
    load_tests,
    score_context,
    track_expected_terms,
    calculate_mrr,
    calculate_hit_at_k,
    calculate_all_terms_hit_at_k,
)

from case_store import (
    build_case_index,
    resolve_case,
    build_case_scoped_context,
)

OUTPUT_DIR = Path("data/evals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("🔎 Loading vector store...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        OllamaEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    case_index = build_case_index(vectorstore)
    print(f"✅ Built case index: {len(case_index)} cases")
    for title, rec in sorted(case_index.items()):
        roles = ", ".join(sorted({o["role"] for o in rec["opinions"]}))
        print(f"   - {title} | {len(rec['opinions'])} opinion(s) | "
              f"{rec['total_chars']} chars | roles: {roles}")

    tests = load_tests()
    results = []

    for test in tests:
        question = test["question"]
        expected_terms = test["expected_terms"]
        gold_case = test["case"]

        print("\n" + "=" * 80)
        print(f"Question: {question}")

        # Resolve from the QUESTION ONLY (no answer key).
        resolved_case, method, _ = resolve_case(question, case_index, vectorstore)
        routed_correct = (resolved_case == gold_case)

        print(f"Routing: resolved='{resolved_case}' via {method} | "
              f"gold='{gold_case}' | {'✅' if routed_correct else '❌'}")

        if not resolved_case:
            print("❌ Unresolved — skipping context build.")
            results.append({
                "case": gold_case, "question": question,
                "expected_terms": expected_terms, "max_score": len(expected_terms),
                "resolved_case": None, "routing_method": method,
                "routed_correct": False, "mode": "unresolved",
                "context_score": 0, "context_missing_terms": expected_terms,
                "mrr": 0.0, "hit_at_5": False, "hit_at_10": False, "hit_at_20": False,
                "all_terms_hit_at_5": False, "all_terms_hit_at_10": False,
                "all_terms_hit_at_20": False, "segments": [],
            })
            continue

        context, segments, mode = build_case_scoped_context(
            resolved_case, case_index, question, vectorstore
        )

        term_results = track_expected_terms(segments, expected_terms)
        score, missing = score_context(context, expected_terms)
        mrr = calculate_mrr(term_results)

        print(f"Mode: {mode} | segments: {len(segments)}")
        print(f"Context coverage: {score}/{len(expected_terms)}")
        print(f"MRR: {mrr:.3f}")
        print(f"Hit@5={calculate_hit_at_k(term_results, 5)} | "
              f"AllTerms@5={calculate_all_terms_hit_at_k(term_results, 5)}")

        print("Expected term positions (segment index within the case):")
        for item in term_results:
            status = "✅" if item["found"] else "❌"
            print(f"  {status} {item['term']} | segment={item['first_rank']}")

        if missing:
            print(f"Missing: {', '.join(missing)}")

        seg_rows = []
        for i, seg in enumerate(segments, start=1):
            m = seg.metadata
            seg_rows.append({
                "segment": i,
                "role": m.get("effective_opinion_role") or m.get("opinion_role"),
                "author": m.get("opinion_author"),
                "chunk_id": m.get("chunk_id"),
                "preview": seg.page_content[:300].replace("\n", " "),
            })

        results.append({
            "case": gold_case,
            "question": question,
            "expected_terms": expected_terms,
            "max_score": len(expected_terms),
            "resolved_case": resolved_case,
            "routing_method": method,
            "routed_correct": routed_correct,
            "mode": mode,
            "context_score": score,
            "context_missing_terms": missing,
            "mrr": mrr,
            "hit_at_5": calculate_hit_at_k(term_results, 5),
            "hit_at_10": calculate_hit_at_k(term_results, 10),
            "hit_at_20": calculate_hit_at_k(term_results, 20),
            "all_terms_hit_at_5": calculate_all_terms_hit_at_k(term_results, 5),
            "all_terms_hit_at_10": calculate_all_terms_hit_at_k(term_results, 10),
            "all_terms_hit_at_20": calculate_all_terms_hit_at_k(term_results, 20),
            "term_positions": term_results,
            "segments": seg_rows,
        })

    # ---- Corpus summary ----
    if results:
        n = len(results)
        routed = sum(1 for r in results if r["routed_correct"]) / n
        total_found = sum(r["context_score"] for r in results)
        total_max = sum(r["max_score"] for r in results)
        avg_mrr = sum(r["mrr"] for r in results) / n
        allterms10 = sum(1 for r in results if r["all_terms_hit_at_10"]) / n
        full_cover = sum(1 for r in results if r["context_score"] == r["max_score"]) / n

        print("\n" + "=" * 80)
        print("CORPUS SUMMARY (case-scoped)")
        print(f"  Questions:                 {n}")
        print(f"  Routing accuracy:          {routed:.0%}")
        print(f"  Context term coverage:     {total_found}/{total_max} ({total_found / total_max:.0%})")
        print(f"  Questions fully covered:   {full_cover:.0%}")
        print(f"  Mean MRR:                  {avg_mrr:.3f}")
        print(f"  AllTerms@10 rate:          {allterms10:.0%}")
        print("\n  (Prior global pipeline for reference: coverage ~72%, "
              "AllTerms@10 ~45%, mean MRR ~0.40)")

    output_file = OUTPUT_DIR / "case_scoped_eval_results.json"
    output_file.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✅ Saved case-scoped eval results to {output_file}")


if __name__ == "__main__":
    main()
