# SCOTUS RAG — Change Log

Updates applied across the three pipeline files. Issue numbers match the prior
review. Nothing here changes the embedding model, the corpus, or the on-disk
vector store format, so existing indexes remain loadable.

---

## CourtListener_clean.py (ingestion)

**Majority opinions no longer labeled `"unknown"` (#4)**
In `split_opinion_sections`, the regex matches the "...delivered the opinion of
the Court" author line, but the role ladder had no branch for it, so the court
opinion fell through to `role = "unknown"`. Added an explicit
`delivered the opinion of the court -> court_opinion` branch and changed the
final `else` default from `"unknown"` to `"court_opinion"`. Stored metadata is
now correct at the source instead of relying on query-time re-derivation.

**Removed duplicate dead branch**
`split_opinion_sections` had two identical
`elif "concurring in the judgment" in label_lower:` branches; the second was
unreachable. Removed.

**Removed dead function**
`infer_opinion_role` was defined but never called (section splitting supersedes
it). Removed.

**Eliminated double dossier build**
`build_opinion_documents` rebuilt the dossier even though `fetch_demo_documents`
had already built it for the same case. It now accepts an optional `dossier`
argument and only builds one if called standalone. `fetch_demo_documents` passes
the already-built dossier through.

**Documented the legal separators**
Added a note to `create_legal_document_splitter` explaining that the uppercase
headers (`OPINION OF THE COURT`, etc.) rarely match real CourtListener
`plain_text` and that splitting falls back to paragraph/sentence boundaries.
Behavior unchanged — kept as harmless forward-compatible anchors so chunk
boundaries (and therefore the current vector store) stay stable.

---

## query_demo_clean.py (retrieval + generation)

**"Concur in part / dissent in part" no longer collapses to pure dissent (#5)**
`infer_role_from_doc` checked the bare substring `"dissent"` first, so a
`concurrence_dissent` role (and combined text markers) resolved to `"dissent"`.
Added an explicit combined-role check ahead of the dissent/concur checks, in
both the metadata path and the text-marker path, returning
`"concurrence_dissent"`.

**Reranker blend now actually blends (#3)**
Previously `final = 0.65 * vector_score + 0.35 * heuristic`, but `vector_score`
lived in (0, 1] while `heuristic` routinely reached 7-10, so the heuristic
dominated roughly 5:1 and the vector weight was cosmetic. The reranker now runs
in two passes: collect raw vector and heuristic signals, **min-max normalize
both across the candidate set to [0, 1]**, then blend. The 0.65/0.35 weights now
express their intended relative importance. Degenerate (all-equal) ranges
collapse to a neutral 0.5 to avoid divide-by-zero.

**`concurrence_dissent` handled in rerank weighting**
Added `concurrence_dissent` alongside `dissent`/`concurrence` in the
dissent-asking, concurrence-asking, and default weight branches so the new role
is scored sensibly instead of falling through to 0.

**Model sees the corrected role (#6)**
`format_context` printed the raw stored `opinion_role` (possibly `"unknown"`)
while the eval used `effective_opinion_role`. It now prefers
`effective_opinion_role` (falling back to `opinion_role`), so the generation
prompt's "distinguish majority from dissent" instruction is grounded in the same
labels the eval measures.

---

## test_retreival.py (eval harness)

**Honest (non-oracle) metrics are now primary (#1)**
The old loop filtered retrieved docs to the gold case using the answer key
(`filter_docs_by_case(docs, test["case"])`) before scoring, which made case
precision ~100% by construction and measured "given perfect case routing, do we
find the terms." The rewrite reports three tiers:

  1. **HONEST retrieval** — term positions/MRR/Hit@k measured against the full
     reranked list, before neighbor expansion and with no case filter. This is
     the primary signal and reflects true retrieval quality.
  2. **Production answer context** — top-N reranked -> neighbor expansion -> cap,
     with no knowledge of the gold case; reports term coverage and an honest
     case precision over the docs actually fed to the LLM.
  3. **DIAGNOSTIC oracle** — the old answer-key-filtered path, kept but clearly
     labeled as a ceiling/upper bound, not deployable behavior.

**Proper MRR (#2)**
`calculate_mrr` previously returned `1 / min(rank)` — the reciprocal of the
single best-ranked term — so a question scored 1.0 if any one term hit rank 1,
even with every other term missing. It now returns the mean reciprocal rank
across the expected terms (each term contributes `1/rank` if found, else 0).

**Pre-expansion ranks**
Reported term ranks are now measured against the reranked list before neighbor
expansion, so retrieval quality isn't conflated with neighbor padding.

**Removed dead code**
Deleted `infer_role_from_text` and `role_weight_for_question` (both defined but
never called; the latter also hardcoded specific justice names, the kind of
case-specificity the reranker deliberately avoids). Removed the unused
top-level `import nltk` (the `SnowballStemmer` import is separate and retained).

**New helpers**
Added `build_answer_context`, `case_precision`, and `evaluate_term_block` to
keep the three metric tiers DRY, plus a corpus-level summary (mean MRR, Hit@10
rate, AllTerms@10 rate) printed at the end. The JSON output now separates
`retrieval_*`, `context_*`, and `oracle_*` fields.

---

## Not changed (and why)

- **Filename typo `test_retreival.py`** — kept as-is to avoid breaking any
  references on your side; rename to `test_retrieval.py` at your discretion.
- **Two separate `normalize_text` implementations** — the eval keeps
  apostrophes/hyphens for term matching while the query module strips all
  punctuation for lexical scoring; these are legitimately different needs.
  Unifying risks subtly shifting match behavior mid-eval, so left as a noted
  divergence rather than forced into a shared util.
- **Chunking / separators** — left functionally unchanged so the existing vector
  store stays valid; only documented.

## Suggested next step

Re-run the eval and compare the new **HONEST** numbers against the old
(oracle-inflated) ones — the gap is your real-world retrieval headroom. Once the
metrics are trustworthy, the normalized reranker is the next thing worth tuning,
and scaling the corpus past 6 cases will make the eval far more informative.
