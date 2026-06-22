"""
case_store.py

Case-scoped retrieval for the SCOTUS analysis tool.

Design (matches the "junior assistant pulls the file" model):
  1. RESOLVE the case the question is about (cheap; not similarity search).
  2. READ within that case:
       - small case  -> hand the whole case (all opinions) to the model
       - large case  -> retrieve within the case subset only (uncontaminated)

The global FAISS store is demoted here: it is used only as (a) a fallback for
"which case?" when the question doesn't name one, and (b) within-case search for
opinions too large to read whole. Per-case analysis no longer competes across
the whole corpus, which is what caused the cross-case contamination in the eval.

Reuses model/config from query_demo_clean so behavior stays consistent.
"""

import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import requests
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from query_demo_clean import (
    VECTORSTORE_PATH,
    GEN_MODEL,
    OLLAMA_GENERATE_ENDPOINT,
    BACKEND,
    OllamaEmbeddings,
    rerank_docs,
    normalize_text,
    anthropic_generate,
)

# Budgets. llama3.2 via Ollama defaults to a SMALL context window (num_ctx)
# unless told otherwise, so we set it explicitly and keep the read-whole budget
# comfortably under it (leaving room for the prompt scaffold and the answer).
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
READ_WHOLE_CHAR_BUDGET = int(os.getenv("READ_WHOLE_CHAR_BUDGET", "18000"))
WITHIN_CASE_CHAR_BUDGET = int(os.getenv("WITHIN_CASE_CHAR_BUDGET", "14000"))

# Minimum FAISS distance (lower = closer) the single nearest chunk must clear
# before resolve_case()'s similarity fallback is trusted. Without this, the
# fallback always returns *some* case -- even for questions with no real
# connection to any indexed case -- because nearest-neighbor search has no
# concept of "no good match." Calibrated empirically: genuine in-corpus
# matches land ~0.5-0.72; off-topic/meta-ish questions land ~1.0-1.2.
SIMILARITY_MAX_DISTANCE = float(os.getenv("SIMILARITY_MAX_DISTANCE", "0.9"))

# Tokens that don't help identify a case (parties/structure words common to many).
GENERIC_TITLE_TOKENS = {
    "v", "vs", "the", "of", "and", "in", "re", "ex", "rel",
    "united", "states", "state", "real", "property", "board",
    "education", "city", "county", "department", "dept", "commissioner",
    "commission", "company", "co", "corp", "inc", "et", "al",
}

# Ordering for presenting opinions when reading a whole case.
_ROLE_ORDER = {
    "court_opinion": 0,
    "majority": 0,
    "syllabus": 1,
    "concurrence": 2,
    "concurrence_in_judgment": 2,
    "concurrence_dissent": 3,
    "dissent": 4,
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _title_tokens(text: str) -> List[str]:
    # Split on apostrophes too, so possessives ("Nebraska's") match the bare
    # token and names ("O'Connor") yield a usable token ("connor").
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in toks if len(t) > 2 and t not in GENERIC_TITLE_TOKENS]


def _dedup_join(prev: str, nxt: str, max_overlap: int = 320, min_overlap: int = 25) -> str:
    """Concatenate two consecutive (overlapping) chunks, removing the overlap.

    RecursiveCharacterTextSplitter emits ~chunk_overlap characters of shared text
    between neighbors; we strip the largest exact suffix/prefix match we can find.
    """
    if not prev:
        return nxt
    if not nxt:
        return prev
    cap = min(len(prev), len(nxt), max_overlap)
    for k in range(cap, min_overlap - 1, -1):
        if prev[-k:] == nxt[:k]:
            return prev + nxt[k:]
    joiner = "" if prev.endswith("\n") else "\n"
    return prev + joiner + nxt


def classify_opinion_role(text: str, stored_role: str = "") -> str:
    """Classify an opinion by reading its opening, not by trusting stored metadata.

    Stored opinion_role is unreliable in the current corpus (section splitting
    didn't fire, so most opinions were stamped "court_opinion" regardless of who
    actually wrote them). We re-derive from the opening lines, where SCOTUS
    opinions announce author and role.
    """
    head = normalize_text(text[:500])
    stored = (stored_role or "").lower()

    # Combined writings first (substring "dissent" would otherwise swallow them).
    if "concurring in part and dissenting in part" in head or "concurrence_dissent" in stored:
        return "concurrence_dissent"
    if "concurring in the judgment" in head:
        return "concurrence_in_judgment"
    if "delivered the opinion of the court" in head:
        return "court_opinion"
    if re.search(r"\bdissent(ing|ed|s)?\b", head):
        return "dissent"
    if re.search(r"\bconcurr", head):
        return "concurrence"
    if "syllabus" in head or "syllabus" in stored:
        return "syllabus"
    if "dissent" in stored:
        return "dissent"
    if "concur" in stored:
        return "concurrence"
    return "court_opinion"


# ---------------------------------------------------------------------------
# Case index
# ---------------------------------------------------------------------------

def build_case_index(vectorstore: FAISS) -> Dict[str, dict]:
    """Group all stored chunks into per-case records with reconstructed opinions.

    Returns: { case_title: {
        case_title, citation, total_chars,
        opinions: [ {opinion_key, role, author, text, char_len, n_chunks,
                     source, chunks: [Document ordered by chunk_id]} ],
    } }
    """
    # Group chunks by (case_title, opinion identity).
    groups: Dict[str, Dict[tuple, List[Document]]] = defaultdict(lambda: defaultdict(list))
    citations: Dict[str, str] = {}

    for doc in vectorstore.docstore._dict.values():
        meta = doc.metadata
        case_title = meta.get("case_title")
        if not case_title:
            continue
        citations.setdefault(case_title, meta.get("citation") or "")
        opinion_key = (
            meta.get("opinion_id"),
            meta.get("section_label"),
            meta.get("section_order"),
        )
        groups[case_title][opinion_key].append(doc)

    index: Dict[str, dict] = {}

    for case_title, opinion_groups in groups.items():
        opinions = []
        for opinion_key, chunks in opinion_groups.items():
            chunks_sorted = sorted(
                chunks,
                key=lambda d: (d.metadata.get("chunk_id") if d.metadata.get("chunk_id") is not None else 0),
            )
            text = ""
            for ch in chunks_sorted:
                text = _dedup_join(text, ch.page_content)
            text = text.strip()
            if not text:
                continue

            stored_role = chunks_sorted[0].metadata.get("opinion_role", "")
            role = classify_opinion_role(text, stored_role)
            author = chunks_sorted[0].metadata.get("opinion_author") or ""
            source = chunks_sorted[0].metadata.get("source") or ""

            opinions.append(
                {
                    "opinion_key": opinion_key,
                    "role": role,
                    "author": author,
                    "text": text,
                    "char_len": len(text),
                    "n_chunks": len(chunks_sorted),
                    "source": source,
                    "chunks": chunks_sorted,
                }
            )

        opinions.sort(key=lambda o: (_ROLE_ORDER.get(o["role"], 5), str(o["opinion_key"])))
        index[case_title] = {
            "case_title": case_title,
            "citation": citations.get(case_title, ""),
            "total_chars": sum(o["char_len"] for o in opinions),
            "opinions": opinions,
        }

    return index


# ---------------------------------------------------------------------------
# Case resolution
# ---------------------------------------------------------------------------

def resolve_case(
    question: str,
    case_index: Dict[str, dict],
    vectorstore: Optional[FAISS] = None,
    explicit: Optional[str] = None,
) -> Tuple[Optional[str], str, dict]:
    """Decide which case the question is about.

    Order of preference (cheap -> expensive):
      1. explicit selection (case-scoped product: the user already chose)
      2. lexical match of distinctive case-title tokens present in the question
      3. similarity fallback: majority case among the top-k nearest chunks

    Returns (case_title, method, debug_scores).
    """
    if explicit:
        # Tolerate slight formatting differences; match on normalized title.
        want = normalize_text(explicit)
        for title in case_index:
            if normalize_text(title) == want:
                return title, "explicit", {}
        # Explicit but unknown -> fall through to inference.

    q_tokens = set(_title_tokens(question))

    lexical_scores = {}
    for title in case_index:
        title_toks = set(_title_tokens(title))
        if not title_toks:
            lexical_scores[title] = 0.0
            continue
        hits = len(title_toks & q_tokens)
        # Fraction of the title's distinctive tokens named in the question.
        lexical_scores[title] = hits / len(title_toks) if hits else 0.0

    ranked = sorted(lexical_scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] > 0:
        top_title, top_score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        # Confident if the top case is clearly ahead.
        if top_score >= 0.5 and top_score - runner >= 0.25:
            return top_title, "lexical", {"lexical": lexical_scores}

    # Similarity fallback: which case do the nearest chunks belong to?
    # Only trust this when the single nearest chunk is genuinely close --
    # see SIMILARITY_MAX_DISTANCE above for why the threshold exists.
    if vectorstore is not None:
        nearest = vectorstore.similarity_search_with_score(question, k=10)
        if nearest and float(nearest[0][1]) <= SIMILARITY_MAX_DISTANCE:
            tally = defaultdict(float)
            for doc, score in nearest:
                title = doc.metadata.get("case_title")
                if title:
                    # Closer chunks (lower FAISS distance) count for more.
                    tally[title] += 1.0 / (1.0 + max(float(score), 0.0))
            if tally:
                best = max(tally.items(), key=lambda kv: kv[1])[0]
                return best, "similarity", {"lexical": lexical_scores, "similarity": dict(tally)}

    # Last resort: best lexical even if weak.
    if ranked and ranked[0][1] > 0:
        return ranked[0][0], "lexical_weak", {"lexical": lexical_scores}

    return None, "unresolved", {"lexical": lexical_scores}


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

def _opinion_to_document(case_title: str, citation: str, opinion: dict) -> Document:
    return Document(
        page_content=opinion["text"],
        metadata={
            "case_title": case_title,
            "citation": citation,
            "opinion_role": opinion["role"],
            "effective_opinion_role": opinion["role"],
            "opinion_author": opinion["author"],
            "section_label": opinion["opinion_key"][1] if opinion["opinion_key"] else "",
            "source": opinion["source"],
            "n_chunks": opinion["n_chunks"],
        },
    )


def _within_case_retrieve(
    case_title: str,
    case_record: dict,
    question: str,
    vectorstore: FAISS,
    char_budget: int,
) -> List[Document]:
    """Retrieve within a single (large) case only. No cross-case competition."""
    raw = vectorstore.similarity_search_with_score(question, k=80)
    filtered = [(d, s) for d, s in raw if d.metadata.get("case_title") == case_title]
    if not filtered:
        filtered = raw  # degenerate; shouldn't happen for a resolved case

    ranked = rerank_docs(question, filtered)

    # Neighbor lookup within this case, keyed per opinion so chunk_ids don't collide.
    lookup: Dict[tuple, Dict[int, Document]] = defaultdict(dict)
    for opinion in case_record["opinions"]:
        key = opinion["opinion_key"]
        for ch in opinion["chunks"]:
            cid = ch.metadata.get("chunk_id")
            if cid is not None:
                lookup[key][cid] = ch

    assembled: List[Document] = []
    seen = set()
    used_chars = 0

    def _key_of(doc: Document) -> tuple:
        m = doc.metadata
        return (m.get("opinion_id"), m.get("section_label"), m.get("section_order"))

    for doc in ranked:
        okey = _key_of(doc)
        cid = doc.metadata.get("chunk_id")
        for offset in (0, -1, 1):  # include the hit and its immediate neighbors
            if cid is None and offset != 0:
                continue
            neighbor = doc if (cid is None or offset == 0) else lookup.get(okey, {}).get(cid + offset)
            if neighbor is None:
                continue
            ident = (okey, neighbor.metadata.get("chunk_id"))
            if ident in seen:
                continue
            if used_chars + len(neighbor.page_content) > char_budget and assembled:
                continue
            assembled.append(neighbor)
            seen.add(ident)
            used_chars += len(neighbor.page_content)
        if used_chars >= char_budget:
            break

    return assembled


def build_case_scoped_context(
    case_title: str,
    case_index: Dict[str, dict],
    question: str,
    vectorstore: FAISS,
    read_whole_budget: int = READ_WHOLE_CHAR_BUDGET,
    within_case_budget: int = WITHIN_CASE_CHAR_BUDGET,
) -> Tuple[str, List[Document], str]:
    """Return (context_text, segment_documents, mode).

    mode is "read_whole" (segments == opinions) or "within_case" (segments == chunks).
    """
    record = case_index.get(case_title)
    if record is None:
        return "", [], "missing_case"

    if record["total_chars"] <= read_whole_budget:
        segments = [
            _opinion_to_document(case_title, record["citation"], op)
            for op in record["opinions"]
        ]
        mode = "read_whole"
    else:
        segments = _within_case_retrieve(
            case_title, record, question, vectorstore, within_case_budget
        )
        mode = "within_case"

    context = format_case_context(case_title, record["citation"], segments)
    return context, segments, mode


def format_case_context(case_title: str, citation: str, segments: List[Document]) -> str:
    parts = [f"CASE: {case_title}\nCITATION: {citation}"]
    for i, seg in enumerate(segments, start=1):
        m = seg.metadata
        role = m.get("effective_opinion_role") or m.get("opinion_role")
        author = m.get("opinion_author") or "unknown"
        parts.append(
            f"--- SEGMENT {i} | role={role} | author={author} ---\n{seg.page_content}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Generation (case-aware, with explicit num_ctx)
# ---------------------------------------------------------------------------

def _clean_generated_answer(answer: str) -> str:
    """
    Remove leaked model scratch/context if the local model accidentally emits it.
    """
    if not answer:
        return answer

    # Remove MiniMax / reasoning-style thinking tags if they appear.
    answer = re.sub(
        r"<mm:think>.*?</mm:think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    answer = answer.replace("</mm:think>", "").replace("<mm:think>", "")

    # Remove common Qwen/DeepSeek thinking tags if they appear.
    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    answer = answer.replace("</think>", "").replace("<think>", "")

    # If the model leaks source segments into the answer, keep only the part before them.
    leak_markers = [
        "\nSEGMENT 1 |",
        "\nSEGMENT 2 |",
        "\nSEGMENT 3 |",
        "\nSEGMENT 4 |",
        "\nSEGMENT 5 |",
        "\n--- SEGMENT 1",
        "\n--- SEGMENT 2",
        "\n--- SEGMENT 3",
        "\n--- SEGMENT 4",
        "\n--- SEGMENT 5",
    ]

    cut_at = len(answer)
    for marker in leak_markers:
        idx = answer.find(marker)
        if idx != -1:
            cut_at = min(cut_at, idx)

    return answer[:cut_at].strip()

def case_generate_answer(question: str, case_title: str, context: str) -> str:
    prompt = f"""
You are a legal research assistant analyzing a single U.S. Supreme Court case: {case_title}.

Answer the question using only the provided case material.

Rules:
- Use only the provided case material.
- Treat SEGMENT headers as metadata, but do not invent an author if author=unknown.
- Attribute reasoning to a named Justice only when the SEGMENT header identifies that author or the segment text itself clearly identifies that Justice.
- Distinguish the Court's holding from concurring, dissenting, or concurring-in-part/dissenting-in-part reasoning.
- Do not add notes about companion cases, separate opinions, or related cases unless the question asks for them or they are necessary to answer the question.
- If the available segments do not include enough material from a concurrence or dissent, say that rather than summarizing it from memory.
- Do not rely on outside legal knowledge beyond the provided material.
- If the material does not answer the question, say so plainly.
- For evaluation questions, answer directly and include the named doctrine, test, holding, or rule when the context supports it.
- If the question asks for a test, list the elements of the test.
- If the question asks why the Court ruled a certain way, give the Court's main reasons, not only the conclusion.
- Do not reverse the holding. If the Court struck down a law, do not describe it as valid.
- Do not mention any Justice, separate opinion, concurrence, or dissent unless that Justice or opinion appears in the provided case material.
- If a question asks about precedents, list only the precedents appearing in the provided case material.
- Do not speculate about missing opinions, omitted dissents, omitted concurrences, or Justices who may have joined them.
- If the provided material does not include a separate opinion, simply omit it rather than commenting on its absence.
- Be concise but legally precise.
- Do not reproduce the SEGMENT text.
- Do not include SEGMENT headers in the answer.
- Do not include hidden reasoning, scratchpad text, XML-like tags, or model thinking tags.
- Provide only the final answer.

Question:
{question}

Case material:
{context}

Answer:
""".strip()

    if BACKEND == "anthropic":
        return _clean_generated_answer(anthropic_generate(prompt))

    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0.1,
        },
    }

    timeout_seconds = int(os.getenv("OLLAMA_GENERATE_TIMEOUT", "1200"))
    max_attempts = int(os.getenv("OLLAMA_GENERATE_MAX_ATTEMPTS", "6"))
    last_error = None

    for attempt in range(max_attempts):
        try:
            response = requests.post(
                OLLAMA_GENERATE_ENDPOINT,
                json=payload,
                timeout=timeout_seconds,
            )
        except requests.exceptions.ReadTimeout as e:
            last_error = e
            wait_seconds = 10 + (attempt * 10)
            print(
                f"⚠️ Generation timed out after {timeout_seconds}s. "
                f"Waiting {wait_seconds}s before retry {attempt + 1}/{max_attempts}..."
            )
            time.sleep(wait_seconds)
            continue

        if response.status_code == 429:
            last_error = response
            wait_seconds = 10 + (attempt * 10)
            print(
                f"⚠️ 429 from generation model. "
                f"Waiting {wait_seconds}s before retry {attempt + 1}/{max_attempts}..."
            )
            time.sleep(wait_seconds)
            continue

        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return _clean_generated_answer(answer)

    if isinstance(last_error, requests.Response):
        last_error.raise_for_status()

    if last_error is not None:
        raise last_error

    raise RuntimeError("Generation failed without a response.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("🔎 Loading vector store...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        OllamaEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    case_index = build_case_index(vectorstore)
    print(f"✅ Loaded {len(case_index)} cases:")
    for title, rec in sorted(case_index.items()):
        print(f"   - {title} | {len(rec['opinions'])} opinion(s) | {rec['total_chars']} chars")

    print(
        "\nAsk a question. Prefix with 'case: <name>' to pin a case, "
        "or just ask and it will be resolved.\nType 'exit' to quit.\n"
    )

    pinned: Optional[str] = None
    while True:
        question = input("Question> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        explicit = pinned
        if question.lower().startswith("case:"):
            explicit = question.split(":", 1)[1].strip()
            print(f"📌 Pinned case: {explicit}")
            pinned = explicit
            continue

        case_title, method, _ = resolve_case(question, case_index, vectorstore, explicit=explicit)
        if not case_title:
            print("❌ Could not resolve a case for that question.\n")
            continue
        print(f"\n📂 Case: {case_title}  (resolved by {method})")

        context, segments, mode = build_case_scoped_context(
            case_title, case_index, question, vectorstore
        )
        print(f"📖 Mode: {mode} | {len(segments)} segment(s)")

        print("🤖 Generating answer...\n")
        answer = case_generate_answer(question, case_title, context)
        print("ANSWER")
        print("=" * 80)
        print(answer)

        print("\nSEGMENTS USED")
        print("-" * 80)
        for i, seg in enumerate(segments, start=1):
            m = seg.metadata
            role = m.get("effective_opinion_role") or m.get("opinion_role")
            print(f"[{i}] role={role} | author={m.get('opinion_author') or 'unknown'} | "
                  f"chunk={m.get('chunk_id')}")
        print()


if __name__ == "__main__":
    main()

