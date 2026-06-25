"""
app.py

Standalone Streamlit app for SCOTUS Legal Aid. Calls the RAG pipeline
(query_demo_clean.py / case_store.py / faithfulness.py) directly -- no
api.py / FastAPI process required. Suitable for Streamlit Cloud, where the
process is single-file and secrets are injected as environment variables.

api.py is kept in the repo as reference architecture (FastAPI wrapper around
the same pipeline) but is not used by this file.

Required secrets (Streamlit Cloud: Settings -> Secrets, or local .env):
    ANTHROPIC_API_KEY
    ANTHROPIC_MODEL     (default: claude-haiku-4-5)
    ACCESS_KEY           (optional; raises the per-session question limit)
    DAILY_COST_CAP       (default: 2.00)

Run with:
    streamlit run app.py
"""

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_config(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return os.getenv(key, default)


# BACKEND and ANTHROPIC_MODEL are read via plain os.getenv() inside
# query_demo_clean.py at import time, and ANTHROPIC_API_KEY is read directly
# by the anthropic SDK's client constructor -- none of those know about
# st.secrets. Seed them into os.environ here, before query_demo_clean (or
# the anthropic client) is ever imported/constructed, so a value set only in
# Streamlit Cloud's secrets still reaches both.
for _key in ("BACKEND", "ANTHROPIC_MODEL", "ANTHROPIC_API_KEY"):
    _val = get_config(_key)
    if _val is not None:
        os.environ[_key] = str(_val)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import faithfulness
from query_demo_clean import (
    VECTORSTORE_PATH,
    BACKEND,
    ANTHROPIC_MODEL,
    HF_EMBED_MODEL,
    rerank_docs,
    format_context,
    generate_answer,
)
from case_store import (
    build_case_index,
    resolve_case,
    build_case_scoped_context,
    case_generate_answer,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ACCESS_KEY = get_config("ACCESS_KEY", "")
DAILY_COST_CAP = float(get_config("DAILY_COST_CAP", "2.00"))
COMPARATIVE_MODEL = get_config("COMPARATIVE_MODEL", "claude-sonnet-4-6")

SESSION_LIMIT_ANON = 20
SESSION_LIMIT_KEYED = 100

_MODEL_PRICE_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}
_DEFAULT_PRICE = {"input": 1.00, "output": 5.00}  # haiku-tier fallback

# Indexed case titles, as built by case_store.build_case_index(). Selecting a
# case in the sidebar biases routing via a "Regarding X:" prefix folded into
# the question text sent to resolve_case() -- a nudge, not a guarantee.
# CASES itself is derived from the live index at startup (see below), not
# hardcoded here, so newly ingested cases appear with no code change.
ANY_CASE = "Any / Let the system decide"

EXAMPLE_QUESTIONS = [
    "What liberty interest did Meyer v. Nebraska recognize?",
    "Why did the Court strike down the Nebraska statute?",
    "What right did Gideon v. Wainwright establish?",
    "What test did the Court apply in Miranda v. Arizona?",
]


# ---------------------------------------------------------------------------
# Cached pipeline resources (loaded once per server process)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading case index...")
def load_resources():
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL),
        allow_dangerous_deserialization=True,
    )
    case_index = build_case_index(vectorstore)
    return vectorstore, case_index


CASE_MANIFEST_PATH = Path("data/cases/case_manifest.json")  # mirrors ingest.py's MANIFEST_PATH


@st.cache_resource
def load_case_metadata() -> dict:
    """Read era/topics/decision_year out of the case manifest, keyed on the
    manifest's "title" field so it lines up with build_case_index()'s keys."""
    with open(CASE_MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    return {
        entry["title"]: {
            "era": entry.get("court_era"),
            "topics": entry.get("topics", []),
            "year": entry.get("decision_year"),
        }
        for entry in manifest
    }


def cases_by_topic(topic: str, metadata: dict) -> List[str]:
    return [title for title, meta in metadata.items() if topic in meta.get("topics", [])]


def cases_by_era(era: str, metadata: dict) -> List[str]:
    return [title for title, meta in metadata.items() if meta.get("era") == era]


@st.cache_resource
def get_cost_tracker() -> dict:
    """A single dict shared by every session in this server process, so
    DAILY_COST_CAP is one budget for the whole deployed app, not one per
    visitor. st.cache_resource with no args returns the same object on every
    call -- this is the process-wide-singleton pattern, the equivalent of
    api.py's module-level _cost_state dict."""
    return {"date": None, "total": 0.0}


# ---------------------------------------------------------------------------
# Topic / prompt-injection filter (heuristic, not ML-based)
# ---------------------------------------------------------------------------

import re

_INJECTION_PATTERNS = [
    r"ignore (all |the )?(previous|above|prior) instructions",
    r"disregard (all |the )?(previous|above|prior) instructions",
    r"reveal (your|the) (system )?prompt",
    r"print (your|the) (system )?prompt",
    r"you are now",
    r"act as (a|an) (?!attorney|lawyer|legal)",
    r"jailbreak",
    r"pretend (you are|to be)",
    r"developer mode",
    r"\bDAN\b",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

_OFF_TOPIC_DENYLIST = [
    "recipe", "weather forecast", "write a poem", "write code", "write a song",
    "tell me a joke", "sports score", "stock price", "translate this into",
    "write python", "write javascript", "homework help with math",
]
_OFF_TOPIC_RE = re.compile("|".join(re.escape(t) for t in _OFF_TOPIC_DENYLIST), re.IGNORECASE)


def topic_violation(question: str) -> Optional[str]:
    q = (question or "").strip()
    if not q:
        return "Question must not be empty."
    if _INJECTION_RE.search(q):
        return "Question appears to contain a prompt injection attempt."
    if _OFF_TOPIC_RE.search(q):
        return "Question appears unrelated to SCOTUS / legal research."
    return None


# ---------------------------------------------------------------------------
# Daily cost cap (process-wide via get_cost_tracker(): one shared budget for
# every visitor, not one per browser session)
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_cost_if_new_day(tracker: dict) -> None:
    today = _today_utc()
    if tracker["date"] != today:
        tracker["date"] = today
        tracker["total"] = 0.0


def daily_cap_reached() -> bool:
    if BACKEND != "anthropic":
        return False  # Ollama generation is local/free; cap only guards API spend
    tracker = get_cost_tracker()
    _reset_cost_if_new_day(tracker)
    return tracker["total"] >= DAILY_COST_CAP


def record_estimated_cost(prompt_text: str, answer_text: str, model: str = ANTHROPIC_MODEL) -> None:
    if BACKEND != "anthropic":
        return
    tracker = get_cost_tracker()
    _reset_cost_if_new_day(tracker)
    prices = _MODEL_PRICE_PER_MTOK.get(model, _DEFAULT_PRICE)
    # Heuristic ~4 chars/token estimate -- a safety guard, not a billing figure.
    input_tokens = len(prompt_text) / 4
    output_tokens = len(answer_text) / 4
    cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
    tracker["total"] += cost


# ---------------------------------------------------------------------------
# Question classification + meta/overview/research handlers
# ---------------------------------------------------------------------------

_meta_anthropic_client = None


def _get_meta_anthropic_client():
    global _meta_anthropic_client
    if _meta_anthropic_client is None:
        import anthropic
        _meta_anthropic_client = anthropic.Anthropic()
    return _meta_anthropic_client


META_SYSTEM_PROMPT = """You are a helpful assistant for SCOTUS Legal Aid, a working
portfolio application built by Stephen Dubay, an Azure cloud
infrastructure engineer repositioning toward AI engineering roles.

About this app:
- It uses retrieval-augmented generation (RAG) to answer precise
  questions about 6 landmark U.S. Supreme Court opinions
- The 6 indexed cases are: Marbury v. Madison (1803), Meyer v. Nebraska
  (1923), Brown v. Board of Education (1955), Gideon v. Wainwright (1963),
  Miranda v. Arizona (1966), and United States v. James Daniel Good Real
  Property (1993)
- It demonstrates: FAISS vector search, case-scoped retrieval,
  faithfulness checking, LLM-powered synthesis, and API design
- It is NOT legal advice, NOT a comprehensive legal database, and does
  NOT reflect current law beyond these 6 opinions
- Built to demonstrate practical AI engineering skills for career
  repositioning toward AI/ML roles

Answer questions about the app warmly and concisely. If asked a legal
research question, say: that sounds like a research question — type it
in the input box and the system will search the indexed opinions."""

CASE_SUMMARIES = {
    "Marbury v. Madison (1803)": (
        "Outgoing President John Adams appointed William Marbury as a justice "
        "of the peace, but Secretary of State James Madison, under incoming "
        "President Jefferson, withheld his commission. Marbury asked the "
        "Supreme Court to issue a writ of mandamus compelling delivery under "
        "the Judiciary Act of 1789. Chief Justice John Marshall held that "
        "Marbury had a right to the commission, but that the Court could not "
        "issue the writ because the provision of the Judiciary Act granting "
        "the Court original jurisdiction over such writs exceeded the "
        "jurisdiction Article III allows Congress to confer. In so ruling, "
        "Marshall established the principle of judicial review: federal "
        "courts have the power and duty to declare acts of Congress "
        "unconstitutional when they conflict with the Constitution. This "
        "founding decision cemented the judiciary's role as a coequal branch "
        "and is the cornerstone of American constitutional law."
    ),
    "Brown v. Board of Education (1955, Brown II)": (
        "In Brown I (1954), the Court held that racial segregation in public "
        "schools violates the Equal Protection Clause of the Fourteenth "
        "Amendment, overturning Plessy v. Ferguson's 'separate but equal' "
        "doctrine. Because of the scale and difficulty of dismantling "
        "segregated school systems, the Court withheld a remedy and instead "
        "asked for further argument on implementation. Brown II, decided in "
        "1955, supplied that remedy: it remanded enforcement to the federal "
        "district courts, which were directed to take such proceedings as "
        "needed to admit students to public schools on a racially "
        "nondiscriminatory basis 'with all deliberate speed.' The Court "
        "balanced the personal interest of plaintiffs in prompt admission "
        "against considerations of public interest and practical "
        "administrative problems, giving lower courts equitable discretion "
        "to weigh local conditions while still requiring good-faith, prompt, "
        "and reasonable progress toward full compliance."
    ),
    "Gideon v. Wainwright (1963)": (
        "Clarence Earl Gideon was charged with a felony in Florida state "
        "court and, too poor to afford a lawyer, was denied appointed counsel "
        "because Florida only required it in capital cases. Representing "
        "himself, he was convicted. From prison, Gideon filed a handwritten "
        "petition asking the Supreme Court to reconsider Betts v. Brady "
        "(1942), which had held that appointed counsel was not a fundamental "
        "right binding on the states. The Court unanimously overruled Betts, "
        "holding that the Sixth Amendment right to counsel is a fundamental "
        "right made obligatory on the states through the Fourteenth "
        "Amendment's Due Process Clause. Justice Black's opinion reasoned "
        "that a fair trial in an adversarial criminal justice system is "
        "impossible without the guiding hand of counsel, and reaffirmed that "
        "the right applies regardless of the defendant's ability to pay. "
        "Gideon was retried with a lawyer and acquitted."
    ),
    "Meyer v. Nebraska (1923)": (
        "Nebraska passed a statute prohibiting the teaching of any modern "
        "foreign language to students who had not yet completed the eighth "
        "grade, aiming to promote civic unity and English fluency in the "
        "aftermath of World War I. Robert Meyer, a teacher at a parochial "
        "school, was convicted for teaching German to a young student. The "
        "Supreme Court struck down the statute, holding that the liberty "
        "protected by the Fourteenth Amendment's Due Process Clause includes "
        "the right of teachers to teach and of parents to control the "
        "upbringing and education of their children, including engaging "
        "instructors to teach foreign languages. Justice McReynolds' opinion "
        "found the prohibition arbitrary and without reasonable relation to "
        "any legitimate state purpose, since mere knowledge of a foreign "
        "language could not reasonably be regarded as harmful. Meyer is a "
        "foundational substantive due process and parental-rights precedent."
    ),
    "Miranda v. Arizona (1966)": (
        "Ernesto Miranda was arrested and interrogated by police without "
        "being informed of his constitutional rights, and he signed a "
        "written confession used to convict him of kidnapping and rape. The "
        "Supreme Court held that the Fifth Amendment's protection against "
        "compelled self-incrimination requires that suspects in custodial "
        "interrogation be clearly informed, before questioning, of their "
        "right to remain silent, that anything said can be used against "
        "them, their right to an attorney, and that an attorney will be "
        "appointed if they cannot afford one. Absent these warnings (and a "
        "valid waiver), statements made during custodial interrogation are "
        "inadmissible. Chief Justice Warren's opinion emphasized the "
        "inherently coercive pressures of incommunicado police interrogation. "
        "The resulting 'Miranda warnings' became a standard, instantly "
        "recognizable feature of American policing and criminal procedure."
    ),
    "United States v. James Daniel Good Real Property (1993)": (
        "The federal government seized James Good's house and four acres "
        "under civil forfeiture law after he pleaded guilty to a drug "
        "offense, doing so without prior notice or a hearing, more than a "
        "year and a half after his conviction. The Supreme Court held that "
        "the Due Process Clause of the Fifth Amendment requires the "
        "government to provide notice and a meaningful opportunity to be "
        "heard before seizing real property subject to civil forfeiture, "
        "absent exigent circumstances. Applying the Mathews v. Eldridge "
        "balancing test, Justice Kennedy's opinion reasoned that real "
        "property, unlike contraband or movable assets, can be neither "
        "concealed nor destroyed, so the government's interest in immediate, "
        "unannounced seizure is minimal while the owner's interest in "
        "continued use and possession of the home is substantial. The Court "
        "found no exigency justifying the delay and departure from ordinary "
        "due process protections in this case."
    ),
}


def _case_summaries_block() -> str:
    return "\n\n".join(f"{title}\n{summary}" for title, summary in CASE_SUMMARIES.items())


OVERVIEW_SYSTEM_PROMPT = (
    "You are a helpful assistant for SCOTUS Legal Aid, a RAG demo over 6 "
    "indexed Supreme Court cases. Answer the user's question using ONLY the "
    "case summaries below as context. These are short overviews, not full "
    "opinion text -- if the user wants exact quotes, holdings, or detailed "
    "reasoning, tell them to ask a more specific question so the research "
    "pipeline can retrieve the actual opinion text. Keep the answer concise "
    "and clear.\n\nCASE SUMMARIES:\n" + _case_summaries_block()
)

_CLASSIFY_SYSTEM_PROMPT = (
    "You are a router for a legal research app about U.S. Supreme Court cases.\n"
    "Classify the user's question into exactly one of four categories:\n\n"
    "META — questions about the APP ITSELF: its purpose, who built it, how it \n"
    "works technically, what technology it uses, what its limitations are, or \n"
    "whether a specific case is covered. The answer describes the system, not \n"
    "legal content.\n"
    "Examples: \"why does this exist\", \"who built you\", \"what is RAG\", \n"
    "\"is Roe v Wade in your database\", \"what cases do you have\"\n\n"
    "OVERVIEW — questions that want ACTUAL CASE CONTENT delivered in summary \n"
    "form: asking to describe, explain, or summarize what one or more cases \n"
    "held, decided, or reasoned. The answer contains legal substance from the \n"
    "cases themselves.\n"
    "Examples: \"summarize all the cases\", \"tell me about Miranda\", \n"
    "\"give me an overview of each case\", \"describe what Marbury v Madison decided\"\n\n"
    "COMPARATIVE — questions asking to COMPARE, CONTRAST, or TRACE EVOLUTION \n"
    "across multiple cases, court eras, or time periods. The answer synthesizes \n"
    "patterns or changes across cases.\n"
    "Examples: \"how has Fourth Amendment doctrine evolved\", \"compare Warren \n"
    "Court to Rehnquist Court\", \"what themes connect Miranda Gideon and \n"
    "Escobedo\", \"how did the exclusionary rule develop over time\"\n\n"
    "RESEARCH — questions requiring SPECIFIC LEGAL TEXT retrieved from a \n"
    "particular opinion: specific holdings, tests, reasoning, dissents, or \n"
    "doctrine from one case.\n"
    "Examples: \"what balancing test did Good Real Property apply\", \"what did \n"
    "Justice White argue in Miranda\", \"what standard did Terry establish for \n"
    "a stop\"\n\n"
    "Tie-break rules:\n"
    "- META vs OVERVIEW: if the answer would contain actual legal holdings or \n"
    "case reasoning, choose OVERVIEW\n"
    "- OVERVIEW vs COMPARATIVE: if multiple cases or time periods are implied, \n"
    "choose COMPARATIVE\n"
    "- COMPARATIVE vs RESEARCH: if asking about trends or evolution rather \n"
    "than a specific holding, choose COMPARATIVE\n"
    "- When genuinely uncertain, prefer RESEARCH\n\n"
    "Reply with exactly one word: META, OVERVIEW, RESEARCH, or COMPARATIVE."
)

_VALID_CLASSES = ("META", "OVERVIEW", "RESEARCH", "COMPARATIVE")

_COMPARATIVE_SELECT_SYSTEM_PROMPT = (
    "You are helping a legal research system identify relevant cases. "
    "Given a question and a list of indexed cases with their topics and eras, "
    "return a JSON list of the 3-6 most relevant case titles. "
    "Return only a JSON array of strings, nothing else."
)

_COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a legal research assistant analyzing how Supreme Court doctrine "
    "has evolved across multiple cases. Use only the provided case excerpts "
    "to support your analysis. Organize your answer chronologically where "
    "relevant. Cite specific cases by name when making claims about doctrine."
)


def _format_case_metadata_block(case_metadata: dict) -> str:
    lines = []
    for title, meta in case_metadata.items():
        topics = ", ".join(meta.get("topics", []))
        lines.append(f"- {title} ({meta.get('year', 'n.d.')}, {meta.get('era', 'unknown era')}): {topics}")
    return "\n".join(lines)


def _comparative_case_block(case_title: str, case_metadata: dict, segments: list) -> str:
    meta = case_metadata.get(case_title, {})
    header = f"=== {case_title} ({meta.get('year', 'n.d.')}, {meta.get('era', 'unknown era')}) ==="
    body = "\n\n".join(seg.page_content for seg in segments)
    return f"{header}\n{body}"


def classify_question(question: str) -> str:
    """Single fast Haiku call. Defaults to RESEARCH (the unchanged retrieval
    pipeline) on any failure or when BACKEND != "anthropic"."""
    if BACKEND != "anthropic":
        return "RESEARCH"
    try:
        response = _get_meta_anthropic_client().messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=10,
            system=_CLASSIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        record_estimated_cost(_CLASSIFY_SYSTEM_PROMPT + question, text)
        upper = text.upper()
        for cls in _VALID_CLASSES:
            if cls in upper:
                return cls
        return "RESEARCH"
    except Exception:
        return "RESEARCH"


def _flagged_reason(faith: dict) -> Optional[str]:
    if not faith["flagged"]:
        return None
    parts = []
    if faith["unsupported_justices"]:
        parts.append("unsupported justices: " + ", ".join(faith["unsupported_justices"]))
    if faith["unsupported_cases"]:
        parts.append("unsupported cases: " + ", ".join(faith["unsupported_cases"]))
    if faith["role_conflicts"]:
        parts.append("role conflicts: " + "; ".join(faith["role_conflicts"]))
    return " | ".join(parts) if parts else None


def _sources_from_segments(case_title: str, segments: list) -> list:
    sources = []
    for seg in segments:
        m = seg.metadata
        role = m.get("effective_opinion_role") or m.get("opinion_role") or "unknown"
        citation = m.get("citation") or ""
        title = m.get("case_title") or case_title or ""
        sources.append(f"{title} | {citation} | role={role}".strip(" |"))
    return sources


def handle_meta(question: str) -> dict:
    response = _get_meta_anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=META_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    answer = "".join(b.text for b in response.content if b.type == "text").strip()
    record_estimated_cost(META_SYSTEM_PROMPT + question, answer)
    return {
        "answer": answer, "case_name": None, "routing_method": "meta",
        "faithful": True, "sources": [], "flagged_reason": None,
    }


def handle_overview(question: str) -> dict:
    response = _get_meta_anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=768,
        system=OVERVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    answer = "".join(b.text for b in response.content if b.type == "text").strip()
    record_estimated_cost(OVERVIEW_SYSTEM_PROMPT + question, answer)
    return {
        "answer": answer, "case_name": None, "routing_method": "overview",
        "faithful": True, "sources": [], "flagged_reason": None,
    }


def handle_research(question: str, case_hint: Optional[str] = None) -> dict:
    vectorstore, case_index = load_resources()
    search_question = f"Regarding {case_hint}: {question}" if case_hint else question

    case_title, routing_method, _ = resolve_case(search_question, case_index, vectorstore)

    if case_title:
        context, segments, _mode = build_case_scoped_context(
            case_title, case_index, search_question, vectorstore
        )
        answer = case_generate_answer(search_question, case_title, context)
        sources = _sources_from_segments(case_title, segments)
    else:
        raw_docs = vectorstore.similarity_search_with_score(search_question, k=30)
        segments = rerank_docs(search_question, raw_docs)[:8]
        context = format_context(segments)
        answer = generate_answer(search_question, context)
        sources = _sources_from_segments("", segments)

    record_estimated_cost(search_question + context, answer)

    faith = faithfulness.check_answer(
        answer=answer, context=context, segments=segments, case_title=case_title or "",
    )

    return {
        "answer": answer,
        "case_name": case_title,
        "routing_method": routing_method,
        "faithful": not faith["flagged"],
        "sources": sources,
        "flagged_reason": _flagged_reason(faith),
    }


def handle_comparative(question: str, case_metadata: dict) -> dict:
    vectorstore, case_index = load_resources()

    # 4a -- identify relevant cases with a fast Haiku call.
    select_user_msg = f"Question: {question}\n\nIndexed cases:\n{_format_case_metadata_block(case_metadata)}"
    select_response = _get_meta_anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=256,
        system=_COMPARATIVE_SELECT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": select_user_msg}],
    )
    raw = "".join(b.text for b in select_response.content if b.type == "text").strip()
    record_estimated_cost(_COMPARATIVE_SELECT_SYSTEM_PROMPT + select_user_msg, raw, model=ANTHROPIC_MODEL)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned[4:] if cleaned.lower().startswith("json") else cleaned

    try:
        relevant_cases = json.loads(cleaned)
        if not isinstance(relevant_cases, list) or not all(isinstance(c, str) for c in relevant_cases):
            raise ValueError("response was not a JSON array of strings")
    except Exception:
        relevant_cases = list(case_metadata.keys())

    relevant_cases = [c for c in relevant_cases if c in case_index] or list(case_metadata.keys())

    # 4b -- multi-case retrieval, up to 3 segments per case.
    blocks = []
    sources = []
    for case_title in relevant_cases:
        record = case_index.get(case_title)
        if record is None:
            continue
        _context, segments, _mode = build_case_scoped_context(case_title, case_index, question, vectorstore)
        segments = segments[:3]
        if not segments:
            continue
        blocks.append(_comparative_case_block(case_title, case_metadata, segments))
        meta = case_metadata.get(case_title, {})
        sources.append(f"{case_title} ({meta.get('era', 'unknown era')}, {meta.get('year', 'n.d.')})")

    combined_context = "\n\n".join(blocks)

    # 4c -- synthesis with COMPARATIVE_MODEL.
    synthesis_user_msg = f"Question: {question}\n\nCase excerpts:\n{combined_context}"
    synthesis_response = _get_meta_anthropic_client().messages.create(
        model=COMPARATIVE_MODEL,
        max_tokens=1536,
        system=_COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": synthesis_user_msg}],
    )
    answer = "".join(b.text for b in synthesis_response.content if b.type == "text").strip()
    record_estimated_cost(_COMPARATIVE_SYNTHESIS_SYSTEM_PROMPT + synthesis_user_msg, answer, model=COMPARATIVE_MODEL)

    return {
        "answer": answer,
        "case_name": None,
        "routing_method": "comparative",
        "faithful": True,
        "sources": sources,
        "flagged_reason": None,
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_error" not in st.session_state:
    st.session_state.last_error = None
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "pending_case_reset" not in st.session_state:
    st.session_state.pending_case_reset = False

# Apply any pending case-selector reset BEFORE the selectbox below is
# instantiated this run -- Streamlit forbids writing to a widget-bound
# session_state key in the same run where that widget already rendered, so
# the reset is requested on one run (see the rerun() below) and applied here,
# on the next run, before the widget exists yet.
if st.session_state.pending_case_reset:
    st.session_state.selected_case = ANY_CASE
    st.session_state.pending_case_reset = False

st.set_page_config(page_title="SCOTUS Legal Aid", page_icon="⚖️")

st.title("SCOTUS Legal Aid")
st.caption(
    "Retrieval-augmented Q&A over a small, fixed set of indexed U.S. Supreme Court opinions."
)
st.caption(
    "Not legal advice, not a comprehensive legal database, and not a reflection of current law beyond these opinions."
)

col1, col2 = st.columns(2)
with col1:
    st.caption("In scope, e.g.:")
    st.markdown(
        "- What liberty interest did *Meyer v. Nebraska* recognize?\n"
        "- Why did the Court strike down the Nebraska statute?\n"
        "- What right did *Gideon v. Wainwright* establish?\n"
        "- What test did the Court apply in *Miranda v. Arizona*?"
    )
with col2:
    st.caption("Out of scope, e.g.:")
    st.markdown(
        "- Current or pending Supreme Court cases\n"
        "- General legal advice for your own situation\n"
        "- Cases outside the 6 indexed opinions\n"
        "- Non-legal questions"
    )

_vectorstore, _case_index = load_resources()
CASES = sorted(_case_index.keys())

with st.sidebar:
    st.subheader("Options")
    selected_case = st.selectbox("Case", [ANY_CASE] + CASES, key="selected_case")
    access_key = st.text_input("Access key (optional)", type="password")

with st.form("ask_form"):
    question = st.text_input("Your question")
    submitted = st.form_submit_button("Ask")

if submitted:
    st.session_state.last_response = None
    st.session_state.last_error = None
    st.session_state.suggested_questions = []

    if not question.strip():
        st.session_state.last_error = "Please enter a question."
    else:
        sent_question = question.strip()
        key_input = access_key.strip()
        has_valid_key = bool(key_input and ACCESS_KEY and secrets.compare_digest(key_input, ACCESS_KEY))
        limit = SESSION_LIMIT_KEYED if has_valid_key else SESSION_LIMIT_ANON

        if st.session_state.question_count >= limit:
            st.session_state.last_error = (
                f"Session limit reached ({limit} questions). Please try again later."
            )
        else:
            st.session_state.question_count += 1

            violation = topic_violation(sent_question)
            if violation:
                st.session_state.last_error = f"Out of scope: {violation}"
                st.session_state.suggested_questions = EXAMPLE_QUESTIONS
            elif daily_cap_reached():
                st.session_state.last_error = "Daily usage limit reached. Try again tomorrow."
            else:
                with st.spinner("Researching..."):
                    try:
                        case_metadata = load_case_metadata()

                        question_type = classify_question(sent_question)
                        if question_type == "META":
                            data = handle_meta(sent_question)
                        elif question_type == "OVERVIEW":
                            data = handle_overview(sent_question)
                        elif question_type == "COMPARATIVE":
                            data = handle_comparative(sent_question, case_metadata)
                        else:
                            case_hint = selected_case if selected_case != ANY_CASE else None
                            data = handle_research(sent_question, case_hint)
                    except Exception as e:
                        st.session_state.last_error = f"Something went wrong: {e}"
                    else:
                        st.session_state.last_response = data
                        st.session_state.pending_case_reset = True

if st.session_state.pending_case_reset:
    st.rerun()

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    if st.session_state.suggested_questions:
        st.write("Try one of these instead:")
        for q in st.session_state.suggested_questions:
            st.write(f"- {q}")

if st.session_state.last_response:
    data = st.session_state.last_response

    case_name = data.get("case_name")
    routing_method = data.get("routing_method") or "unknown"

    if routing_method == "meta":
        st.caption("ℹ️ About this app")
    elif routing_method == "overview":
        st.caption("ℹ️ Case overview")
    elif routing_method == "comparative":
        st.caption("Cross-case analysis")
    else:
        st.caption(f"Case: {case_name or 'Unresolved'} | Routing: {routing_method}")
        if not case_name:
            st.info(
                "This question didn't match a specific indexed case. "
                "The answer below is based on a broader search across the corpus "
                "and may be less precise."
            )

    if data.get("faithful") is False:
        reason = data.get("flagged_reason") or "Potential faithfulness issue detected."
        st.warning(reason)

    st.write(data.get("answer", ""))

    sources = data.get("sources") or []
    deduped_sources = list(dict.fromkeys(sources))
    with st.expander(f"Sources ({len(deduped_sources)})"):
        if deduped_sources:
            for src in deduped_sources:
                st.write(f"- {src}")
        else:
            st.write("No sources returned.")
