"""
api.py

FastAPI wrapper around the existing SCOTUS RAG pipeline (query_demo_clean.py /
case_store.py). Does not reimplement retrieval or generation — it calls the
existing resolve_case() / build_case_scoped_context() / case_generate_answer()
/ generate_answer() / faithfulness.check_answer() functions and reshapes their
output into a JSON API.

Run with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import re
import secrets
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Deque, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from langchain_community.vectorstores import FAISS

load_dotenv()

META_SYSTEM_PROMPT = """You are a helpful assistant for SCOTUS Legal Aid, a portfolio
demonstration app built by Stephen Dubay, an Azure infrastructure
engineer repositioning toward AI engineering.

The app uses retrieval-augmented generation (RAG) to answer questions
about 6 landmark U.S. Supreme Court cases: Marbury v. Madison,
Brown v. Board of Education, Gideon v. Wainwright, Meyer v. Nebraska,
Miranda v. Arizona, and United States v. James Daniel Good Real Property.

When asked about the site, explain:
- What it is: a working demo of AI-powered legal research using RAG
- What it does: retrieves text from indexed Supreme Court opinions and
  uses an LLM to synthesize precise, citation-grounded answers
- What it does NOT do: give legal advice, cover all SCOTUS cases,
  or reflect law beyond these 6 opinions
- Who built it and why: Stephen Dubay, to demonstrate practical AI
  engineering skills including vector search, RAG pipelines,
  faithfulness checking, and API design

If asked a legal research question, respond with:
'That sounds like a legal research question — type it in the main
input to search the indexed cases.'

Keep answers concise and friendly."""

EXAMPLE_QUESTIONS = [
    "What liberty interest did Meyer v. Nebraska recognize?",
    "Why did the Court strike down the Nebraska statute?",
    "What right did Gideon v. Wainwright establish?",
    "What test did the Court apply in Miranda v. Arizona?",
]

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

import faithfulness
from query_demo_clean import (
    VECTORSTORE_PATH,
    BACKEND,
    ANTHROPIC_MODEL,
    OllamaEmbeddings,
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

ACCESS_KEY = os.getenv("ACCESS_KEY", "")
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() not in ("false", "0", "no")
RATE_LIMIT_ANON = int(os.getenv("RATE_LIMIT_ANON", "5"))
RATE_LIMIT_KEYED = int(os.getenv("RATE_LIMIT_KEYED", "50"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "3600"))
DAILY_COST_CAP = float(os.getenv("DAILY_COST_CAP", "2.00"))

# Rough $/million-token estimate used only for the daily cost guard below.
# Not exact billing — generate_answer()/case_generate_answer() return text
# only, not a usage object, and we are not rewriting them to expose one.
_MODEL_PRICE_PER_MTOK = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}
_DEFAULT_PRICE = {"input": 1.00, "output": 5.00}  # haiku-tier fallback


# ---------------------------------------------------------------------------
# In-memory rate limiting (single-process; resets on restart)
# ---------------------------------------------------------------------------

_rate_buckets: Dict[str, Deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"


def _check_rate_limit(ip: str, has_valid_key: bool) -> Optional[int]:
    """Return retry_after seconds if rate limited, else None (and record the hit)."""
    limit = RATE_LIMIT_KEYED if has_valid_key else RATE_LIMIT_ANON
    now = time.time()
    bucket = _rate_buckets[ip]

    while bucket and now - bucket[0] > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()

    if len(bucket) >= limit:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])) + 1
        return max(retry_after, 1)

    bucket.append(now)
    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-IP rate limiting, scoped to POST /ask only.

    access_key lives in the JSON body (not a header), so this middleware reads
    request.body() to peek it. Starlette caches the body bytes on the Request
    object, so the route handler's own body parsing later reuses them rather
    than re-reading the (already consumed) network stream.
    """

    async def dispatch(self, request: Request, call_next):
        if not RATE_LIMIT_ENABLED or request.method != "POST" or request.url.path != "/ask":
            return await call_next(request)

        access_key = None
        try:
            body = await request.body()
            if body:
                import json
                access_key = json.loads(body).get("access_key")
        except Exception:
            access_key = None  # malformed body; let the route's own validation handle it

        has_valid_key = bool(
            access_key and ACCESS_KEY and secrets.compare_digest(access_key, ACCESS_KEY)
        )

        retry_after = _check_rate_limit(_client_ip(request), has_valid_key)
        if retry_after is not None:
            return JSONResponse(
                status_code=429,
                content={"error": "rate limited", "retry_after": retry_after},
            )

        return await call_next(request)


# ---------------------------------------------------------------------------
# In-memory daily cost cap (single-process; resets at UTC midnight)
# ---------------------------------------------------------------------------

_cost_state = {"date": None, "total": 0.0}


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _reset_cost_if_new_day() -> None:
    today = _today_utc()
    if _cost_state["date"] != today:
        _cost_state["date"] = today
        _cost_state["total"] = 0.0


def _daily_cap_reached() -> bool:
    if BACKEND != "anthropic":
        return False  # Ollama generation is local/free; cap only guards API spend
    _reset_cost_if_new_day()
    return _cost_state["total"] >= DAILY_COST_CAP


def _record_estimated_cost(prompt_text: str, answer_text: str) -> None:
    if BACKEND != "anthropic":
        return
    _reset_cost_if_new_day()
    prices = _MODEL_PRICE_PER_MTOK.get(ANTHROPIC_MODEL, _DEFAULT_PRICE)
    # Heuristic ~4 chars/token estimate -- a safety guard, not a billing figure.
    input_tokens = len(prompt_text) / 4
    output_tokens = len(answer_text) / 4
    cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
    _cost_state["total"] += cost


# ---------------------------------------------------------------------------
# Topic / prompt-injection filter (heuristic, not ML-based)
# ---------------------------------------------------------------------------

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


def _topic_violation(question: str) -> Optional[str]:
    q = (question or "").strip()
    if not q:
        return "Question must not be empty."
    if _INJECTION_RE.search(q):
        return "Question appears to contain a prompt injection attempt."
    if _OFF_TOPIC_RE.search(q):
        return "Question appears unrelated to SCOTUS / legal research."
    return None


# ---------------------------------------------------------------------------
# Question classification + meta/overview handlers (about the app itself, or
# general case overviews, rather than precise retrieval from opinion text)
# ---------------------------------------------------------------------------

_meta_anthropic_client = None


def _get_meta_anthropic_client():
    global _meta_anthropic_client
    if _meta_anthropic_client is None:
        import anthropic
        _meta_anthropic_client = anthropic.Anthropic()
    return _meta_anthropic_client


_CLASSIFY_SYSTEM_PROMPT = (
    "Classify the following question as exactly one of these types:\n"
    "META — about this app, who built it, how RAG works, why it exists\n"
    "OVERVIEW — general question about one or more Supreme Court cases, "
    "asking for summaries, overviews, or case descriptions\n"
    "RESEARCH — specific legal question requiring precise retrieval "
    "from opinion text\n"
    "Reply with only the single word: META, OVERVIEW, or RESEARCH."
)

_VALID_CLASSES = ("META", "OVERVIEW", "RESEARCH")


def classify_question(question: str) -> str:
    """Single fast Haiku call replacing keyword-list routing. Defaults to
    RESEARCH (the existing, unchanged retrieval pipeline) on any failure or
    when BACKEND != "anthropic", so Ollama-only setups behave exactly as
    before this feature was added."""
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
        _record_estimated_cost(_CLASSIFY_SYSTEM_PROMPT + question, text)
        upper = text.upper()
        for cls in _VALID_CLASSES:
            if cls in upper:
                return cls
        return "RESEARCH"
    except Exception:
        return "RESEARCH"


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


def handle_meta(question: str) -> str:
    response = _get_meta_anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=512,
        system=META_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def handle_overview(question: str) -> str:
    response = _get_meta_anthropic_client().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=768,
        system=OVERVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Faithfulness reformatting (uses faithfulness.check_answer() output as-is)
# ---------------------------------------------------------------------------

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


def _sources_from_segments(case_title: str, segments: list) -> List[str]:
    sources = []
    for seg in segments:
        m = seg.metadata
        role = m.get("effective_opinion_role") or m.get("opinion_role") or "unknown"
        citation = m.get("citation") or ""
        title = m.get("case_title") or case_title or ""
        sources.append(f"{title} | {citation} | role={role}".strip(" |"))
    return sources


# ---------------------------------------------------------------------------
# App lifespan: load the vector store + case index once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        OllamaEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    case_index = build_case_index(vectorstore)
    app.state.vectorstore = vectorstore
    app.state.case_index = case_index
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    access_key: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    case_name: Optional[str]
    routing_method: str
    faithful: bool
    sources: List[str]
    flagged_reason: Optional[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request):
    return {"status": "ok", "cases_loaded": len(request.app.state.case_index)}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request):
    violation = _topic_violation(payload.question)
    if violation:
        return JSONResponse(
            status_code=400,
            content={"error": "out of scope", "detail": violation, "suggestions": EXAMPLE_QUESTIONS},
        )

    if _daily_cap_reached():
        return JSONResponse(
            status_code=503,
            content={"error": "daily limit reached"},
        )

    question_type = classify_question(payload.question)

    if question_type == "META":
        meta_answer = handle_meta(payload.question)
        _record_estimated_cost(META_SYSTEM_PROMPT + payload.question, meta_answer)
        return AskResponse(
            answer=meta_answer,
            case_name=None,
            routing_method="meta",
            faithful=True,
            sources=[],
            flagged_reason=None,
        )

    if question_type == "OVERVIEW":
        overview_answer = handle_overview(payload.question)
        _record_estimated_cost(OVERVIEW_SYSTEM_PROMPT + payload.question, overview_answer)
        return AskResponse(
            answer=overview_answer,
            case_name=None,
            routing_method="overview",
            faithful=True,
            sources=[],
            flagged_reason=None,
        )

    vectorstore = request.app.state.vectorstore
    case_index = request.app.state.case_index
    question = payload.question

    case_title, routing_method, _ = resolve_case(question, case_index, vectorstore)

    if case_title:
        context, segments, _mode = build_case_scoped_context(
            case_title, case_index, question, vectorstore
        )
        answer = case_generate_answer(question, case_title, context)
        sources = _sources_from_segments(case_title, segments)
    else:
        raw_docs = vectorstore.similarity_search_with_score(question, k=30)
        segments = rerank_docs(question, raw_docs)[:8]
        context = format_context(segments)
        answer = generate_answer(question, context)
        sources = _sources_from_segments("", segments)

    _record_estimated_cost(question + context, answer)

    faith = faithfulness.check_answer(
        answer=answer,
        context=context,
        segments=segments,
        case_title=case_title or "",
    )

    return AskResponse(
        answer=answer,
        case_name=case_title,
        routing_method=routing_method,
        faithful=not faith["flagged"],
        sources=sources,
        flagged_reason=_flagged_reason(faith),
    )
