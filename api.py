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
        if request.method != "POST" or request.url.path != "/ask":
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
            content={"error": "out of scope", "detail": violation},
        )

    if _daily_cap_reached():
        return JSONResponse(
            status_code=503,
            content={"error": "daily limit reached"},
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
