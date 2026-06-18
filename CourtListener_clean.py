"""
CourtListener_clean.py

Purpose:
  Build a small Supreme Court RAG demo corpus from CourtListener using a case-dossier-first model.

What this script does:
  1. Searches CourtListener for a curated list of demo SCOTUS cases.
  2. Builds a normalized top-level case dossier for each case.
  3. Fetches full opinion text from the authenticated CourtListener opinions API.
  4. Converts each opinion into LangChain Documents with canonical metadata.
  5. Chunks the documents using legal-aware separators.
  6. Embeds the chunks with local Ollama nomic-embed-text.
  7. Saves a local FAISS vector store and JSONL exports.

Required environment:
  $env:COURTLISTENER_TOKEN = "<token>"

Required local models:
  ollama pull nomic-embed-text
"""
import time
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

COURTLISTENER_SEARCH_URL = "https://www.courtlistener.com/api/rest/v4/search/"
COURTLISTENER_OPINION_URL_TEMPLATE = "https://www.courtlistener.com/api/rest/v4/opinions/{opinion_id}/"
COURTLISTENER_BASE_URL = "https://www.courtlistener.com"

COURTLISTENER_REQUEST_DELAY_SECONDS = float(
    os.getenv("COURTLISTENER_REQUEST_DELAY_SECONDS", "30")
)

COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN")

HEADERS = {
    "User-Agent": "SupremeCourtLegalAidDemo/0.1",
    "Accept": "application/json",
}

if COURTLISTENER_TOKEN:
    HEADERS["Authorization"] = f"Token {COURTLISTENER_TOKEN}"

DEMO_QUERIES = [
    "Meyer v. Nebraska 262 U.S. 390",
    "United States v. James Daniel Good Real Property 510 U.S. 43",
    "Brown v. Board of Education 347 U.S. 483",
    "Miranda v. Arizona 384 U.S. 436",
    "Gideon v. Wainwright 372 U.S. 335",
    "Marbury v. Madison 5 U.S. 137",
]

OUTPUT_DIR = Path("data")
DOSSIER_DIR = OUTPUT_DIR / "dossiers"
DOCUMENT_DIR = OUTPUT_DIR / "documents"
CHUNK_DIR = OUTPUT_DIR / "chunks"
VECTOR_DIR = OUTPUT_DIR / "vectors" / "rag_vectorstore_courtlistener_demo"
EXPORT_DIR = OUTPUT_DIR / "exports"

EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_ENDPOINT = os.getenv("OLLAMA_EMBED_ENDPOINT", "http://localhost:11434/api/embed")

# =============================================================================
# Utilities
# =============================================================================


def ensure_dirs() -> None:
    for path in [DOSSIER_DIR, DOCUMENT_DIR, CHUNK_DIR, VECTOR_DIR.parent, EXPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def clean_html(value: str) -> str:
    if not value:
        return ""
    return BeautifulSoup(value, "html.parser").get_text("\n", strip=True)


def make_source_url(absolute_url: Optional[str]) -> str:
    if not absolute_url:
        return ""
    return f"{COURTLISTENER_BASE_URL}{absolute_url}" if absolute_url.startswith("/") else absolute_url


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def safe_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"

# =============================================================================
# Ollama embeddings
# =============================================================================


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

        raise RuntimeError(f"Ollama embedding response did not contain embeddings: {data.keys()}")

# =============================================================================
# CourtListener ingestion
# =============================================================================

def get_with_retry(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 90,
    retries: int = 3,
) -> requests.Response:
    last_error = None

    for attempt in range(1, retries + 1):
        if attempt > 1:
            sleep_seconds = COURTLISTENER_REQUEST_DELAY_SECONDS * attempt
            logger.info("Waiting %.1f seconds before retry...", sleep_seconds)
            time.sleep(sleep_seconds)

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=timeout,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = (
                    int(retry_after)
                    if retry_after and retry_after.isdigit()
                    else COURTLISTENER_REQUEST_DELAY_SECONDS * attempt
                )

                logger.warning(
                    "Rate limited by CourtListener. Waiting %.1f seconds...",
                    wait_seconds,
                )
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            time.sleep(COURTLISTENER_REQUEST_DELAY_SECONDS)
            return response

        except requests.exceptions.RequestException as exc:
            last_error = exc
            logger.warning(
                "Request failed attempt %s/%s for %s: %s",
                attempt,
                retries,
                url,
                exc,
            )

    raise RuntimeError(
        f"Request failed after {retries} attempts for {url}: {last_error}"
    )

def search_courtlistener_case(query: str) -> Dict[str, Any]:
    params = {
        "q": query,
        "type": "o",
        "court": "scotus",
        "order_by": "score desc",
    }

    response = get_with_retry(
        COURTLISTENER_SEARCH_URL,
        params=params,
        timeout=90,
        retries=3,
    )
    data = response.json()
    results = data.get("results", [])

    if not results:
        raise RuntimeError(f"No CourtListener results found for query: {query}")

    return results[0]


def build_case_dossier(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    opinions = result.get("opinions", []) or []
    citations = [str(c) for c in as_list(result.get("citation")) if c]
    cluster_id = result.get("cluster_id")
    source_url = make_source_url(result.get("absolute_url"))

    return {
        "case_dossier_id": f"courtlistener_cluster_{cluster_id}",
        "source_system": "CourtListener",
        "source_url": source_url,
        "query": query,
        "case_title": result.get("caseName") or query,
        "case_title_full": result.get("caseNameFull") or result.get("caseName") or query,
        "court": result.get("court"),
        "court_id": result.get("court_id"),
        "citations": citations,
        "cluster_id": cluster_id,
        "docket_id": result.get("docket_id"),
        "docket_number": result.get("docketNumber"),
        "date_filed": result.get("dateFiled"),
        "date_argued": result.get("dateArgued"),
        "scdb_id": result.get("scdb_id"),
        "cite_count": result.get("citeCount"),
        "judge": result.get("judge"),
        "attorney": result.get("attorney"),
        "procedural_history": result.get("procedural_history"),
        "posture": result.get("posture"),
        "status": result.get("status"),
        "opinion_ids": [op.get("id") for op in opinions if op.get("id")],
        "opinion_records": opinions,
        "components": {
            "opinions": "seeded",
            "briefs": "pending",
            "oral_arguments": "pending",
            "docket_entries": "pending",
            "lower_court_history": "pending",
            "party_records": "pending",
            "attorney_records": "partial",
            "citation_network": "partial",
        },
    }


def fetch_opinion_text(opinion_id: int) -> Dict[str, Any]:
    api_url = COURTLISTENER_OPINION_URL_TEMPLATE.format(opinion_id=opinion_id)
    response = get_with_retry(
        api_url,
        timeout=90,
        retries=3,
    )
    data = response.json()

    print("\n===== OPINION RECORD =====")
    print(f"id: {data.get('id')}")
    print(f"type: {data.get('type')}")
    print(f"author_str: {data.get('author_str')}")
    print(f"per_curiam: {data.get('per_curiam')}")

    print("\n===== AVAILABLE TEXT FIELDS =====")
    for key in data.keys():
        if "html" in key.lower() or "text" in key.lower():
            value = data.get(key)
            if value:
                print(f"{key}: {len(str(value))} chars")

    preferred_fields = [
        "plain_text",
        "html_with_citations",
        "html",
        "html_lawbox",
        "html_columbia",
        "html_anon_2020",
    ]

    for field in preferred_fields:
        value = data.get(field)
        if value:
            text = clean_html(str(value)).strip()
            if len(text) > 1000:
                data["selected_text_field"] = field
                data["selected_text"] = text
                return data

    data["selected_text_field"] = None
    data["selected_text"] = ""
    return data

def split_opinion_sections(text: str) -> List[Dict[str, Any]]:
    """
    Split a combined Supreme Court opinion into role-level sections.

    This is generic SCOTUS structure parsing, not case-specific filtering.
    It looks for common opinion markers such as:
      - delivered the opinion of the Court
      - concurring
      - dissenting
      - concurring in part and dissenting in part
    """

    marker_pattern = re.compile(
    r"""
    (?m)
    ^\s*
    (?P<label>
        (?:
            (?:JUSTICE|Justice|CHIEF\s+JUSTICE|Chief\s+Justice)\s+
            [A-Z][A-Za-z'\-]+
            .*?
            (?:
                delivered\s+the\s+opinion\s+of\s+the\s+Court|
                concurring\s+in\s+part\s+and\s+dissenting\s+in\s+part|
                concurring\s+in\s+the\s+judgment|
                concurring|
                dissenting
            )
        )
        |
        (?:
            [A-Z][A-Za-z'\-]+,\s+J\.,\s+
            (?:
                concurring\s+in\s+part\s+and\s+dissenting\s+in\s+part|
                concurring\s+in\s+the\s+judgment|
                concurring|
                dissenting
            )
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

    matches = list(marker_pattern.finditer(text))

    if not matches:
        return [
            {
                "opinion_role": "court_opinion",
                "section_label": "unsplit_opinion",
                "section_order": 0,
                "text": text,
            }
        ]

    sections = []

    # Text before the first separate opinion marker is treated as the court/majority opinion.
    first_start = matches[0].start()

    if first_start > 500:
        sections.append(
            {
                "opinion_role": "court_opinion",
                "section_label": "opinion_before_separate_writings",
                "section_order": 0,
                "text": text[:first_start].strip(),
            }
        )

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        label = match.group("label").strip()
        section_text = text[start:end].strip()
        label_lower = label.lower()

        if "concurring in part and dissenting in part" in label_lower:
            role = "concurrence_dissent"
        elif "dissent" in label_lower:
            role = "dissent"
        elif "concurring in the judgment" in label_lower:
            role = "concurrence_in_judgment"
        elif "concur" in label_lower:
            role = "concurrence"
        elif "delivered the opinion of the court" in label_lower:
            # FIX (#4): the majority author marker was previously falling through
            # to "unknown". Label it explicitly as the court opinion.
            role = "court_opinion"
        else:
            role = "court_opinion"

        if len(section_text) > 500:
            sections.append(
                {
                    "opinion_role": role,
                    "section_label": label,
                    "section_order": len(sections),
                    "text": section_text,
                }
            )

    return sections

def build_opinion_documents(result: Dict[str, Any], query: str, dossier: Optional[Dict[str, Any]] = None) -> List[Document]:
    # FIX: accept a prebuilt dossier to avoid building it twice per case
    # (fetch_demo_documents already builds it). Falls back to building one
    # if called standalone.
    if dossier is None:
        dossier = build_case_dossier(result, query)
    documents: List[Document] = []

    for index, opinion_record in enumerate(dossier["opinion_records"]):
        opinion_id = opinion_record.get("id")
        if not opinion_id:
            continue

        try:
            print(f"📥 Fetching opinion API: {COURTLISTENER_OPINION_URL_TEMPLATE.format(opinion_id=opinion_id)}")
            opinion_data = fetch_opinion_text(opinion_id)
        except Exception as exc:
            logger.warning("Failed to fetch opinion %s for %s: %s", opinion_id, dossier["case_title"], exc)
            opinion_data = {"selected_text": "", "selected_text_field": None}

        text = opinion_data.get("selected_text") or clean_html(str(opinion_record.get("snippet") or ""))
        if not text:
            logger.warning("No usable text for opinion %s in %s", opinion_id, dossier["case_title"])
            continue

        source_url = make_source_url(opinion_data.get("absolute_url") or result.get("absolute_url"))

        sections = split_opinion_sections(text)

        print(
            f"🧩 Split opinion {opinion_id} into "
            f"{len(sections)} role-level section document(s)"
        )

        for section in sections:
            metadata = canonical_metadata(
                dossier=dossier,
                source_url=source_url,
                document_type="supreme_court_opinion",
                opinion_role=section["opinion_role"],
                opinion_id=opinion_id,
                opinion_author=opinion_data.get("author_str") or opinion_record.get("author_id"),
                text_length=len(section["text"]),
                selected_text_field=opinion_data.get("selected_text_field"),
                section_label=section["section_label"],
                section_order=section["section_order"],
            )

            documents.append(
                Document(
                    page_content=section["text"],
                    metadata=metadata,
                )
            )

    if not documents:
        fallback_text = "\n\n".join(
            clean_html(str(x))
            for x in [
                result.get("caseName"),
                result.get("caseNameFull"),
                result.get("citation"),
                result.get("attorney"),
                result.get("procedural_history"),
                result.get("posture"),
                result.get("syllabus"),
            ]
            if x
        ).strip()

        if fallback_text:
            documents.append(
                Document(
                    page_content=fallback_text,
                    metadata=canonical_metadata(
                        dossier=dossier,
                        source_url=dossier["source_url"],
                        document_type="case_dossier_seed",
                        opinion_role="metadata_only",
                        text_length=len(fallback_text),
                        section_label="metadata_fallback",
                        section_order=0,
                    ),
                )
            )

    return documents

def canonical_metadata(
    dossier: Dict[str, Any],
    source_url: str,
    document_type: str,
    opinion_role: str = "unknown",
    opinion_id: Optional[int] = None,
    opinion_author: Optional[Any] = None,
    text_length: Optional[int] = None,
    selected_text_field: Optional[str] = None,
        section_label: Optional[str] = None,
        section_order: Optional[int] = None,
        ) -> Dict[str, Any]:
    return {
        "case_dossier_id": dossier["case_dossier_id"],
        "source_system": dossier["source_system"],
        "source": source_url,
        "case_title": dossier["case_title"],
        "case_title_full": dossier["case_title_full"],
        "court": dossier["court"],
        "court_id": dossier["court_id"],
        "citation": ", ".join(dossier["citations"]),
        "cluster_id": dossier["cluster_id"],
        "docket_id": dossier["docket_id"],
        "docket_number": dossier["docket_number"],
        "date_filed": dossier["date_filed"],
        "date_argued": dossier["date_argued"],
        "scdb_id": dossier["scdb_id"],
        "cite_count": dossier["cite_count"],
        "document_type": document_type,
        "opinion_role": opinion_role,
        "opinion_id": opinion_id,
        "opinion_author": str(opinion_author) if opinion_author else "",
        "section_label": section_label or "",
        "section_order": section_order,
        "text_length": text_length,
        "selected_text_field": selected_text_field or "",
    }

# =============================================================================
# Chunking and exports
# =============================================================================


def create_legal_document_splitter() -> RecursiveCharacterTextSplitter:
    # NOTE: role-level splitting already happens upstream in
    # split_opinion_sections(), so each document here is a single
    # opinion role. The uppercase headers below ("OPINION OF THE COURT",
    # etc.) rarely appear verbatim in CourtListener plain_text, which uses
    # "JUSTICE X delivered the opinion of the Court"-style lines instead;
    # in practice splitting falls back to paragraph/sentence boundaries.
    # They are kept as harmless higher-priority anchors for sources that
    # do use them.
    legal_separators = [
        "\n\nSYLLABUS\n\n",
        "\n\nOPINION OF THE COURT\n\n",
        "\n\nCONCURRING OPINION\n\n",
        "\n\nDISSENTING OPINION\n\n",
        "\n\nI\n\n",
        "\n\nII\n\n",
        "\n\nIII\n\n",
        "\n\n",
        ". ",
        " ",
        "",
    ]

    return RecursiveCharacterTextSplitter(
        separators=legal_separators,
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )


def generate_chunk_metadata(doc: Document, chunk_idx: int, total_chunks: int) -> Dict[str, Any]:
    metadata = dict(doc.metadata)
    metadata["chunk_id"] = chunk_idx
    metadata["total_chunks"] = total_chunks
    metadata["is_first_chunk"] = chunk_idx == 0
    metadata["is_last_chunk"] = chunk_idx == total_chunks - 1
    return metadata


def smart_legal_chunking(documents: List[Document]) -> List[Document]:
    splitter = create_legal_document_splitter()
    chunked_docs: List[Document] = []

    for doc in documents:
        chunks = splitter.split_text(doc.page_content)
        total_chunks = len(chunks)
        for i, chunk_text in enumerate(chunks):
            chunked_docs.append(
                Document(page_content=chunk_text, metadata=generate_chunk_metadata(doc, i, total_chunks))
            )

    return chunked_docs


def export_jsonl(items: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def export_documents(documents: List[Document], output_path: Path) -> None:
    rows = [{"page_content": d.page_content, "metadata": d.metadata} for d in documents]
    export_jsonl(rows, output_path)

# =============================================================================
# Main
# =============================================================================


def fetch_demo_documents() -> tuple[List[Document], List[Dict[str, Any]]]:
    documents: List[Document] = []
    dossiers: List[Dict[str, Any]] = []

    for query in DEMO_QUERIES:
        print(f"🔎 Searching CourtListener: {query}")
        result = search_courtlistener_case(query)
        dossier = build_case_dossier(result, query)
        dossiers.append(dossier)

        case_docs = build_opinion_documents(result, query, dossier=dossier)
        documents.extend(case_docs)

        total_chars = sum(len(d.page_content) for d in case_docs)
        print(f"✅ Found: {dossier['case_title']} | documents={len(case_docs)} | chars={total_chars}")

    return documents, dossiers


def main() -> None:
    ensure_dirs()

    if COURTLISTENER_TOKEN:
        print("🔑 CourtListener token loaded")
    else:
        print("⚠️ No CourtListener token found in environment")

    print("🚀 Starting CourtListener demo ingestion")

    documents, dossiers = fetch_demo_documents()
    print(f"📄 Retrieved {len(documents)} CourtListener source documents")

    if not documents:
        raise RuntimeError("No CourtListener documents were collected.")

    export_jsonl(dossiers, EXPORT_DIR / "case_dossiers_demo.jsonl")
    for dossier in dossiers:
        output_file = DOSSIER_DIR / f"{safe_filename(dossier['case_dossier_id'])}.json"
        output_file.write_text(json.dumps(dossier, ensure_ascii=False, indent=2), encoding="utf-8")

    export_documents(documents, EXPORT_DIR / "source_documents_demo.jsonl")

    print("🔪 Performing smart legal-aware chunking...")
    chunks = smart_legal_chunking(documents)

    if not chunks:
        raise RuntimeError("Documents were collected, but chunking produced zero chunks.")

    export_documents(chunks, EXPORT_DIR / "chunks_demo.jsonl")
    print(f"📦 Prepared {len(chunks)} chunks for embedding.")

    print("🔐 Creating vector store with Ollama embeddings...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_DIR))

    print(f"✅ Vector store saved as '{VECTOR_DIR}'")
    print(f"✅ Exported dossiers to '{EXPORT_DIR / 'case_dossiers_demo.jsonl'}'")
    print(f"✅ Exported source documents to '{EXPORT_DIR / 'source_documents_demo.jsonl'}'")
    print(f"✅ Exported chunks to '{EXPORT_DIR / 'chunks_demo.jsonl'}'")


if __name__ == "__main__":
    main()
