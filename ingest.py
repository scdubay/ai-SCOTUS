"""
ingest.py

Manifest-driven ingestion pipeline for SCOTUS Legal Aid. Upgrades
CourtListener_clean.py's one-shot hardcoded ingestion into a repeatable,
validated, versioned process:

  1. Read cases to add from data/cases/case_manifest.json (not hardcoded).
  2. Skip cases already present in the existing index (duplicate detection).
  3. Validate each CourtListener result against the manifest's expected
     citation and decision year before trusting it.
  4. Embed with the same model app.py / query_demo_clean.py use
     (BAAI/bge-small-en-v1.5) -- never the old Ollama embeddings.
  5. Never overwrite the existing vector store. Save additions to a new,
     version-numbered folder.
  6. Write a structured JSON ingestion report.

CourtListener_clean.py is left untouched and reused for the parts of the
pipeline that didn't change (CourtListener search/fetch, opinion-section
splitting, canonical metadata, legal-aware chunking).

Required environment:
  $env:COURTLISTENER_TOKEN = "<token>"

Usage:
  python ingest.py                       # ingest all manifest cases not already indexed
  python ingest.py --dry-run             # validate only, write nothing
  python ingest.py --case "Terry v. Ohio"  # ingest a single manifest case
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from query_demo_clean import VECTORSTORE_PATH, HF_EMBED_MODEL, normalize_text
from CourtListener_clean import (
    search_courtlistener_case,
    build_case_dossier,
    build_opinion_documents,
    smart_legal_chunking,
    get_with_retry,
    COURTLISTENER_OPINION_URL_TEMPLATE,
)

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/cases/case_manifest.json")
VECTOR_BASE_DIR = Path("data/vectors")
REPORT_DIR = Path("data/ingestion")

COURTLISTENER_CLUSTER_URL_TEMPLATE = "https://www.courtlistener.com/api/rest/v4/clusters/{cluster_id}/"

YEAR_TOLERANCE = 1

# =============================================================================
# Manifest
# =============================================================================

_REQUIRED_MANIFEST_KEYS = ("title", "citation", "courtlistener_query", "expected_decision_year")


def load_manifest(path: Path = MANIFEST_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {path}. Create it with the cases to ingest -- "
            f"ingest.py does not auto-generate this file."
        )

    entries = json.loads(path.read_text(encoding="utf-8"))

    for i, entry in enumerate(entries):
        missing = [k for k in _REQUIRED_MANIFEST_KEYS if k not in entry]
        if missing:
            raise ValueError(f"Manifest entry {i} is missing required key(s): {missing}")

    return entries


# =============================================================================
# Direct opinion-ID fetch (bypasses search)
# =============================================================================

def fetch_result_by_opinion_id(opinion_id: int) -> Dict[str, Any]:
    """Fetch a case directly by a known CourtListener opinion ID, bypassing
    free-text search entirely.

    Exists for cases where search_courtlistener_case()'s relevance ranking
    returns the wrong cluster -- e.g. Brown v. Board of Education, where a
    query for the 1954 decision (347 U.S. 483) reliably returns the 1955
    Brown II cluster (349 U.S. 294) instead, because both share an identical
    case name and CourtListener's ranking doesn't weight the citation in the
    query strongly enough to disambiguate them.

    Builds a dict shaped like a search-API result (the shape
    build_case_dossier() expects: camelCase keys, a flat citation string
    list) from the cluster-detail REST endpoint, which uses different field
    names entirely (snake_case, structured citation objects) since it's the
    database resource, not the search index.
    """
    opinion_resp = get_with_retry(
        COURTLISTENER_OPINION_URL_TEMPLATE.format(opinion_id=opinion_id),
        timeout=90,
        retries=3,
    )
    opinion_data = opinion_resp.json()
    cluster_id = opinion_data.get("cluster_id")

    cluster_resp = get_with_retry(
        COURTLISTENER_CLUSTER_URL_TEMPLATE.format(cluster_id=cluster_id),
        timeout=90,
        retries=3,
    )
    cluster = cluster_resp.json()

    citations = [
        f"{c['volume']} {c['reporter']} {c['page']}"
        for c in cluster.get("citations", [])
        if c.get("volume") and c.get("reporter") and c.get("page")
    ]

    sub_opinion_ids = []
    for url in cluster.get("sub_opinions", []):
        match = re.search(r"/opinions/(\d+)/", url)
        if match:
            sub_opinion_ids.append(int(match.group(1)))

    return {
        "caseName": cluster.get("case_name"),
        "caseNameFull": cluster.get("case_name_full"),
        # This bypass path is only ever used for cases already confirmed to
        # be SCOTUS opinions (that's the whole corpus), so it's safe to set
        # these directly rather than following another URL (docket) to look
        # them up.
        "court": "Supreme Court of the United States",
        "court_id": "scotus",
        "citation": citations,
        "cluster_id": cluster_id,
        "docket_id": cluster.get("docket_id"),
        "docketNumber": None,
        "dateFiled": cluster.get("date_filed"),
        "dateArgued": None,
        "scdb_id": cluster.get("scdb_id"),
        "citeCount": cluster.get("citation_count"),
        "judge": cluster.get("judges"),
        "attorney": cluster.get("attorneys"),
        "procedural_history": cluster.get("procedural_history"),
        "posture": cluster.get("posture"),
        "status": cluster.get("precedential_status"),
        "absolute_url": cluster.get("absolute_url"),
        "opinions": [{"id": oid} for oid in sub_opinion_ids],
    }


# =============================================================================
# Duplicate detection
# =============================================================================

def existing_case_index(vectorstore: FAISS) -> Dict[str, set]:
    """Map normalized case_title -> set of raw citation strings stored under it.

    Keyed by title AND citation (not title alone) so two distinct decisions
    that happen to share a case name -- e.g. Brown v. Board of Education I
    (1954, 347 U.S. 483) vs. Brown II (1955, 349 U.S. 294) -- aren't treated
    as the same case just because the name string matches.
    """
    index: Dict[str, set] = {}
    for doc in vectorstore.docstore._dict.values():
        title = doc.metadata.get("case_title")
        if not title:
            continue
        key = normalize_text(title)
        index.setdefault(key, set()).add(doc.metadata.get("citation") or "")
    return index


def is_duplicate(manifest_entry: Dict[str, Any], existing_index: Dict[str, set]) -> bool:
    key = normalize_text(manifest_entry["title"])
    citations = existing_index.get(key)
    if not citations:
        return False  # title not present at all -> definitely not a duplicate
    want_citation = _normalize_citation(manifest_entry["citation"])
    return any(want_citation in _normalize_citation(c) for c in citations)


# =============================================================================
# Citation / year validation
# =============================================================================

def _normalize_citation(citation: str) -> str:
    return re.sub(r"\s+", "", citation or "").lower()


def _year_from_date_filed(date_filed: Optional[str]) -> Optional[int]:
    if not date_filed:
        return None
    match = re.match(r"(\d{4})", str(date_filed))
    return int(match.group(1)) if match else None


def validate_case(
    dossier: Dict[str, Any],
    manifest_entry: Dict[str, Any],
) -> Tuple[bool, Optional[str], Optional[int], List[str]]:
    """Validate a fetched dossier against manifest expectations.

    Returns (valid, citation_found, year_found, warnings).
    citation_found / year_found are reported even on failure, for the report.
    """
    warnings: List[str] = []

    expected_citation = manifest_entry["citation"]
    expected_year = manifest_entry["expected_decision_year"]

    citations = dossier.get("citations") or []
    citation_found = ", ".join(citations) if citations else None

    want_citation = _normalize_citation(expected_citation)
    citation_ok = any(want_citation in _normalize_citation(c) for c in citations)
    if not citation_ok:
        warnings.append(
            f"Expected citation '{expected_citation}' not found in returned "
            f"citations {citations!r}"
        )

    year_found = _year_from_date_filed(dossier.get("date_filed"))
    if year_found is None:
        warnings.append("CourtListener result had no date_filed; cannot verify decision year")
        year_ok = False
    else:
        year_ok = abs(year_found - expected_year) <= YEAR_TOLERANCE
        if not year_ok:
            warnings.append(
                f"Expected decision year {expected_year} (±{YEAR_TOLERANCE}), "
                f"got {year_found}"
            )

    return (citation_ok and year_ok), citation_found, year_found, warnings


# =============================================================================
# Per-case processing
# =============================================================================

def process_case(
    manifest_entry: Dict[str, Any],
    existing_index: Dict[str, set],
) -> Tuple[str, Dict[str, Any], List[Document]]:
    """Run duplicate check -> CourtListener fetch -> validation -> chunking
    for a single manifest entry.

    Returns (status, report_detail, chunks). chunks is empty unless status == "added".
    """
    title = manifest_entry["title"]
    detail: Dict[str, Any] = {
        "title": title,
        "citation_expected": manifest_entry["citation"],
        "citation_found": None,
        "year_expected": manifest_entry["expected_decision_year"],
        "year_found": None,
        "status": None,
        "opinion_count": 0,
        "chunk_count": 0,
        "warnings": [],
    }

    if is_duplicate(manifest_entry, existing_index):
        detail["status"] = "skipped_duplicate"
        detail["warnings"].append(f"'{title}' already present in the index; skipped")
        logger.warning(detail["warnings"][-1])
        return "skipped_duplicate", detail, []

    query = manifest_entry["courtlistener_query"]
    direct_opinion_id = manifest_entry.get("courtlistener_opinion_id")

    try:
        if direct_opinion_id:
            result = fetch_result_by_opinion_id(direct_opinion_id)
        else:
            result = search_courtlistener_case(query)
    except Exception as exc:
        detail["status"] = "failed_fetch"
        source = f"opinion id {direct_opinion_id}" if direct_opinion_id else "search"
        detail["warnings"].append(f"CourtListener fetch by {source} failed: {exc}")
        logger.warning(detail["warnings"][-1])
        return "failed_fetch", detail, []

    dossier = build_case_dossier(result, query)

    valid, citation_found, year_found, val_warnings = validate_case(dossier, manifest_entry)
    detail["citation_found"] = citation_found
    detail["year_found"] = year_found
    detail["warnings"].extend(val_warnings)

    if not valid:
        detail["status"] = "failed_validation"
        for w in val_warnings:
            logger.warning("[%s] %s", title, w)
        return "failed_validation", detail, []

    case_docs = build_opinion_documents(result, query, dossier=dossier)
    if not case_docs:
        detail["status"] = "failed_fetch"
        detail["warnings"].append("No usable opinion documents were built for this case")
        logger.warning("[%s] %s", title, detail["warnings"][-1])
        return "failed_fetch", detail, []

    # The manifest's title is authoritative, not whatever CourtListener's API
    # happens to call the case. This matters when two distinct decisions share
    # an identical case_name (e.g. Brown I vs. Brown II) -- without this, two
    # decisions can end up tagged with the same case_title metadata and get
    # merged together by case_store.py's build_case_index(), which groups
    # purely by that field.
    for doc in case_docs:
        doc.metadata["case_title"] = title

    detail["opinion_count"] = len(case_docs)

    chunks = smart_legal_chunking(case_docs)
    detail["chunk_count"] = len(chunks)
    detail["status"] = "added"

    return "added", detail, chunks


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Manifest-driven SCOTUS case ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; write nothing")
    parser.add_argument("--case", type=str, default=None, help="Ingest only this manifest case")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the CourtListener response cache for this run")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["COURTLISTENER_CACHE_ENABLED"] = "false"

    manifest = load_manifest()

    if args.case:
        want = normalize_text(args.case)
        manifest = [e for e in manifest if normalize_text(e["title"]) == want]
        if not manifest:
            raise SystemExit(f"No manifest entry found matching --case '{args.case}'")

    print(f"📋 Loaded manifest: {len(manifest)} case(s) requested")

    print("🔎 Loading existing vector store...")
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    existing_index = existing_case_index(vectorstore)
    existing_count = len(existing_index)
    print(f"✅ Existing index: {existing_count} case(s)")

    cases_requested = [e["title"] for e in manifest]
    cases_added: List[str] = []
    cases_skipped: List[str] = []
    cases_failed: List[str] = []
    details: List[Dict[str, Any]] = []
    all_new_chunks: List[Document] = []

    for entry in manifest:
        title = entry["title"]
        print("\n" + "=" * 80)
        print(f"Case: {title}")

        status, detail, chunks = process_case(entry, existing_index)
        details.append(detail)

        if status == "added":
            cases_added.append(title)
            all_new_chunks.extend(chunks)
            print(f"✅ Validated | citation={detail['citation_found']} | year={detail['year_found']} "
                  f"| opinions={detail['opinion_count']} | chunks={detail['chunk_count']}")
        elif status == "skipped_duplicate":
            cases_skipped.append(title)
            print("⏭️  Skipped (duplicate)")
        else:
            cases_failed.append(title)
            print(f"❌ {status} | citation={detail['citation_found']} | year={detail['year_found']}")
            for w in detail["warnings"]:
                print(f"   ⚠️ {w}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print(f"  Requested: {len(cases_requested)}")
    print(f"  Added:     {len(cases_added)} -> {cases_added}")
    print(f"  Skipped:   {len(cases_skipped)} -> {cases_skipped}")
    print(f"  Failed:    {len(cases_failed)} -> {cases_failed}")

    if args.dry_run:
        print("\n🧪 Dry run -- no vector store or report written.")
        return

    if not all_new_chunks:
        print("\nℹ️ No cases were added; existing index left untouched.")
        new_vectorstore_path = str(VECTORSTORE_PATH)
    else:
        print(f"\n🔐 Embedding and merging {len(all_new_chunks)} new chunk(s)...")
        vectorstore.add_documents(all_new_chunks)

        new_total_cases = existing_count + len(cases_added)
        new_vectorstore_path = str(VECTOR_BASE_DIR / f"rag_vectorstore_scotus_{new_total_cases:03d}_cases")
        vectorstore.save_local(new_vectorstore_path)
        print(f"✅ Saved new vector store version: {new_vectorstore_path}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vectorstore_path": new_vectorstore_path,
        "previous_vectorstore_path": str(VECTORSTORE_PATH),
        "cases_requested": cases_requested,
        "cases_added": cases_added,
        "cases_skipped": cases_skipped,
        "cases_failed": cases_failed,
        "details": details,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_n = existing_count + len(cases_added)
    report_path = REPORT_DIR / f"ingestion_report_{report_n:03d}_cases.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Wrote ingestion report: {report_path}")


if __name__ == "__main__":
    main()
