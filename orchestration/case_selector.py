"""
orchestration/case_selector.py

Agent 1 — Case Selector.

Reads case_manifest.json to understand the current corpus, makes a Sonnet call
to recommend new cases that balance the corpus across topics and eras, presents
them for human review at a mandatory approval gate, then (on approval) merges
the new entries into the manifest.

The 'rationale' field is display-only: shown at review time, stripped on write
so the manifest stays consistent with existing entries.

Sonnet is called exactly once per generate-and-approve cycle. Recommendations
are written to a staging file (data/ingestion/staged_recs.json) immediately
after parsing. If the script is restarted before the approval gate is answered,
it loads from staging rather than calling Sonnet again — guaranteeing that
what is displayed for review is exactly what gets written on approval.
Use --fresh to clear the staging file and regenerate.

Usage:
    python orchestration/case_selector.py
    python orchestration/case_selector.py --topic "First Amendment"
    python orchestration/case_selector.py --era "Burger Court" --count 3
    python orchestration/case_selector.py --topic "First Amendment" --era "Warren Court" --count 4
    python orchestration/case_selector.py --output path/to/recs.json
    python orchestration/case_selector.py --fresh --topic "First Amendment"
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import anthropic

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = _ROOT / "data" / "cases" / "case_manifest.json"
STAGING_PATH = _ROOT / "data" / "ingestion" / "staged_recs.json"
SONNET_MODEL = "claude-sonnet-4-6"

# Keys written to the manifest on approval — rationale is explicitly excluded.
_MANIFEST_WRITE_KEYS = (
    "title",
    "citation",
    "courtlistener_query",
    "expected_decision_year",
    "court_era",
    "topics",
    "decision_year",
)

_KNOWN_ERAS = (
    "Marshall Court",
    "Early Court",
    "Taney Court",
    "Lochner Era",
    "Warren Court",
    "Burger Court",
    "Rehnquist Court",
    "Roberts Court",
)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}\n"
            "Run from the project root or check MANIFEST_PATH."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_to_manifest(
    new_entries: List[Dict[str, Any]],
    path: Path = MANIFEST_PATH,
) -> None:
    """Read full manifest → merge new entries (rationale stripped) → rewrite."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stripped = [
        {k: entry[k] for k in _MANIFEST_WRITE_KEYS if k in entry}
        for entry in new_entries
    ]
    manifest.extend(stripped)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Staging file (one-call guarantee)
# ---------------------------------------------------------------------------

def _save_staging(recs: List[Dict[str, Any]], args_summary: Dict[str, Any]) -> None:
    """Persist recommendations to disk before showing the review table.

    This means a restart (e.g. because the process was backgrounded and
    couldn't receive stdin) loads the same candidates instead of calling
    Sonnet again — so what is displayed for review is always exactly what
    gets written on approval.
    """
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)
    STAGING_PATH.write_text(
        json.dumps({"args": args_summary, "recs": recs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_staging() -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    """Return (recs, args_summary) from the staging file, or (None, None)."""
    if not STAGING_PATH.exists():
        return None, None
    payload = json.loads(STAGING_PATH.read_text(encoding="utf-8"))
    return payload.get("recs"), payload.get("args")


def _clear_staging() -> None:
    if STAGING_PATH.exists():
        STAGING_PATH.unlink()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _corpus_summary(manifest: List[Dict[str, Any]]) -> str:
    lines = ["Title | Citation | Era | Topics", "-" * 80]
    for entry in manifest:
        topics = ", ".join(entry.get("topics") or [])
        lines.append(
            f"{entry['title']} | {entry['citation']} | "
            f"{entry.get('court_era', 'unknown')} | {topics}"
        )
    return "\n".join(lines)


def _existing_identifiers(manifest: List[Dict[str, Any]]) -> Tuple[set, set]:
    titles = {e["title"].lower().strip() for e in manifest}
    citations = {e["citation"].lower().strip() for e in manifest}
    return titles, citations


def build_prompt(
    manifest: List[Dict[str, Any]],
    topic: Optional[str],
    era: Optional[str],
    count: int,
) -> str:
    corpus = _corpus_summary(manifest)
    existing_titles = "\n".join(f"  - {e['title']}" for e in manifest)

    # Count cases per era for the gap analysis in the prompt
    era_counts: Dict[str, int] = {}
    for e in manifest:
        era_counts[e.get("court_era", "unknown")] = era_counts.get(e.get("court_era", "unknown"), 0) + 1

    topic_counts: Dict[str, int] = {}
    for e in manifest:
        for t in e.get("topics") or []:
            topic_counts[t] = topic_counts.get(t, 0) + 1

    era_summary = "\n".join(f"  {era}: {n} case(s)" for era, n in sorted(era_counts.items()))
    topic_summary = "\n".join(
        f"  {t}: {n}" for t, n in sorted(topic_counts.items(), key=lambda x: -x[1])
    )

    constraints: List[str] = []
    if topic:
        constraints.append(f'- Prioritize cases relevant to the topic: "{topic}"')
    if era:
        constraints.append(f'- Prioritize cases from the court era: "{era}"')
    if not constraints:
        constraints.append("- Select based on corpus balance: fill topic and era gaps")
    constraint_block = "\n".join(constraints)

    known_eras_str = ", ".join(f'"{e}"' for e in _KNOWN_ERAS)

    return f"""You are curating a corpus of landmark U.S. Supreme Court opinions for a legal research RAG system.

CURRENT CORPUS ({len(manifest)} cases indexed):
{corpus}

ERA DISTRIBUTION:
{era_summary}

TOPIC FREQUENCY (cases per topic tag):
{topic_summary}

CASES ALREADY INDEXED — do NOT recommend any of these:
{existing_titles}

TASK: Recommend exactly {count} additional U.S. Supreme Court cases.

SELECTION CONSTRAINTS:
{constraint_block}
- Must NOT be in the already-indexed list above (check carefully by title and citation)
- Must be landmark SCOTUS decisions significant enough to appear in legal education
- Should fill gaps in the corpus — consider topics with 0 or 1 case, and underrepresented eras
- Must be available on CourtListener (all SCOTUS opinions with U.S. Reports citations are)
- Prefer cases that connect doctrinally to cases already in the corpus

KNOWN TOPIC GAPS TO CONSIDER:
- First Amendment (free speech, free exercise, establishment): 0 cases
- Second Amendment: 0 cases
- Commerce Clause and federalism (beyond Marbury): thin
- Equal protection (beyond Brown): thin
- Substantive due process (beyond Meyer): thin
- Burger Court: 0 cases
- Roberts Court: 1 case only (Riley v. California)

OUTPUT: Return ONLY a JSON array with exactly {count} objects. No prose, no markdown fences.
Each object must have exactly these keys:

[
  {{
    "title": "Case Name v. Other Party",
    "citation": "### U.S. ###",
    "courtlistener_query": "Case Name v. Other Party ### U.S. ###",
    "expected_decision_year": YYYY,
    "court_era": one of {known_eras_str},
    "topics": ["topic1", "topic2"],
    "decision_year": YYYY,
    "rationale": "one sentence: which specific corpus gap this fills"
  }}
]

Rules:
- "expected_decision_year" and "decision_year" must be the same integer
- "citation" must be the U.S. Reports citation (### U.S. ###); use S. Ct. only if no U.S. Reports cite exists
- "courtlistener_query" is the string that will be passed to CourtListener's search API: include the case name and citation
- "topics" must be a JSON array of 2–5 short lowercase strings
- "rationale" is one sentence stating which gap in the current corpus this case fills
- Return only the JSON array — no text before or after"""


# ---------------------------------------------------------------------------
# Sonnet call
# ---------------------------------------------------------------------------

def call_sonnet(prompt: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------

def parse_recommendations(text: str) -> List[Dict[str, Any]]:
    """Extract the first JSON array from the model response.

    Strips markdown fences and leading/trailing prose so parsing is robust
    to common model formatting habits.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(
            f"No JSON array found in model response.\n\nRaw response:\n{text}"
        )
    return json.loads(match.group(0))


def validate_recommendations(
    recs: List[Dict[str, Any]],
    manifest: List[Dict[str, Any]],
) -> List[str]:
    """Return a list of validation error strings. Empty list means all good."""
    errors: List[str] = []
    existing_titles, existing_citations = _existing_identifiers(manifest)
    required_keys = set(_MANIFEST_WRITE_KEYS) | {"rationale"}

    seen_titles: set = set()
    seen_citations: set = set()

    for i, rec in enumerate(recs):
        label = f"[{i + 1}] {rec.get('title', '?')}"

        missing = required_keys - rec.keys()
        if missing:
            errors.append(f"{label}: missing keys: {sorted(missing)}")
            continue

        title_key = rec["title"].lower().strip()
        citation_key = rec["citation"].lower().strip()

        if title_key in existing_titles:
            errors.append(f"{label}: title already in manifest")
        if citation_key in existing_citations:
            errors.append(f"{label}: citation already in manifest")
        if title_key in seen_titles:
            errors.append(f"{label}: duplicate title within recommendations")
        if citation_key in seen_citations:
            errors.append(f"{label}: duplicate citation within recommendations")

        seen_titles.add(title_key)
        seen_citations.add(citation_key)

        if rec["expected_decision_year"] != rec["decision_year"]:
            errors.append(
                f"{label}: expected_decision_year ({rec['expected_decision_year']}) "
                f"!= decision_year ({rec['decision_year']})"
            )

        if not isinstance(rec.get("topics"), list) or not rec["topics"]:
            errors.append(f"{label}: 'topics' must be a non-empty list")

        if rec.get("court_era") not in _KNOWN_ERAS:
            errors.append(
                f"{label}: unrecognized court_era '{rec.get('court_era')}' "
                f"(known: {', '.join(_KNOWN_ERAS)})"
            )

    return errors


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_review_table(recs: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(f"RECOMMENDED CASES ({len(recs)})")
    print("=" * 80)
    for i, rec in enumerate(recs, 1):
        topics = ", ".join(rec.get("topics") or [])
        print(f"\n[{i}] {rec['title']}")
        print(f"    Citation:   {rec.get('citation', '?')}")
        print(f"    Era:        {rec.get('court_era', '?')}")
        print(f"    Year:       {rec.get('decision_year', '?')}")
        print(f"    Topics:     {topics}")
        print(f"    Rationale:  {rec.get('rationale', '')}")
        print(f"    CL query:   {rec.get('courtlistener_query', '')}")
    print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent 1 — recommend new SCOTUS cases for corpus expansion"
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help='Bias recommendations toward a topic, e.g. "First Amendment"',
    )
    parser.add_argument(
        "--era", type=str, default=None,
        help='Bias recommendations toward a court era, e.g. "Burger Court"',
    )
    parser.add_argument(
        "--count", type=int, default=4,
        help="Number of cases to recommend (default: 4)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write recommendations JSON to this path and exit without modifying the manifest",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any staged recommendations and call Sonnet again",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    # ── Clear staging if --fresh ─────────────────────────────────────────────
    if args.fresh:
        _clear_staging()
        print(f"[staged] Cleared staged recommendations ({STAGING_PATH.name})")

    # ── Load manifest ────────────────────────────────────────────────────────
    print(f"[manifest] Loading: {MANIFEST_PATH}")
    manifest = load_manifest()
    print(f"   {len(manifest)} cases currently indexed")

    filters = []
    if args.topic:
        filters.append(f"topic={args.topic!r}")
    if args.era:
        filters.append(f"era={args.era!r}")
    print(f"   Parameters: count={args.count}, " + (", ".join(filters) if filters else "no filters"))

    # ── Load staged recommendations or call Sonnet (exactly once) ───────────
    staged_recs, staged_args = _load_staging()
    if staged_recs is not None:
        print(f"\n[staged] Loading recommendations from prior run ({STAGING_PATH.name})")
        if staged_args:
            summary = ", ".join(f"{k}={v!r}" for k, v in staged_args.items())
            print(f"   Generated with: {summary}")
        print("   Skipping Sonnet call. Use --fresh to regenerate.")
        recs = staged_recs
    else:
        print("\n[sonnet] Calling for recommendations...")
        prompt = build_prompt(manifest, args.topic, args.era, args.count)
        raw = call_sonnet(prompt)

        try:
            recs = parse_recommendations(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"\n[error] Failed to parse model response: {exc}", file=sys.stderr)
            sys.exit(1)

        # Persist immediately — before display, before prompt.
        # Any restart will load from here rather than calling Sonnet again.
        args_summary = {k: v for k, v in vars(args).items() if v is not None and k not in ("fresh",)}
        _save_staging(recs, args_summary)
        print(f"   Staged to {STAGING_PATH.name}")

    # ── Validate ─────────────────────────────────────────────────────────────
    errors = validate_recommendations(recs, manifest)
    if errors:
        print("\n[warn] Validation issues:")
        for err in errors:
            print(f"   - {err}")
        print()

    # ── Display for review ───────────────────────────────────────────────────
    print_review_table(recs)

    # ── --output path: write JSON and exit without touching the manifest ─────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(recs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[ok] Recommendations written to {out_path}")
        print("   Manifest not modified (--output mode).")
        return

    # ── Human approval gate (Gate 1) ─────────────────────────────────────────
    if errors:
        print("[warn] There are validation issues above. Review carefully before approving.")

    try:
        answer = input("Add these cases to case_manifest.json? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)

    if answer != "y":
        print("Aborted -- manifest not modified.")
        print(f"   Staged recommendations preserved at {STAGING_PATH.name} -- re-run to approve.")
        sys.exit(0)

    # ── Write (read → merge → rewrite) ───────────────────────────────────────
    write_to_manifest(recs, MANIFEST_PATH)
    _clear_staging()

    print(f"\n[ok] Added {len(recs)} case(s) to {MANIFEST_PATH.name}:")
    for rec in recs:
        print(f"   + {rec['title']} ({rec['citation']})")
    print("\n   Next step: run python ingest.py to fetch and embed these cases.")


if __name__ == "__main__":
    main()
