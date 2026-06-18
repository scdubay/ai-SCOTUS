"""
faithfulness.py

A heuristic guard against the failure modes term-coverage can't see — the ones
that make a legal tool dangerous:

  1. Hallucinated authority: a justice or a case named in the ANSWER that does
     not appear anywhere in the CONTEXT the model was given. (Catches the
     invented "Justice Holmes concurrence" and "Scalia concurrence" we saw.)

  2. Role misattribution: the answer assigns an opinion role (majority /
     concurrence / dissent) to an author whose actual segment role differs.
     (Catches calling Kennedy's majority "a concurrence".)

This is deliberately high-precision, not high-recall: a name that is simply
absent from the context is a strong hallucination signal, so we flag those
confidently; attribution conflicts are only raised when we actually have an
authored segment to check against, to avoid false alarms. It is a tripwire to
sit next to the retrieval metrics, not a substitute for a capable model.
"""

import re
from typing import Dict, List


# "Mr. Justice Holmes", "Chief Justice Warren", "Justice O'Connor"
_JUSTICE_TITLED = re.compile(
    r"(?:Mr\.\s+)?(?:Chief\s+)?Justice\s+([A-Z][A-Za-z'’\-]+)"
)

# Reporter-style author tags: "Scalia, J.", "Rehnquist, C. J."
_JUSTICE_TAG = re.compile(
    r"\b([A-Z][A-Za-z'’\-]+),\s+(?:C\.\s+)?J\."
)

# "Foo v. Bar" / "Calero-Toledo v. Pearson Yacht"
_CASE_WORD = (
    r"(?:"
    r"[A-Z][A-Za-z'’\-$&.]*"
    r"|of|the|and|for|in|on|at|to"
    r"|New|United|State|States|City|County|Board"
    r"|Commissioner|Commission|Department|Hospital"
    r"|Lessee|Executives|Association|Ass'n|Trustees|Committee|Refugee"
    r"|Railway|Finance|Trust|Co\.?|Corp\.?|Assn\.?"
    r")"
)

_CASE = re.compile(
    rf"\b({_CASE_WORD}(?:[ \t]+{_CASE_WORD}){{0,8}})"
    rf"[ \t]+v\.[ \t]+"
    rf"({_CASE_WORD}(?:[ \t]+{_CASE_WORD}){{0,8}})"
)


_ROLE_FAMILY = {
    "majority": "court",
    "court opinion": "court",
    "opinion of the court": "court",
    "lead opinion": "court",
    "concurrence": "concur",
    "concurring": "concur",
    "concurred": "concur",
    "dissent": "dissent",
    "dissenting": "dissent",
}

# Map stored opinion roles to the same families.
_SEG_ROLE_FAMILY = {
    "court_opinion": "court",
    "majority": "court",
    "syllabus": "court",
    "concurrence": "concur",
    "concurrence_in_judgment": "concur",
    "dissent": "dissent",
    "concurrence_dissent": "mixed",  # legitimately both; never conflicts
}

_COMMON_WORDS = {
    "the",
    "court",
    "government",
    "united",
    "states",
    "state",
    "act",
}


def _surname(token: str) -> str:
    token = re.sub(r"['’]s$", "", token)
    return re.sub(r"[^a-z]", "", token.lower())


def _meta(seg):
    """Accept langchain Documents (.metadata) or plain dicts."""
    if hasattr(seg, "metadata"):
        return seg.metadata
    if isinstance(seg, dict):
        return seg.get("metadata", seg)
    return {}


def justices_in(text: str) -> set:
    names = set()

    for m in _JUSTICE_TITLED.findall(text or ""):
        names.add(_surname(m))

    for m in _JUSTICE_TAG.findall(text or ""):
        s = _surname(m)
        if s and s not in _COMMON_WORDS:
            names.add(s)

    return {n for n in names if n}


def _case_name_norm(value: str) -> str:
    """
    Normalize case names only.

    Do not use a general-purpose normalizer here because case names need
    words like 'New', 'United', 'State', 'Board', etc.
    """
    value = value or ""
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.rstrip(".,;:)]}")
    value = value.lower()

    # Normalize citation-intro abbreviations captured before the case name.
    value = re.sub(r"\bcf\.", "cf", value)
    value = re.sub(r"\be\.g\.", "eg", value)
    value = re.sub(r"\bi\.e\.", "ie", value)

    return value


def _trim_left_party(parts: List[str]) -> List[str]:
    """
    Trim prose/citation-intro words accidentally captured before the real case name.

    Examples:
      "liberty interest recognized in meyer" -> "meyer"
      "see austin" -> "austin"
      "cf gerstein" -> "gerstein"
      "and connecticut" -> "connecticut"
    """
    if not parts:
        return parts

    prefix_noise = {
        "answer",
        "holding",
        "precedents",
        "recognized",
        "recognised",
        "liberty",
        "interest",
        "interests",
        "right",
        "rights",
        "court",
        "opinion",
        "majority",
        "dissent",
        "concurrence",
        "under",
        "in",

        # citation-intro noise
        "see",
        "cf",
        "compare",
        "quoting",
        "citing",
        "and",
        "or",
        "but",
        "also",
        "e.g",
        "eg",
        "supra",
        "ante",
        "id",
        "ibid",
        "of",
        "on",
        "at",
        "to",
        "the",
    }

    while len(parts) > 1 and parts[0] in prefix_noise:
        parts.pop(0)

    # Special case seen in markdown headings:
    # "liberty interest recognized in meyer" -> "meyer"
    if len(parts) > 1 and "recognized" in parts:
        idx = parts.index("recognized")
        if idx + 1 < len(parts):
            parts = parts[idx + 1:]

    while len(parts) > 1 and parts[0] in prefix_noise:
        parts.pop(0)

    return parts


def cases_in(text: str) -> set:
    found = set()

    if not text:
        return found

    false_positives = {
        "meyer v. under",
    }

    bad_right_party_starts = {
        "under",
        "this",
        "that",
        "which",
        "what",
        "when",
        "where",
        "why",
        "how",
    }

    trailing_stop_words = {
        "under",
        "recognized",
        "applied",
        "relied",
        "reasoned",
        "concluded",
        "stated",
        "explained",
        "held",
        "holding",
        "court",
        "opinion",
        "case",
    }

    for m in _CASE.finditer(text):
        left = _case_name_norm(m.group(1))
        right = _case_name_norm(m.group(2))

        left_parts = left.split()
        right_parts = right.split()

        while left_parts and left_parts[-1] in trailing_stop_words:
            left_parts.pop()

        while right_parts and right_parts[-1] in trailing_stop_words:
            right_parts.pop()

        left_parts = _trim_left_party(left_parts)

        if not left_parts or not right_parts:
            continue

        if right_parts[0] in bad_right_party_starts:
            continue

        case_name = f"{' '.join(left_parts)} v. {' '.join(right_parts)}"

        if case_name in false_positives:
            continue

        found.add(case_name)

    return found


def case_pair_from_title(case_title: str) -> set:
    """
    Convert a known case title like 'Meyer v. Nebraska' into the same
    normalized format used by cases_in().
    """
    if not case_title:
        return set()

    found = cases_in(case_title)
    if found:
        return found

    return set()


_GENERIC_PARTY = _COMMON_WORDS | {
    "united",
    "states",
    "state",
    "city",
    "county",
    "board",
    "commissioner",
    "commission",
    "department",
    "co",
    "inc",
    "corp",
    "company",
    "of",
    "the",
    "and",
    "new",
    "et",
    "al",
    "bank",
    "trust",
    "municipal",
    "v",
}


def _significant_case_tokens(case_str: str) -> list:
    """
    Distinctive party tokens that actually identify a case.

    Drops generic legal/corporate/government words so that variants like:
      Chicago, Burlington & Quincy R.R. Co. v. McGuire
      Quincy R.R. Co. v. McGuire
    can still be treated as related when their distinctive tokens appear.
    """
    toks = re.findall(r"[a-z0-9]+", (case_str or "").lower())
    return [t for t in toks if len(t) > 2 and t not in _GENERIC_PARTY]


def _context_tokens(context: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (context or "").lower()))


def justices_from_segments(segments: List) -> set:
    """
    Treat segment metadata authors as supported authorities.

    This avoids false positives where the context header says author=Kennedy
    but the prose does not contain 'Justice Kennedy' or 'Kennedy, J.'.
    """
    names = set()

    for seg in segments:
        m = _meta(seg)
        author = m.get("opinion_author") or ""

        toks = [t for t in re.findall(r"[A-Za-z'’\-]+", author) if len(t) > 1]
        if toks:
            sn = _surname(toks[-1])
            if sn and sn not in _COMMON_WORDS:
                names.add(sn)

    return names


def _segment_author_roles(segments: List) -> Dict[str, set]:
    """surname -> set of role families actually authored in the provided segments."""
    author_roles: Dict[str, set] = {}

    for seg in segments:
        m = _meta(seg)
        author = m.get("opinion_author") or ""
        role = m.get("effective_opinion_role") or m.get("opinion_role") or ""
        fam = _SEG_ROLE_FAMILY.get(str(role).lower())

        if not author or not fam:
            continue

        # author_str may be "Justice Anthony M. Kennedy"; take the last name token.
        toks = [t for t in re.findall(r"[A-Za-z'’\-]+", author) if len(t) > 1]
        if not toks:
            continue

        sn = _surname(toks[-1])
        author_roles.setdefault(sn, set()).add(fam)

    return author_roles


def _claimed_attributions(answer: str) -> List[tuple]:
    """Find (justice_surname, role_family) the answer asserts."""
    claims = []

    # "Justice X's dissent", "Justice X, dissenting", "Justice X ... concurrence"
    for m in re.finditer(
        r"(?:Mr\.\s+)?(?:Chief\s+)?Justice\s+([A-Z][A-Za-z'’\-]+)[^.]{0,60}?"
        r"\b(majority|court opinion|opinion of the court|concurrence|concurring|"
        r"concurred|dissent|dissenting)\b",
        answer or "",
        flags=re.IGNORECASE,
    ):
        sn = _surname(m.group(1))
        fam = _ROLE_FAMILY.get(m.group(2).lower())
        if sn and fam:
            claims.append((sn, fam))

    # "X's concurrence" / "X's dissent" (name without the Justice title)
    for m in re.finditer(
        r"\b([A-Z][A-Za-z'’\-]+)'s\s+(majority|concurrence|dissent)\b",
        answer or "",
    ):
        sn = _surname(m.group(1))
        fam = _ROLE_FAMILY.get(m.group(2).lower())
        if sn and fam and sn not in _COMMON_WORDS:
            claims.append((sn, fam))

    return claims


def _case_sides(case_str: str) -> tuple:
    """Distinctive tokens of each side of the 'v.'"""
    if " v. " in case_str:
        left, right = case_str.split(" v. ", 1)
    else:
        left, right = case_str, ""
    return _significant_case_tokens(left), _significant_case_tokens(right)


def _case_supported(case_str: str, supported_tokens: set, ctx_cases: set) -> bool:
    """
    Supported if each SIDE that has distinctive tokens has at least one present.
    Survives party-name abbreviation (lead surname almost always appears) while
    still catching wrong-party fabrications (the invented side has no token).
    """
    if case_str in ctx_cases:
        return True
    left_sig, right_sig = _case_sides(case_str)
    if not left_sig and not right_sig:
        return True  # nothing distinctive -> do not flag
    for side in (left_sig, right_sig):
        if side and not any(t in supported_tokens for t in side):
            return False
    return True


def _unsupported_cases(ans_cases: set, ctx_cases: set, context: str, case_title: str = "") -> List[str]:
    supported_tokens = _context_tokens(context)
    if case_title:
        supported_tokens = supported_tokens | set(_significant_case_tokens(case_title))
    return [
        c for c in sorted(ans_cases)
        if not _case_supported(c, supported_tokens, ctx_cases)
    ]

# Phrases by which an answer explicitly disclaims that some opinion/material is
# in its context. A justice or case named only inside such a statement is being
# flagged as ABSENT by the model itself — that is honesty, not hallucination, so
# it must not be treated as an unsupported authority.
_DISCLAIMER = re.compile(
    r"\b("
    r"do(?:es)?\s+not\s+(?:include|contain|mention|appear|address)"
    r"|not\s+(?:included|contained|provided|mentioned|present)"
    r"|cannot\s+be\s+(?:identified|determined|found)"
    r"|is\s+not\s+(?:in|part\s+of|included\s+in)\s+the\s+(?:provided|furnished|given)?"
    r"|no\s+(?:text|portion|part|excerpt)\s+of"
    r"|material\s+(?:furnished|provided|given)"
    r"|not\s+in\s+the\s+(?:provided|furnished|given|supplied)\s+(?:material|context|record)"
    r")",
    re.IGNORECASE,
)


def _split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text or "")


def _disclaimed_justices(answer: str) -> set:
    """Surnames named only inside an explicit 'not in the provided material' clause."""
    disclaimed = set()
    for sent in _split_sentences(answer):
        if not _DISCLAIMER.search(sent):
            continue
        for m in _JUSTICE_TITLED.findall(sent):
            disclaimed.add(_surname(m))
        for m in _JUSTICE_TAG.findall(sent):
            s = _surname(m)
            if s and s not in _COMMON_WORDS:
                disclaimed.add(s)
    return disclaimed


def _disclaimed_cases(answer: str) -> set:
    """Case names appearing only inside an explicit disclaimer clause."""
    disclaimed = set()
    for sent in _split_sentences(answer):
        if _DISCLAIMER.search(sent):
            disclaimed |= cases_in(sent)
    return disclaimed


def check_answer(answer: str, context: str, segments: List, case_title: str = "") -> dict:
    """Return faithfulness findings for one answer."""
    ctx_justices = justices_in(context) | justices_from_segments(segments)
    ctx_cases = cases_in(context) | case_pair_from_title(case_title)

    ans_justices = justices_in(answer)
    ans_cases = cases_in(answer)

    # A justice/case the answer explicitly says is NOT in the provided material
    # is being disclaimed, not asserted as authority — don't flag honesty.
    disclaimed_justices = _disclaimed_justices(answer)
    disclaimed_cases = _disclaimed_cases(answer)

    unsupported_justices = sorted((ans_justices - ctx_justices) - disclaimed_justices)
    unsupported_cases = [
        c for c in _unsupported_cases(
            ans_cases=ans_cases,
            ctx_cases=ctx_cases,
            context=context,
            case_title=case_title,
        )
        if c not in disclaimed_cases
    ]

    author_roles = _segment_author_roles(segments)
    role_conflicts = []

    for sn, claimed_fam in _claimed_attributions(answer):
        actual = author_roles.get(sn)

        if not actual:
            continue

        if "mixed" in actual:
            continue

        if claimed_fam not in actual:
            msg = (
                f"answer calls {sn} '{claimed_fam}' but segment role is "
                f"'{'/'.join(sorted(actual))}'"
            )
            if msg not in role_conflicts:
                role_conflicts.append(msg)

    flagged = bool(unsupported_justices or unsupported_cases or role_conflicts)

    return {
        "flagged": flagged,
        "unsupported_justices": unsupported_justices,
        "unsupported_cases": unsupported_cases,
        "role_conflicts": role_conflicts,
    }


def summarize(findings: dict) -> str:
    if not findings["flagged"]:
        return "✅ faithful (no unsupported authorities or role conflicts detected)"

    bits = []

    if findings["unsupported_justices"]:
        bits.append(
            "unsupported justice(s): "
            + ", ".join(findings["unsupported_justices"])
        )

    if findings["unsupported_cases"]:
        bits.append(
            "unsupported case(s): "
            + ", ".join(findings["unsupported_cases"])
        )

    if findings["role_conflicts"]:
        bits.append(
            "role conflict(s): "
            + "; ".join(findings["role_conflicts"])
        )

    return "⚠️ " + " | ".join(bits)