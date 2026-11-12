"""Turn a public search-engine result for a LinkedIn profile into structured fields.

Google renders public LinkedIn profiles with fairly stable title/snippet shapes, e.g.

    title:   "Jane Doe - Principal - Example Public School | LinkedIn"
    snippet: "Experience: Example Public School · Education: ... · Location: New Delhi ·
              500+ connections on LinkedIn. View Jane Doe's profile on LinkedIn ..."

Nothing here touches linkedin.com itself; we only read what the search API returned.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

_PROFILE_RE = re.compile(r"linkedin\.com/in/([^/?#]+)", re.IGNORECASE)
_TITLE_SUFFIX_RE = re.compile(r"\s*[|\-–—]\s*LinkedIn\s*$", re.IGNORECASE)
_SEP_RE = re.compile(r"\s+[\-–—|]\s+")

ROLE_WORDS = (
    "principal", "director", "founder", "co-founder", "owner", "trustee", "chairman", "chairperson",
    "consultant", "advisor", "adviser", "head", "dean", "manager", "ceo", "coo", "cto", "president",
    "teacher", "coordinator", "vice", "lead", "officer", "mentor", "coach", "trainer", "secretary",
    "headmaster", "headmistress", "academic", "educator", "specialist", "partner", "executive",
    "member", "board", "retired", "former", "ex-", "professor", "lecturer", "counsellor", "counselor",
)

_FIELD_RE = {
    "location": re.compile(r"Location:\s*([^·|\n]+?)(?:\s*·|\s*\||\.\s|$)", re.IGNORECASE),
    "experience": re.compile(r"Experience:\s*([^·|\n]+?)(?:\s*·|\s*\||$)", re.IGNORECASE),
    "education": re.compile(r"Education:\s*([^·|\n]+?)(?:\s*·|\s*\||$)", re.IGNORECASE),
}
# Leading "City, State, India · ..." style location.
_LEAD_LOC_RE = re.compile(r"^([A-Z][A-Za-z .]+(?:,\s*[A-Z][A-Za-z .]+){0,2})\s*·")
_CONNECTIONS_RE = re.compile(r"(\d[\d,]*\+?)\s+(?:connections|followers)", re.IGNORECASE)


def profile_slug(url: str) -> str | None:
    """Return the canonical LinkedIn slug for a /in/ profile URL, else None."""
    if not url:
        return None
    match = _PROFILE_RE.search(url)
    if not match:
        return None
    slug = unquote(match.group(1)).strip().strip("/").lower()
    return slug or None


def normalise_profile_url(url: str) -> str | None:
    """Map country subdomains (in.linkedin.com) and tracking params to one canonical URL."""
    slug = profile_slug(url)
    if not slug:
        return None
    host = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
    if "linkedin.com" not in host:
        return None
    return f"https://www.linkedin.com/in/{slug}"


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.rstrip(" .…").replace("...", "").strip(" -–—|·")


def _looks_like_role(text: str) -> bool:
    low = text.lower()
    return any(w in low for w in ROLE_WORDS)


def parse_title(title: str) -> dict[str, str]:
    """Split a SERP title into name / headline / organisation."""
    cleaned = _TITLE_SUFFIX_RE.sub("", title or "").strip()
    cleaned = cleaned.replace("...", "").replace("…", "").strip()
    parts = [p.strip() for p in _SEP_RE.split(cleaned) if p.strip()]
    out = {"name": "", "headline": "", "organisation": ""}
    if not parts:
        return out
    out["name"] = _clean(parts[0])
    if len(parts) == 2:
        if _looks_like_role(parts[1]):
            out["headline"] = _clean(parts[1])
        else:
            out["organisation"] = _clean(parts[1])
    elif len(parts) >= 3:
        out["headline"] = _clean(" - ".join(parts[1:-1]))
        out["organisation"] = _clean(parts[-1])
    return out


def parse_snippet(snippet: str) -> dict[str, str]:
    """Pull location / experience / education / connections out of a SERP snippet."""
    snippet = re.sub(r"\s+", " ", snippet or "").strip()
    out: dict[str, str] = {}
    for key, rx in _FIELD_RE.items():
        m = rx.search(snippet)
        if m:
            out[key] = _clean(m.group(1))
    if "location" not in out:
        m = _LEAD_LOC_RE.match(snippet)
        if m and not _looks_like_role(m.group(1)):
            out["location"] = _clean(m.group(1))
    m = _CONNECTIONS_RE.search(snippet)
    if m:
        out["connections"] = m.group(1)
    return out


def clean_snippet(snippet: str) -> str:
    """Drop LinkedIn boilerplate so embeddings focus on the professional content."""
    text = re.sub(r"\s+", " ", snippet or "")
    text = re.sub(r"View [^.]{0,80}?profile on LinkedIn[^.]*\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"a professional community of [\d.,]+\s*\w* members\.?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\d[\d,]*\+?\s+(?:connections|followers)( on LinkedIn)?\.?", "", text, flags=re.IGNORECASE)
    return text.strip(" ·.")


def parse_result(title: str, snippet: str) -> dict[str, str]:
    """Combine title and snippet parsing into one record."""
    t = parse_title(title)
    s = parse_snippet(snippet)
    organisation = t["organisation"] or s.get("experience", "")
    headline = t["headline"]
    role = headline
    # "Principal at Example School" → role / organisation
    m = re.match(r"(.+?)\s+(?:at|@)\s+(.+)", headline or "", flags=re.IGNORECASE)
    if m:
        role = m.group(1).strip()
        organisation = organisation or m.group(2).strip()
    return {
        "name": t["name"],
        "headline": headline,
        "role": role,
        "organisation": organisation,
        "location": s.get("location", ""),
        "education": s.get("education", ""),
        "connections": s.get("connections", ""),
        "snippet": clean_snippet(snippet),
    }
