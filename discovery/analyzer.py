"""Stage 1-2: understand the reference profile and plan discovery queries."""
from __future__ import annotations

import re
from typing import Any

from .llm import LLMClient, LLMError
from .models import ARCHETYPES, DiscoveryQuery, Persona, ReferenceProfile

PERSONA_SYSTEM = """You analyse a LinkedIn profile from the Indian school-education ecosystem for a
school-network outreach team. Return ONLY a JSON object. Base every field on the profile text;
do not invent employers, places or titles. Use empty strings/lists when unknown."""

PERSONA_USER = """Profile URL: {url}

Profile text:
\"\"\"{text}\"\"\"

Return JSON with these keys:
- name: person's name if stated
- headline: one-line professional headline
- summary: 2-3 sentences on what this person does and why they matter to schools
- current_role, organisation, location
- seniority: one of "founder/owner", "executive", "school leadership", "senior", "mid", "junior"
- archetypes: list from {archetypes}
- ecosystem_signals: 4-8 short phrases showing school-ecosystem involvement (e.g. "ran a CBSE school", "advises 40+ schools")
- boards: school boards/curricula mentioned (CBSE, ICSE, IB, Cambridge/IGCSE, State Board)
- geographies: cities/regions they operate in, most specific first (max 4)
- org_types: kinds of organisations they work with (e.g. "K-12 private schools", "school chains", "edtech")
- keywords: 6-10 search keywords that describe people similar to them (roles, not names)"""

QUERY_SYSTEM = """You write Google search queries that find PUBLIC LinkedIn profiles of people similar to a
reference person in the Indian school-education ecosystem. Return ONLY JSON."""

QUERY_USER = """Reference persona:
{persona}

Target people (archetype keys): {archetypes}

Write {n} diverse queries. Rules:
- every query starts with: site:linkedin.com/in
- 2-4 short quoted phrases per query, e.g. site:linkedin.com/in "former principal" "CBSE" "Gurugram"
- most queries include one geography from the persona; a few can be national
- cover every archetype; vary vocabulary (principal / head of school / director academics / trustee /
  school chain / Sahodaya / school consultant / academic advisor / school partnerships ...)
- never include the reference person's own name
Return: {{"queries": [{{"query": "...", "archetype": "<archetype key>"}}]}}"""

TEMPLATES: dict[str, list[str]] = {
    "consultant": ['"education consultant" "school"', '"school consultant"', '"academic consultant" "schools"',
                   '"school advisor"'],
    "school_leader": ['"former principal"', '"school director"', '"trustee" "school"', '"retired principal"',
                      '"school owner"', '"chairman" "public school"'],
    "entrepreneur": ['"founder" "chain of schools"', '"education entrepreneur"', '"founder" "school management"',
                     '"group of schools" "director"'],
    "multi_school": ['"school partnerships"', '"academic head" "schools"', '"teacher training" "schools"',
                     '"CBSE" "resource person"'],
    "connector": ['"principals association"', '"school relations"', '"school outreach"', '"Sahodaya"'],
}

ROLE_HINTS = {
    "principal": "school_leader", "headmaster": "school_leader", "head of school": "school_leader",
    "trustee": "school_leader", "director": "school_leader", "owner": "school_leader",
    "consultant": "consultant", "advisor": "consultant", "adviser": "consultant",
    "founder": "entrepreneur", "entrepreneur": "entrepreneur", "chain": "entrepreneur",
    "schools": "multi_school", "training": "multi_school", "association": "connector",
}
BOARDS = ["CBSE", "ICSE", "CISCE", "IB", "IGCSE", "Cambridge", "State Board", "NIOS"]
INDIAN_PLACES = [
    "Delhi", "New Delhi", "Gurugram", "Gurgaon", "Noida", "Ghaziabad", "Faridabad", "NCR", "Mumbai", "Pune",
    "Bengaluru", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Indore", "Bhopal", "Dehradun", "Kochi", "Coimbatore", "Nagpur", "Patna", "India",
]


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [v.strip() for v in re.split(r"[;,]", value) if v.strip()]
    return []


def heuristic_persona(ref: ReferenceProfile) -> Persona:
    """Keyword fallback used when no LLM key is configured."""
    text = ref.text or ""
    low = text.lower()
    archetypes = sorted({a for hint, a in ROLE_HINTS.items() if hint in low}) or ["school_leader"]
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return Persona(
        name=ref.name,
        headline=first_line[:160],
        summary=text[:400],
        archetypes=archetypes,
        ecosystem_signals=[w for w in ["school", "principal", "education", "academic", "curriculum", "trustee"]
                           if w in low],
        boards=[b for b in BOARDS if re.search(rf"\b{re.escape(b)}\b", text)],
        geographies=[p for p in INDIAN_PLACES if re.search(rf"\b{re.escape(p)}\b", text)][:4],
        keywords=list(dict.fromkeys(w for w in ROLE_HINTS if w in low))[:8],
        source="heuristic",
    )


def analyse_reference(ref: ReferenceProfile, llm: LLMClient | None) -> Persona:
    if llm is None:
        return heuristic_persona(ref)
    try:
        data = llm.chat_json(
            PERSONA_SYSTEM,
            PERSONA_USER.format(url=ref.url, text=ref.text[:6000], archetypes=list(ARCHETYPES)[:-1]),
            max_tokens=1200,
        )
    except (LLMError, Exception):
        return heuristic_persona(ref)
    persona = Persona(
        name=str(data.get("name") or ref.name or ""),
        headline=str(data.get("headline", "")),
        summary=str(data.get("summary", "")),
        current_role=str(data.get("current_role", "")),
        organisation=str(data.get("organisation", "")),
        location=str(data.get("location", "")),
        seniority=str(data.get("seniority", "")),
        archetypes=[a for a in _as_list(data.get("archetypes")) if a in ARCHETYPES] or ["school_leader"],
        ecosystem_signals=_as_list(data.get("ecosystem_signals")),
        boards=_as_list(data.get("boards")),
        geographies=_as_list(data.get("geographies"))[:4],
        org_types=_as_list(data.get("org_types")),
        keywords=_as_list(data.get("keywords")),
        source="llm",
    )
    if not persona.geographies:
        persona.geographies = heuristic_persona(ref).geographies
    return persona


def _normalise_query(q: str) -> str:
    q = re.sub(r"\s+", " ", q.replace("“", '"').replace("”", '"')).strip()
    if not q.lower().startswith("site:linkedin.com/in"):
        q = "site:linkedin.com/in " + re.sub(r"^site:\S+\s*", "", q)
    return q


def template_queries(persona: Persona) -> list[DiscoveryQuery]:
    geos = persona.geographies or ["India"]
    board = next((b for b in persona.boards if b.upper() in {"CBSE", "ICSE", "IB"}), "")
    out: list[DiscoveryQuery] = []
    # Round-robin across archetypes so every target type gets coverage even with a small budget.
    for i in range(max(len(v) for v in TEMPLATES.values())):
        for arch, phrases in TEMPLATES.items():
            if i >= len(phrases):
                continue
            geo = geos[i % len(geos)]
            extra = f' "{board}"' if board and i % 3 == 1 and board not in phrases[i] else ""
            out.append(DiscoveryQuery(f'site:linkedin.com/in {phrases[i]}{extra} "{geo}"', arch, "template"))
    return out


def plan_queries(persona: Persona, llm: LLMClient | None, max_queries: int = 24) -> list[DiscoveryQuery]:
    llm_queries: list[DiscoveryQuery] = []
    if llm is not None:
        try:
            persona_brief = {k: v for k, v in persona.to_dict().items() if k not in {"source", "name"}}
            data = llm.chat_json(
                QUERY_SYSTEM,
                QUERY_USER.format(persona=persona_brief, archetypes=list(ARCHETYPES)[:-1],
                                  n=max(6, int(max_queries * 0.6))),
                max_tokens=1500,
                temperature=0.4,
            )
            for item in data.get("queries", []) or []:
                q = item.get("query") if isinstance(item, dict) else str(item)
                arch = item.get("archetype", "other") if isinstance(item, dict) else "other"
                if q:
                    llm_queries.append(DiscoveryQuery(_normalise_query(q), arch if arch in ARCHETYPES else "other", "llm"))
        except Exception:
            llm_queries = []

    name = (persona.name or "").lower().strip()
    seen: set[str] = set()
    final: list[DiscoveryQuery] = []
    budget_llm = int(max_queries * 0.6) if llm_queries else 0
    for q in llm_queries[:budget_llm] + template_queries(persona):
        key = q.query.lower()
        if key in seen or (name and len(name) > 3 and name in key):
            continue
        seen.add(key)
        final.append(q)
        if len(final) >= max_queries:
            break
    return final
