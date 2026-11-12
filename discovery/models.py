"""Plain dataclasses passed between pipeline stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# The five people-types the assignment asks us to find.
ARCHETYPES = {
    "consultant": "School / education consultant or advisor",
    "school_leader": "Current or former Principal, School Director, Owner or Trustee",
    "entrepreneur": "Education entrepreneur or school-network professional",
    "multi_school": "Professional who works with many schools (academic, partnerships, training)",
    "connector": "Has working relationships with school decision-makers (associations, boards, vendors to schools)",
    "other": "Not clearly part of the school ecosystem",
}


@dataclass
class ReferenceProfile:
    url: str
    text: str
    name: str = ""


@dataclass
class Persona:
    """Structured understanding of the reference profile, produced by the LLM."""

    name: str = ""
    headline: str = ""
    summary: str = ""
    current_role: str = ""
    organisation: str = ""
    location: str = ""
    seniority: str = ""
    archetypes: list[str] = field(default_factory=list)
    ecosystem_signals: list[str] = field(default_factory=list)
    boards: list[str] = field(default_factory=list)
    geographies: list[str] = field(default_factory=list)
    org_types: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source: str = "llm"  # "llm" or "heuristic"

    def as_text(self) -> str:
        """Text used as the 'query side' of the embedding similarity."""
        parts = [
            self.headline,
            self.summary,
            f"Role: {self.current_role}" if self.current_role else "",
            f"Organisation: {self.organisation}" if self.organisation else "",
            "Signals: " + ", ".join(self.ecosystem_signals) if self.ecosystem_signals else "",
            "Boards: " + ", ".join(self.boards) if self.boards else "",
            "Works with: " + ", ".join(self.org_types) if self.org_types else "",
            "Keywords: " + ", ".join(self.keywords) if self.keywords else "",
        ]
        return ". ".join(p for p in parts if p)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryQuery:
    query: str
    archetype: str = "other"
    source: str = "llm"  # "llm" or "template"
    results_count: int = 0


@dataclass
class SearchHit:
    query: str
    position: int
    url: str
    title: str
    snippet: str
    retrieved_at: str
    from_cache: bool = False


@dataclass
class Candidate:
    profile_id: str  # LinkedIn slug, used for de-duplication
    url: str
    name: str
    headline: str = ""
    role: str = ""
    organisation: str = ""
    location: str = ""
    snippet: str = ""
    raw_title: str = ""
    source_queries: list[str] = field(default_factory=list)
    best_position: int = 99
    retrieved_at: str = ""

    def text(self) -> str:
        """Text used as the 'document side' of the embedding similarity."""
        parts = [self.headline or self.role, self.organisation, self.location, self.snippet]
        return ". ".join(p for p in parts if p)


@dataclass
class ScoreBreakdown:
    semantic: float = 0.0
    llm_ecosystem: float | None = None  # 0-1, None if the LLM did not judge this one
    llm_influence: float | None = None
    rules: float = 0.0
    final: float = 0.0

    @property
    def llm(self) -> float | None:
        if self.llm_ecosystem is None:
            return None
        influence = self.llm_influence if self.llm_influence is not None else self.llm_ecosystem
        return 0.6 * self.llm_ecosystem + 0.4 * influence


@dataclass
class RankedCandidate:
    candidate: Candidate
    scores: ScoreBreakdown
    archetype: str = "other"
    explanation: str = ""
    evidence: list[str] = field(default_factory=list)  # phrases verified to appear in the source text
    signals: list[str] = field(default_factory=list)  # rule-based signals that fired
    rank: int = 0

    def to_row(self) -> dict[str, Any]:
        c, s = self.candidate, self.scores
        return {
            "rank": self.rank,
            "name": c.name,
            "linkedin_url": c.url,
            "role": c.role or c.headline,
            "organisation": c.organisation,
            "location": c.location,
            "archetype": ARCHETYPES.get(self.archetype, self.archetype),
            "relevance_score": round(s.final * 100, 1),
            "semantic": round(s.semantic, 3),
            "llm_ecosystem": None if s.llm_ecosystem is None else round(s.llm_ecosystem, 3),
            "llm_influence": None if s.llm_influence is None else round(s.llm_influence, 3),
            "rules": round(s.rules, 3),
            "why": self.explanation,
            "evidence": " | ".join(self.evidence),
            "signals": ", ".join(self.signals),
            "found_via": " || ".join(c.source_queries),
            "retrieved_at": c.retrieved_at,
            "source_snippet": c.snippet,
        }
