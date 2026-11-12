"""End-to-end orchestration:

Reference profile → AI analysis → query planning → discovery (search API) →
parse + de-duplicate → hybrid ranking → persist + export.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .analyzer import analyse_reference, plan_queries
from .config import RUNS_DIR, Settings
from .llm import LLMClient
from .models import Candidate, DiscoveryQuery, Persona, RankedCandidate, ReferenceProfile, ScoreBreakdown, SearchHit
from .parser import normalise_profile_url, parse_result, profile_slug
from .ranker import Embedder, rank
from .search import SearchClient, SearchError
from .storage import Storage, utcnow

Progress = Callable[[str, float], None]


@dataclass
class RunResult:
    run_id: str
    created_at: str
    reference: ReferenceProfile
    persona: Persona
    queries: list[DiscoveryQuery]
    hits: list[SearchHit]
    candidates: list[Candidate]
    ranked: list[RankedCandidate]
    stats: dict[str, Any] = field(default_factory=dict)

    # ---- serialisation (used for exports and the frozen demo snapshot) -------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "reference": asdict(self.reference),
            "persona": self.persona.to_dict(),
            "queries": [asdict(q) for q in self.queries],
            "hits": [asdict(h) for h in self.hits],
            "ranked": [
                {
                    "rank": r.rank,
                    "archetype": r.archetype,
                    "explanation": r.explanation,
                    "evidence": r.evidence,
                    "signals": r.signals,
                    "scores": asdict(r.scores),
                    "candidate": asdict(r.candidate),
                }
                for r in self.ranked
            ],
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunResult":
        ranked = [
            RankedCandidate(
                candidate=Candidate(**r["candidate"]),
                scores=ScoreBreakdown(**r["scores"]),
                archetype=r.get("archetype", "other"),
                explanation=r.get("explanation", ""),
                evidence=r.get("evidence", []),
                signals=r.get("signals", []),
                rank=r.get("rank", 0),
            )
            for r in d.get("ranked", [])
        ]
        return cls(
            run_id=d["run_id"],
            created_at=d.get("created_at", ""),
            reference=ReferenceProfile(**d["reference"]),
            persona=Persona(**d["persona"]),
            queries=[DiscoveryQuery(**q) for q in d.get("queries", [])],
            hits=[SearchHit(**h) for h in d.get("hits", [])],
            candidates=[r.candidate for r in ranked],
            ranked=ranked,
            stats=d.get("stats", {}),
        )

    def save_json(self, directory: Path = RUNS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load_json(cls, path: Path | str) -> "RunResult":
        return cls.from_dict(json.loads(Path(path).read_text()))


def lookup_reference(url: str, search: SearchClient) -> str:
    """Fetch the public search snippet for the reference profile itself (no LinkedIn access)."""
    slug = profile_slug(url)
    if not slug:
        raise ValueError("That doesn't look like a linkedin.com/in/ profile URL")
    for q in (f"site:linkedin.com/in/{slug}", f'"linkedin.com/in/{slug}"'):
        for h in search.search(q, num=10):
            if profile_slug(h.url) == slug:
                return f"{h.title}\n{h.snippet}".strip()
    return ""


def build_candidates(hits: list[SearchHit], exclude_slugs: set[str]) -> list[Candidate]:
    by_slug: dict[str, Candidate] = {}
    for h in hits:
        slug = profile_slug(h.url)
        url = normalise_profile_url(h.url)
        if not slug or not url or slug in exclude_slugs:
            continue
        parsed = parse_result(h.title, h.snippet)
        if not parsed["name"]:
            continue
        if slug in by_slug:
            c = by_slug[slug]
            if h.query not in c.source_queries:
                c.source_queries.append(h.query)
            c.best_position = min(c.best_position, h.position)
            # keep the richer snippet when the same person shows up twice
            if len(parse_result(h.title, h.snippet)["snippet"]) > len(c.snippet):
                c.snippet = parsed["snippet"]
                c.location = c.location or parsed["location"]
            continue
        by_slug[slug] = Candidate(
            profile_id=slug,
            url=url,
            name=parsed["name"],
            headline=parsed["headline"],
            role=parsed["role"],
            organisation=parsed["organisation"],
            location=parsed["location"],
            snippet=parsed["snippet"],
            raw_title=h.title,
            source_queries=[h.query],
            best_position=h.position,
            retrieved_at=h.retrieved_at,
        )
    return list(by_slug.values())


def run_discovery(
    reference: ReferenceProfile,
    settings: Settings,
    search: SearchClient,
    llm: LLMClient | None,
    embedder: Embedder,
    storage: Storage | None = None,
    progress: Progress | None = None,
) -> RunResult:
    def report(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    t0 = time.time()
    run_id = time.strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    if storage:
        storage.create_run(run_id, reference.url, reference.text, {
            "max_queries": settings.max_queries, "llm_rerank_top_k": settings.llm_rerank_top_k,
            "weights": asdict(settings.weights), "groq_model": settings.groq_model,
            "embedding": embedder.backend, "country": settings.country,
        })

    report("Analysing the reference profile", 0.05)
    persona = analyse_reference(reference, llm)

    report("Planning discovery queries", 0.15)
    queries = plan_queries(persona, llm, settings.max_queries)

    hits: list[SearchHit] = []
    errors: list[str] = []
    for i, q in enumerate(queries):
        report(f"Searching ({i + 1}/{len(queries)}): {q.query}", 0.20 + 0.40 * i / max(len(queries), 1))
        try:
            found = search.search(q.query, num=settings.results_per_query)
        except SearchError as e:
            errors.append(f"{q.query}: {e}")
            if "rejected" in str(e):
                raise
            continue
        q.results_count = len(found)
        hits.extend(found)

    exclude = {s for s in [profile_slug(reference.url)] if s}
    candidates = build_candidates(hits, exclude)
    report(f"Found {len(candidates)} unique profiles — ranking", 0.65)

    def llm_progress(done: int, total: int) -> None:
        report(f"AI relevance review: batch {done}/{total}", 0.65 + 0.30 * done / max(total, 1))

    ranked = rank(
        persona, candidates, embedder, settings.weights, llm=llm,
        llm_top_k=settings.llm_rerank_top_k, batch_size=settings.llm_batch_size, progress=llm_progress,
    )

    stats = {
        "queries": len(queries),
        "raw_hits": len(hits),
        "profile_hits": sum(1 for h in hits if profile_slug(h.url)),
        "unique_profiles": len(candidates),
        "llm_judged": sum(1 for r in ranked if r.scores.llm_ecosystem is not None),
        "score_ge_60": sum(1 for r in ranked if r.scores.final >= 0.60),
        "search_calls": getattr(search, "calls_made", None),
        "llm_calls": getattr(llm, "calls_made", None) if llm else 0,
        "llm_tokens": getattr(llm, "tokens_used", None) if llm else 0,
        "embedding_backend": embedder.backend,
        "persona_source": persona.source,
        "errors": errors,
        "seconds": round(time.time() - t0, 1),
    }
    result = RunResult(run_id, utcnow(), reference, persona, queries, hits, candidates, ranked, stats)
    if storage:
        storage.save_queries(run_id, queries)
        storage.save_hits(run_id, hits)
        storage.save_candidates(run_id, candidates)
        storage.save_scores(run_id, ranked)
        storage.finish_run(run_id, persona, stats)
    report("Done", 1.0)
    return result
