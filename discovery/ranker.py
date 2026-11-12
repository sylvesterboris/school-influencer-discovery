"""Stage 4: hybrid ranking.

final = w_sem * semantic_similarity + w_llm * llm_rubric + w_rules * rule_features

* semantic_similarity — embedding cosine between the reference persona and the candidate text
  (fastembed / BAAI bge-small-en-v1.5, ONNX, CPU). Falls back to TF-IDF if the model can't load.
* llm_rubric          — an LLM judges school-ecosystem relevance and influence (0-10 each) and must
  quote evidence; quotes that do not appear in the source text are discarded.
* rule_features       — transparent keyword/seniority/location signals, so the score never depends
  on the black box alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import Weights
from .llm import LLMClient
from .models import ARCHETYPES, Candidate, Persona, RankedCandidate, ScoreBreakdown

# --------------------------------------------------------------------------- embeddings


class Embedder:
    """fastembed when available, TF-IDF otherwise. Exposes .backend for the UI/technical note."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", allow_neural: bool = True):
        self.model_name = model_name
        self.backend = "tfidf"
        self._model = None
        if allow_neural:
            try:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(model_name=model_name)
                self.backend = f"fastembed:{model_name}"
            except Exception:
                self._model = None

    def similarities(self, query_text: str, docs: list[str]) -> np.ndarray:
        if not docs:
            return np.zeros(0)
        if self._model is not None:
            try:
                vecs = np.array(list(self._model.embed([query_text] + docs)))
                vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
                cos = vecs[1:] @ vecs[0]
                # bge-small cosines for this domain sit roughly in [0.55, 0.88]; rescale to 0-1.
                return np.clip((cos - 0.55) / (0.88 - 0.55), 0.0, 1.0)
            except Exception:
                self.backend = "tfidf"
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        mat = vec.fit_transform([query_text] + docs)
        cos = (mat[1:] @ mat[0].T).toarray().ravel()
        hi = cos.max() if cos.size else 0
        return cos / hi if hi > 0 else cos


# --------------------------------------------------------------------------- rule features

_ROLE_WEIGHTS: list[tuple[str, float, str]] = [
    (r"\b(principal|headmaster|headmistress|head of school|school head)\b", 0.95, "school leadership title"),
    (r"\b(trustee|chairman|chairperson|chair person|school owner|owner)\b", 0.9, "owner/trustee"),
    (r"\b(founder|co-founder|cofounder|managing director)\b", 0.85, "founder"),
    (r"\b(director|dean|ceo|president)\b", 0.8, "director-level"),
    (r"\b(consultant|advisor|adviser|mentor)\b", 0.75, "consultant/advisor"),
    (r"\b(vice principal|academic head|head academics|academic director|coordinator)\b", 0.7, "academic leadership"),
    (r"\b(teacher trainer|resource person|master trainer)\b", 0.65, "trainer for schools"),
    (r"\b(teacher|educator|faculty)\b", 0.35, "teacher"),
]
_EDU_RE = re.compile(
    r"\b(school|schools|vidyalaya|vidya|academy|education|educational|cbse|icse|cisce|ib|igcse|cambridge|"
    r"k-12|k12|curriculum|academic|sahodaya|edtech|pedagogy|learning)\b",
    re.IGNORECASE,
)
_MULTI_RE = re.compile(
    r"(\b\d{2,4}\+?\s*schools\b|group of schools|chain of schools|network of schools|across schools|"
    r"multiple schools|schools across|school network|school partnerships|associations?\b)",
    re.IGNORECASE,
)
_FORMER_RE = re.compile(r"\b(former|ex-|retired|ex\s)", re.IGNORECASE)
_NEGATIVE_RE = re.compile(r"\b(student|intern|fresher|aspiring|looking for opportunities|class \d+)\b", re.IGNORECASE)


def rule_features(c: Candidate, persona: Persona) -> tuple[float, list[str]]:
    text = f"{c.headline} {c.role} {c.organisation} {c.snippet}"
    signals: list[str] = []
    role_score = 0.0
    for pattern, weight, label in _ROLE_WEIGHTS:
        if re.search(pattern, text, re.IGNORECASE):
            if weight > role_score:
                role_score = weight
            signals.append(label)
            break
    edu_hits = len(set(m.lower() for m in _EDU_RE.findall(text)))
    edu = min(edu_hits / 3, 1.0)
    if edu_hits:
        signals.append("education context")
    multi = 1.0 if _MULTI_RE.search(text) else 0.0
    if multi:
        signals.append("works across schools")
    if _FORMER_RE.search(text):
        signals.append("former/retired leader")
    loc = 0.0
    loc_text = f"{c.location} {c.snippet}".lower()
    for g in persona.geographies:
        if g and g.lower() in loc_text:
            loc = 1.0
            signals.append(f"same region ({g})")
            break
    for b in persona.boards:
        if b and re.search(rf"\b{re.escape(b)}\b", text, re.IGNORECASE):
            signals.append(f"{b} board")
            edu = min(edu + 0.2, 1.0)
            break
    score = 0.45 * role_score + 0.30 * edu + 0.15 * multi + 0.10 * loc
    if role_score > 0 and edu_hits == 0:
        score *= 0.6  # a "director" with no education context is weak evidence
    if _NEGATIVE_RE.search(text):
        score *= 0.4
        signals.append("junior/student signal")
    return float(np.clip(score, 0.0, 1.0)), signals


# --------------------------------------------------------------------------- LLM rubric

RUBRIC_SYSTEM = """You screen public LinkedIn search results for a school-network outreach team in India.
Judge ONLY from the given text. Never invent facts. Return ONLY JSON."""

RUBRIC_USER = """Reference person (what "relevant" looks like):
{persona}

Target archetypes:
{archetypes}

For each candidate return:
- id
- archetype: one key from the list above ("other" if not school ecosystem)
- ecosystem_relevance: 0-10, how clearly they work in/with K-12 schools (10 = runs or advises schools)
- influence: 0-10, how likely they are to reach school decision-makers (seniority, many schools, ownership)
- evidence: 1-3 SHORT phrases copied EXACTLY from that candidate's text that justify the scores
- why: one sentence (max 30 words) explaining the match to the reference person

Candidates:
{candidates}

Return: {{"results": [{{"id": "...", "archetype": "...", "ecosystem_relevance": 0, "influence": 0, "evidence": ["..."], "why": "..."}}]}}"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip(" .\"'")


def grounded_evidence(phrases: list[str], source: str, limit: int = 3) -> list[str]:
    """Keep only phrases that really occur in the candidate's source text (anti-hallucination)."""
    src = _norm(source)
    out = []
    for p in phrases or []:
        p = str(p).strip()
        if len(p) >= 3 and _norm(p) in src and p not in out:
            out.append(p)
        if len(out) >= limit:
            break
    return out


@dataclass
class LLMJudgement:
    archetype: str
    ecosystem: float
    influence: float
    evidence: list[str]
    why: str


def llm_judge(
    persona: Persona,
    candidates: list[Candidate],
    llm: LLMClient,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, LLMJudgement]:
    persona_brief = (
        f"{persona.headline}. {persona.summary} Signals: {', '.join(persona.ecosystem_signals)}. "
        f"Region: {', '.join(persona.geographies)}."
    )
    arch_text = "\n".join(f"- {k}: {v}" for k, v in ARCHETYPES.items())
    results: dict[str, LLMJudgement] = {}
    batches = [candidates[i : i + batch_size] for i in range(0, len(candidates), batch_size)]
    for bi, batch in enumerate(batches):
        lines = []
        for idx, c in enumerate(batch):
            lines.append(
                f'[{idx}] id={c.profile_id}\n    title: {c.raw_title}\n    location: {c.location or "unknown"}\n'
                f"    text: {c.snippet[:450]}"
            )
        try:
            data = llm.chat_json(
                RUBRIC_SYSTEM,
                RUBRIC_USER.format(persona=persona_brief, archetypes=arch_text, candidates="\n".join(lines)),
                max_tokens=220 * len(batch) + 200,
            )
        except Exception:
            data = {}
        by_id = {c.profile_id: c for c in batch}
        for item in data.get("results", []) or []:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id", "")).strip()
            if pid not in by_id:
                # models sometimes echo the bracket index instead of the id
                m = re.fullmatch(r"\[?(\d+)\]?", pid)
                if m and int(m.group(1)) < len(batch):
                    pid = batch[int(m.group(1))].profile_id
                else:
                    continue
            c = by_id[pid]

            def _score(v: object) -> float:
                try:
                    return float(np.clip(float(v) / 10.0, 0.0, 1.0))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return 0.0

            arch = str(item.get("archetype", "other"))
            results[pid] = LLMJudgement(
                archetype=arch if arch in ARCHETYPES else "other",
                ecosystem=_score(item.get("ecosystem_relevance")),
                influence=_score(item.get("influence")),
                evidence=grounded_evidence(item.get("evidence") or [], f"{c.raw_title} {c.snippet} {c.location}"),
                why=str(item.get("why", "")).strip()[:300],
            )
        if progress:
            progress(bi + 1, len(batches))
    return results


# --------------------------------------------------------------------------- combine


def _rule_explanation(signals: list[str], c: Candidate) -> str:
    useful = [s for s in signals if s != "education context"]
    role = c.role or c.headline or "profile"
    if not useful:
        return f"{role} — matched on general education context only."
    return f"{role}: " + "; ".join(useful) + "."


def _guess_archetype(signals: list[str]) -> str:
    joined = " ".join(signals)
    if "consultant" in joined:
        return "consultant"
    if "founder" in joined:
        return "entrepreneur"
    if "works across schools" in joined or "trainer" in joined:
        return "multi_school"
    if any(k in joined for k in ("school leadership", "owner/trustee", "director-level", "academic leadership")):
        return "school_leader"
    return "other"


def rank(
    persona: Persona,
    candidates: list[Candidate],
    embedder: Embedder,
    weights: Weights,
    llm: LLMClient | None = None,
    llm_top_k: int = 60,
    batch_size: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[RankedCandidate]:
    if not candidates:
        return []
    w = weights.normalised()
    sem = embedder.similarities(persona.as_text(), [c.text() for c in candidates])
    rules = [rule_features(c, persona) for c in candidates]

    # Pre-rank without the LLM, then spend LLM calls only on the most promising candidates.
    base_w = w.semantic + w.rules
    pre = [(w.semantic * sem[i] + w.rules * rules[i][0]) / base_w for i in range(len(candidates))]
    order = sorted(range(len(candidates)), key=lambda i: pre[i], reverse=True)
    judged: dict[str, LLMJudgement] = {}
    if llm is not None and llm_top_k > 0:
        top = [candidates[i] for i in order[:llm_top_k]]
        judged = llm_judge(persona, top, llm, batch_size=batch_size, progress=progress)

    ranked: list[RankedCandidate] = []
    for i, c in enumerate(candidates):
        rule_score, signals = rules[i]
        j = judged.get(c.profile_id)
        sb = ScoreBreakdown(semantic=float(sem[i]), rules=rule_score)
        if j is not None:
            sb.llm_ecosystem, sb.llm_influence = j.ecosystem, j.influence
            final = w.semantic * sb.semantic + w.llm * (sb.llm or 0) + w.rules * sb.rules
            if j.ecosystem < 0.25:  # LLM says "not school ecosystem" -> cap, whatever the keywords say
                final = min(final, 0.30)
                signals = signals + ["LLM: outside school ecosystem"]
            archetype, why, evidence = j.archetype, j.why, j.evidence
        else:
            final = pre[i]
            if judged:  # LLM ran but this one was outside the top-k: mark as unverified
                final *= 0.8
            archetype, why, evidence = _guess_archetype(signals), "", []
        sb.final = float(np.clip(final, 0.0, 1.0))
        ranked.append(
            RankedCandidate(
                candidate=c,
                scores=sb,
                archetype=archetype,
                explanation=why or _rule_explanation(signals, c),
                evidence=evidence,
                signals=signals,
            )
        )
    ranked.sort(key=lambda r: (r.scores.final, -r.candidate.best_position), reverse=True)
    for n, r in enumerate(ranked, 1):
        r.rank = n
    return ranked
