"""End-to-end test with fake search + LLM clients (no network, synthetic data only)."""
import json
import re

from discovery.config import Settings
from discovery.models import Candidate, Persona, ReferenceProfile, SearchHit
from discovery.pipeline import RunResult, build_candidates, run_discovery
from discovery.ranker import Embedder, grounded_evidence, rule_features
from discovery.storage import Storage

SYNTHETIC = [
    ("test-leader-1", "Test Leader One - Former Principal - Example Public School | LinkedIn",
     "Location: Gurugram · Former principal of a CBSE school, now advises 30+ schools on academics."),
    ("test-consult-2", "Test Consultant Two - School Consultant - Self-employed | LinkedIn",
     "Location: New Delhi · Education consultant helping school owners with CBSE affiliation and school setup."),
    ("test-founder-3", "Test Founder Three - Founder - Example Group of Schools | LinkedIn",
     "Location: Noida · Founder of a chain of schools across NCR. Trustee."),
    ("test-sales-4", "Test Sales Four - Sales Manager - Example Software | LinkedIn",
     "Location: Bengaluru · B2B SaaS sales for retail companies."),
    ("test-student-5", "Test Student Five - Student - Example College | LinkedIn",
     "Location: Delhi · Class 12 student looking for opportunities."),
]


class FakeSearch:
    calls_made = 0

    def search(self, query, num=10, page=1):
        self.calls_made += 1
        return [
            SearchHit(query=query, position=i + 1, url=f"https://in.linkedin.com/in/{slug}", title=t,
                      snippet=s, retrieved_at="2026-11-12T00:00:00+00:00")
            for i, (slug, t, s) in enumerate(SYNTHETIC)
        ] + [SearchHit(query=query, position=9, url="https://www.linkedin.com/company/x", title="X", snippet="",
                       retrieved_at="")]


class FakeLLM:
    calls_made = 0
    tokens_used = 0

    def chat_json(self, system, user, max_tokens=1500, temperature=0.2):
        self.calls_made += 1
        if "Return JSON with these keys" in user:
            return {"name": "Ref", "headline": "Former Principal, school consultant", "summary": "Advises CBSE schools.",
                    "archetypes": ["school_leader", "consultant"], "ecosystem_signals": ["ran a CBSE school"],
                    "boards": ["CBSE"], "geographies": ["Gurugram", "Delhi"], "keywords": ["principal"]}
        if '"queries"' in user:
            return {"queries": [{"query": 'site:linkedin.com/in "former principal" "Gurugram"', "archetype": "school_leader"},
                                {"query": '"school consultant" Delhi', "archetype": "consultant"}]}
        ids = re.findall(r"id=(\S+)", user)
        out = []
        for pid in ids:
            good = pid.startswith(("test-leader", "test-consult", "test-founder"))
            out.append({"id": pid, "archetype": "school_leader" if good else "other",
                        "ecosystem_relevance": 9 if good else 1, "influence": 8 if good else 1,
                        "evidence": ["chain of schools", "made-up phrase not in text"], "why": "test"})
        return {"results": out}


def _settings():
    s = Settings(serper_api_key="x", groq_api_key="y", db_path=":memory:")
    s.max_queries = 4
    return s


def test_build_candidates_dedupes_and_excludes_reference():
    hits = FakeSearch().search("q1") + FakeSearch().search("q2")
    cands = build_candidates(hits, exclude_slugs={"test-sales-4"})
    slugs = {c.profile_id for c in cands}
    assert "test-sales-4" not in slugs and len(cands) == 4
    assert all(len(c.source_queries) == 2 for c in cands)


def test_grounded_evidence_drops_hallucinations():
    src = "Founder of a chain of schools across NCR."
    assert grounded_evidence(["chain of schools", "runs 500 schools"], src) == ["chain of schools"]


def test_rule_features_prefers_school_leaders():
    p = Persona(geographies=["Gurugram"], boards=["CBSE"])
    leader = Candidate("a", "u", "A", headline="Former Principal", snippet="CBSE school, Gurugram, advises 30+ schools")
    sales = Candidate("b", "u", "B", headline="Sales Manager", snippet="SaaS sales for retail companies")
    assert rule_features(leader, p)[0] > 0.7 > rule_features(sales, p)[0]


def test_end_to_end_ranking_and_roundtrip(tmp_path):
    storage = Storage(":memory:")
    result = run_discovery(
        ReferenceProfile(url="https://www.linkedin.com/in/reference-person", text="Former principal, CBSE consultant"),
        _settings(), FakeSearch(), FakeLLM(), Embedder(allow_neural=False), storage,
    )
    top3 = {r.candidate.profile_id for r in result.ranked[:3]}
    assert top3 == {"test-leader-1", "test-consult-2", "test-founder-3"}
    assert result.ranked[-1].scores.final <= 0.30  # LLM said "not school ecosystem" -> capped
    founder = next(r for r in result.ranked if r.candidate.profile_id == "test-founder-3")
    assert founder.evidence == ["chain of schools"]  # invented phrase removed
    assert all(q.query.startswith("site:linkedin.com/in") for q in result.queries)

    path = result.save_json(tmp_path)
    again = RunResult.load_json(path)
    assert [r.candidate.profile_id for r in again.ranked] == [r.candidate.profile_id for r in result.ranked]
    assert json.loads(path.read_text())["stats"]["unique_profiles"] == 5


def test_runs_without_llm():
    result = run_discovery(
        ReferenceProfile(url="https://www.linkedin.com/in/reference-person",
                         text="Former Principal, CBSE school, Gurugram. School consultant."),
        _settings(), FakeSearch(), None, Embedder(allow_neural=False), None,
    )
    assert result.persona.source == "heuristic"
    assert result.ranked[0].candidate.profile_id != "test-student-5"
