"""Streamlit UI for the LinkedIn School Influencer Discovery Tool."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from discovery.config import RUNS_DIR, Settings, Weights, get_settings
from discovery.llm import GroqClient
from discovery.models import ARCHETYPES, RankedCandidate, ReferenceProfile
from discovery.pipeline import RunResult, lookup_reference, run_discovery
from discovery.ranker import Embedder
from discovery.search import SearchError, SerperClient
from discovery.storage import Storage

st.set_page_config(page_title="School Influencer Discovery", page_icon="🏫", layout="wide")

st.markdown(
    """
<style>
.flow {display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.2rem 0 1rem 0;font-size:.85rem}
.flow span.step {padding:.25rem .65rem;border-radius:999px;border:1px solid rgba(120,120,120,.35)}
.flow span.arrow {opacity:.5}
.score {font-size:1.9rem;font-weight:700;line-height:1}
.score-sub {font-size:.75rem;opacity:.7}
.chip {display:inline-block;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;border-radius:6px;
       background:rgba(46,125,255,.10);border:1px solid rgba(46,125,255,.25);font-size:.78rem}
.ev {display:inline-block;padding:.1rem .5rem;margin:.1rem .2rem .1rem 0;border-radius:6px;
     background:rgba(16,163,74,.10);border:1px solid rgba(16,163,74,.3);font-size:.78rem}
.muted {opacity:.7;font-size:.85rem}
</style>
""",
    unsafe_allow_html=True,
)

ARCH_LABEL = {
    "consultant": "Consultant / advisor",
    "school_leader": "School leader / owner / trustee",
    "entrepreneur": "Education entrepreneur / network",
    "multi_school": "Works across many schools",
    "connector": "Connected to decision-makers",
    "other": "Other",
}


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedder(model_name: str) -> Embedder:
    return Embedder(model_name)


@st.cache_resource
def get_storage(db_path: str) -> Storage:
    return Storage(db_path)


def saved_runs() -> list[Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


# --------------------------------------------------------------------------- sidebar
settings: Settings = get_settings()
storage = get_storage(settings.db_path)

with st.sidebar:
    st.header("Setup")
    st.markdown(
        f"{'✅' if settings.has_search else '❌'} Serper search API  \n"
        f"{'✅' if settings.has_llm else '⚠️'} Groq LLM {'' if settings.has_llm else '(heuristic fallback)'}"
    )
    mode = st.radio("Mode", ["Live discovery", "Open a saved run"],
                    index=0 if settings.has_search else 1,
                    help="Saved runs are real past runs stored as JSON — useful when API quota is limited.")

    with st.expander("Ranking settings", expanded=False):
        settings.max_queries = st.slider("Search queries", 8, 40, settings.max_queries,
                                         help="Each query costs one Serper credit (cached repeats are free).")
        settings.llm_rerank_top_k = st.slider("Profiles reviewed by the LLM", 0, 100, settings.llm_rerank_top_k,
                                              help="Top-k after the cheap pre-rank. 0 disables the LLM rubric.")
        st.caption("Score weights (normalised to sum to 1)")
        w_sem = st.slider("Semantic similarity", 0.0, 1.0, 0.30, 0.05)
        w_llm = st.slider("LLM relevance rubric", 0.0, 1.0, 0.45, 0.05)
        w_rules = st.slider("Rule-based signals", 0.0, 1.0, 0.25, 0.05)
        settings.weights = Weights(w_sem, w_llm, w_rules)

    st.divider()
    st.caption(
        "Data comes only from a Google Search API (Serper). The app never opens linkedin.com, "
        "never logs in and never bypasses access controls."
    )

# --------------------------------------------------------------------------- header
st.title("🏫 School Influencer Discovery")
st.markdown(
    '<div class="flow">' + '<span class="arrow">→</span>'.join(
        f'<span class="step">{s}</span>'
        for s in ["1 · Reference profile", "2 · AI analysis", "3 · Discovery", "4 · Ranking", "5 · Results"]
    ) + "</div>",
    unsafe_allow_html=True,
)

if "result" not in st.session_state:
    st.session_state.result = None
if "ref_text" not in st.session_state:
    st.session_state.ref_text = ""

# --------------------------------------------------------------------------- input
if mode == "Live discovery":
    with st.container(border=True):
        st.subheader("1 · Reference profile")
        c1, c2 = st.columns([3, 1])
        url = c1.text_input("LinkedIn profile URL", placeholder="https://www.linkedin.com/in/…")
        c2.write("")
        c2.write("")
        if c2.button("Fetch public headline", width="stretch", disabled=not (settings.has_search and url)):
            try:
                found = lookup_reference(url, SerperClient(settings.serper_api_key, storage, settings.country))
                if found:
                    st.session_state.ref_text = found
                    st.success("Pulled the public search snippet. Add more detail below if you have it.")
                else:
                    st.warning("No public search result for that URL. Paste the profile text below instead.")
            except (ValueError, SearchError) as e:
                st.error(str(e))
        st.session_state.ref_text = st.text_area(
            "Profile text (headline, About, Experience)",
            value=st.session_state.ref_text,
            height=180,
            help="Paste what you can see on the profile, or use LinkedIn's own 'Save to PDF' and copy the text. "
                 "More detail gives a better persona.",
        )
        run = st.button("Discover similar people", type="primary",
                        disabled=not (settings.has_search and url and st.session_state.ref_text.strip()))
        if not settings.has_search:
            st.info("Add SERPER_API_KEY to secrets to run live discovery, or open a saved run from the sidebar.")

    if run:
        llm = (GroqClient(settings.groq_api_key, settings.groq_model, settings.groq_fallback_model)
               if settings.has_llm else None)
        search = SerperClient(settings.serper_api_key, storage, settings.country)
        bar = st.progress(0.0, text="Starting…")
        try:
            result = run_discovery(
                ReferenceProfile(url=url.strip(), text=st.session_state.ref_text.strip()),
                settings, search, llm, get_embedder(settings.embedding_model), storage,
                progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg),
            )
            result.save_json()
            st.session_state.result = result
            bar.empty()
        except SearchError as e:
            bar.empty()
            st.error(f"Search failed: {e}")
else:
    with st.container(border=True):
        st.subheader("Open a saved run")
        runs = saved_runs()
        choice = st.selectbox(
            "Saved runs", runs, format_func=lambda p: p.stem, index=0 if runs else None,
            placeholder="No saved runs yet",
        )
        upload = st.file_uploader("…or upload a run JSON exported from this app", type="json")
        if st.button("Open", type="primary", disabled=not (choice or upload)):
            data = json.load(upload) if upload else json.loads(Path(choice).read_text())
            st.session_state.result = RunResult.from_dict(data)

result: RunResult | None = st.session_state.result
if result is None:
    st.stop()

# --------------------------------------------------------------------------- results
stats = result.stats
st.divider()
m = st.columns(5)
m[0].metric("Search queries", stats.get("queries", len(result.queries)))
m[1].metric("Unique profiles", stats.get("unique_profiles", len(result.ranked)))
m[2].metric("Reviewed by LLM", stats.get("llm_judged", 0))
m[3].metric("Score ≥ 60", stats.get("score_ge_60", 0))
m[4].metric("Run time", f"{stats.get('seconds', 0)} s")
st.caption(f"Run `{result.run_id}` · {result.created_at} · reference: {result.reference.url} · "
           f"embeddings: {stats.get('embedding_backend', '?')} · persona: {stats.get('persona_source', '?')}")

tab_res, tab_ai, tab_trace, tab_eval, tab_export = st.tabs(
    ["Results", "AI analysis", "Traceability", "Evaluation", "Export"]
)


def render_card(r: RankedCandidate) -> None:
    c, s = r.candidate, r.scores
    with st.container(border=True):
        left, mid, right = st.columns([1, 5, 2])
        left.markdown(f'<div class="score">{s.final * 100:.0f}</div><div class="score-sub">#{r.rank} · relevance</div>',
                      unsafe_allow_html=True)
        mid.markdown(f"**[{c.name}]({c.url})**  \n{c.role or c.headline or '—'}"
                     f"{' · ' + c.organisation if c.organisation else ''}"
                     f"{' · 📍 ' + c.location if c.location else ''}")
        mid.markdown(f'<span class="chip">{ARCH_LABEL.get(r.archetype, r.archetype)}</span>'
                     + "".join(f'<span class="chip">{sig}</span>' for sig in r.signals[:4]),
                     unsafe_allow_html=True)
        mid.markdown(f"**Why:** {r.explanation}")
        if r.evidence:
            mid.markdown("Evidence: " + "".join(f'<span class="ev">“{e}”</span>' for e in r.evidence),
                         unsafe_allow_html=True)
        right.progress(s.semantic, text=f"Semantic {s.semantic * 100:.0f}")
        if s.llm is not None:
            right.progress(min(s.llm, 1.0), text=f"LLM rubric {s.llm * 100:.0f}")
        else:
            right.caption("LLM rubric: not reviewed")
        right.progress(s.rules, text=f"Rules {s.rules * 100:.0f}")
        with mid.expander("Source"):
            st.markdown(f'<div class="muted">{c.snippet or "—"}</div>', unsafe_allow_html=True)
            st.caption("Found via: " + " · ".join(f"`{q}`" for q in c.source_queries[:3])
                       + f" · retrieved {c.retrieved_at or 'n/a'}")


with tab_res:
    f1, f2, f3 = st.columns([3, 2, 1])
    archs = f1.multiselect("Archetypes", list(ARCH_LABEL), default=[a for a in ARCH_LABEL if a != "other"],
                           format_func=ARCH_LABEL.get)
    min_score = f2.slider("Minimum relevance", 0, 100, 0)
    top_n = f3.number_input("Show", 10, 200, 25, step=5)
    shown = [r for r in result.ranked if r.archetype in archs and r.scores.final * 100 >= min_score][: int(top_n)]
    st.caption(f"Showing {len(shown)} of {len(result.ranked)} discovered profiles")
    for r in shown:
        render_card(r)

with tab_ai:
    p = result.persona
    a, b = st.columns(2)
    with a:
        st.subheader("Reference persona")
        st.markdown(f"**{p.headline or '—'}**")
        st.write(p.summary)
        st.markdown(f"**Role:** {p.current_role or '—'}  \n**Organisation:** {p.organisation or '—'}  \n"
                    f"**Location:** {p.location or '—'}  \n**Seniority:** {p.seniority or '—'}")
        st.markdown("**Archetypes:** " + ", ".join(ARCH_LABEL.get(x, x) for x in p.archetypes))
    with b:
        st.markdown("**Ecosystem signals**")
        st.markdown("".join(f'<span class="chip">{x}</span>' for x in p.ecosystem_signals) or "—",
                    unsafe_allow_html=True)
        st.markdown(f"**Boards:** {', '.join(p.boards) or '—'}  \n**Geographies:** {', '.join(p.geographies) or '—'}  \n"
                    f"**Works with:** {', '.join(p.org_types) or '—'}")
        st.markdown(f"**Similarity keywords:** {', '.join(p.keywords) or '—'}")
    st.subheader("Discovery queries")
    st.dataframe(pd.DataFrame([{"query": q.query, "archetype": ARCH_LABEL.get(q.archetype, q.archetype),
                                "source": q.source, "results": q.results_count} for q in result.queries]),
                 width="stretch", hide_index=True)
    st.subheader("Archetype mix of results (score ≥ 50)")
    mix = pd.Series([ARCH_LABEL.get(r.archetype, r.archetype) for r in result.ranked if r.scores.final >= 0.5])
    if not mix.empty:
        st.bar_chart(mix.value_counts())

with tab_trace:
    st.markdown("Every profile links back to the search query, result position and retrieval time that produced it. "
                "Raw responses are cached in SQLite (`search_cache`).")
    st.dataframe(pd.DataFrame([r.to_row() for r in result.ranked]), width="stretch", hide_index=True,
                 column_config={"linkedin_url": st.column_config.LinkColumn("linkedin_url")})
    with st.expander(f"Raw search hits ({len(result.hits)})"):
        st.dataframe(pd.DataFrame([{"query": h.query, "position": h.position, "url": h.url, "title": h.title,
                                    "snippet": h.snippet, "retrieved_at": h.retrieved_at, "cached": h.from_cache}
                                   for h in result.hits]), width="stretch", hide_index=True)
    if stats.get("errors"):
        st.warning("Some queries failed:\n\n" + "\n".join(f"- {e}" for e in stats["errors"]))

with tab_eval:
    st.markdown("Mark which of the top results you'd actually reach out to. Precision@k is computed from your labels, "
                "so relevance is judged by a person, not by the model that produced the ranking.")
    k_label = st.slider("Label the top", 10, 50, 25, step=5)
    saved = storage.get_labels(result.run_id)
    rows = [{"rank": r.rank, "name": r.candidate.name, "role": r.candidate.role or r.candidate.headline,
             "organisation": r.candidate.organisation, "score": round(r.scores.final * 100),
             "relevant": saved.get(r.candidate.profile_id, r.scores.final >= 0.6), "profile_id": r.candidate.profile_id}
            for r in result.ranked[:k_label]]
    edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch",
                            disabled=["rank", "name", "role", "organisation", "score", "profile_id"],
                            column_config={"relevant": st.column_config.CheckboxColumn("Relevant?")})
    rel = edited["relevant"].astype(bool).tolist()
    ks = sorted({min(k, len(rel)) for k in (10, 20, k_label) if rel})
    for col, k in zip(st.columns(max(len(ks), 1)), ks):
        col.metric(f"Precision@{k}", f"{sum(rel[:k]) / k:.0%}")
    if st.button("Save labels"):
        storage.save_labels(result.run_id, dict(zip(edited["profile_id"], edited["relevant"].astype(bool))))
        st.success("Labels saved.")
    st.download_button("Download labels CSV", edited.to_csv(index=False), f"{result.run_id}-labels.csv", "text/csv")

with tab_export:
    df = pd.DataFrame([r.to_row() for r in result.ranked])
    st.download_button("Download ranked results (CSV)", df.to_csv(index=False), f"{result.run_id}.csv", "text/csv",
                       type="primary")
    buf = io.StringIO()
    json.dump(result.to_dict(), buf, indent=2, ensure_ascii=False)
    st.download_button("Download full run (JSON, re-openable)", buf.getvalue(), f"{result.run_id}.json",
                       "application/json")
