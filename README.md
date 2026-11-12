# School Influencer Discovery

Give it one LinkedIn profile of a well-connected school-education person. It works out what makes that person relevant (role, seniority, school boards, region, the kind of schools they work with), searches for public profiles of similar people, and returns a ranked list with a relevance score and a short reason for every match.

Built for the Edxso AI Intern assignment.

**Live app:** `<add your Streamlit Cloud URL>`  
**Demo video:** `<add link>`  
**Technical note:** [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md)

```
Reference profile → AI analysis → Discovery → Ranking → Results
   URL + text        Groq LLM      Serper        embeddings + LLM rubric + rules
```

## What it finds

School and education consultants, current and former principals, school directors, owners and trustees, education entrepreneurs and school-chain people, professionals who work across many schools, and people connected to school decision-makers (associations, Sahodaya clusters, school partnerships).

Each result shows name and LinkedIn URL, current role, organisation and location, the school-ecosystem signals that fired, a 0–100 relevance score with its breakdown, a one-line explanation, evidence phrases copied from the source, and the search query that found the person.

## How data is collected (and what it doesn't do)

Profiles come from the **Serper.dev Google Search API**, using queries like:

```
site:linkedin.com/in "former principal" "CBSE" "Gurugram"
```

Search engines index public LinkedIn profiles, and the API returns the URL, title and snippet Google already shows. The tool **never requests linkedin.com**, never logs in, never uses cookies or a browser, and never touches a CAPTCHA. Every profile is stored with the query, result position and retrieval time that produced it, and raw API responses are cached in SQLite so a run can be audited and reproduced.

## Quick start (local)

```bash
git clone https://github.com/sylvesterboris/school-influencer-discovery.git
cd school-influencer-discovery
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add SERPER_API_KEY and GROQ_API_KEY
streamlit run app.py
```

Keys:
- **Serper**: sign up at serper.dev, 2,500 free queries. One discovery run uses about 24.
- **Groq**: console.groq.com, free tier. Without it the app still works using a keyword-based persona and rule + embedding ranking, but the results are noticeably weaker.

The first run downloads the embedding model (BAAI/bge-small-en-v1.5, ~70 MB). If it can't download, ranking falls back to TF-IDF and the app says so in the run caption.

### Command line

```bash
python scripts/run_cli.py --url https://www.linkedin.com/in/<slug> --text-file reference.txt
python scripts/run_cli.py --url https://www.linkedin.com/in/<slug> --fetch   # use the public snippet only
```

Runs are saved to `data/runs/<run-id>.json` and can be opened in the app under **Open a saved run**.

### Tests

```bash
pytest
```

The tests use fake search and LLM clients with obviously synthetic data, so they run offline.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub.
2. At share.streamlit.io create an app pointing at `app.py`.
3. In **Settings → Secrets** paste the contents of `.streamlit/secrets.toml.example` with real keys.
4. Commit at least one real run in `data/runs/` so reviewers can open results without spending API quota.

Streamlit Cloud's disk is temporary, so the SQLite database resets when the app restarts. Use the JSON export or the committed runs for anything you want to keep.

## Project layout

```
app.py                     Streamlit UI
discovery/
  analyzer.py              persona extraction + query planning (LLM, with template fallback)
  search.py                Serper client with SQLite response cache
  parser.py                SERP title/snippet → name, role, organisation, location
  ranker.py                embeddings + LLM rubric + rule features → final score
  pipeline.py              orchestration, de-duplication, JSON export/import
  storage.py               SQLite persistence
  llm.py                   Groq client (JSON mode, retry, fallback model)
  config.py, models.py
schema.sql                 database schema
scripts/run_cli.py         run from the terminal
tests/                     offline tests
data/runs/                 saved runs (demo snapshot)
```

## Using the app

1. Paste the reference profile URL. Click **Fetch public headline** to pull the public snippet, then add anything else you can see on the profile (About, Experience). More text gives a better persona.
2. Click **Discover similar people**. A run takes roughly 1–2 minutes, most of it the LLM review on Groq's free tier.
3. **Results**: filter by archetype or minimum score.
4. **AI analysis**: the persona the model built and every query it ran.
5. **Traceability**: the full table plus every raw search hit.
6. **Evaluation**: tick the people you'd really contact and get precision@10/20.
7. **Export**: CSV for outreach, JSON to reopen the run later.

## Limitations

Short version (details in the technical note): the tool only sees what Google shows in a search snippet, so roles can be out of date and locations are sometimes missing; recall depends on what Google indexes; the LLM can misjudge thin snippets, which is why evidence must match the source text and the score blends in non-LLM signals. For production, the compliant route is LinkedIn's official partner APIs (Sales Navigator / SNAP) or a licensed data provider, with this ranking layer on top.
