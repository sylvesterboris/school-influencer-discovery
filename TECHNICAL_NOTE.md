# Technical note — School Influencer Discovery

## 1. Architecture

```
┌──────────────┐   ┌──────────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────┐
│ Reference    │──▶│ AI analysis      │──▶│ Discovery     │──▶│ Ranking          │──▶│ Results  │
│ URL + text   │   │ persona + query  │   │ Serper API    │   │ embeddings       │   │ UI, CSV, │
│ (+ public    │   │ plan (Groq LLM)  │   │ parse, dedupe │   │ + LLM rubric     │   │ JSON,    │
│  snippet)    │   │                  │   │               │   │ + rule features  │   │ SQLite   │
└──────────────┘   └──────────────────┘   └───────────────┘   └──────────────────┘   └──────────┘
```

- **Stack:** Python 3.11+, Streamlit UI, SQLite, Groq (Llama 3.3 70B, 8B fallback), fastembed (BAAI/bge-small-en-v1.5, ONNX on CPU), scikit-learn (TF-IDF fallback).
- **Modules:** `analyzer` (persona, queries), `search` (Serper + cache), `parser` (SERP → fields), `ranker` (scoring), `pipeline` (orchestration), `storage` (SQLite), `app.py` (UI).
- **Deployment:** Streamlit Community Cloud with keys in Streamlit secrets. The same code runs locally or from the CLI.

## 2. Data retrieval approach

**Source:** the Serper.dev Google Search API. Queries are restricted with `site:linkedin.com/in`, and the API returns the public URL, title and snippet for each indexed profile.

**Why this is lawful and compliant:** we read search-engine results through a paid API we're authorised to use. We never request linkedin.com, never authenticate, never automate a browser, never handle CAPTCHAs, and never call LinkedIn's private endpoints. Only data a person chose to make public, as already shown by Google, is used.

**Reference profile:** the user pastes the profile URL and the text they can see (or LinkedIn's own "Save to PDF"). The app can also pull the reference's public search snippet. The reference is excluded from its own results.

**Processing:**
1. Keep only `linkedin.com/in/<slug>` URLs; normalise country subdomains (`in.linkedin.com`) and tracking parameters.
2. Parse the title (`Name - Headline - Organisation | LinkedIn`) and snippet (`Experience:`, `Location:`, leading `City, State · …`).
3. Strip LinkedIn boilerplate (connections count, "View X's profile…").
4. De-duplicate by slug, merging the queries that found the same person and keeping the richer snippet.

**Traceability:** SQLite tables `runs → queries → search_hits → candidates → scores` (see `schema.sql`). Each candidate records the queries that found them, the best result position and a retrieval timestamp. Raw API responses are cached in `search_cache`, so a run can be audited and re-run without new API calls. Every run also exports to a self-contained JSON file.

## 3. AI/ML method

### 3.1 Persona extraction (LLM)
Groq Llama 3.3 70B in JSON mode reads the reference text and returns a structured persona: headline, summary, role, organisation, location, seniority, target archetypes, school-ecosystem signals, boards (CBSE/ICSE/IB…), geographies, organisation types and similarity keywords. The prompt tells the model to use only what's in the text. If no key is configured, a keyword heuristic produces a simpler persona.

### 3.2 Query planning (LLM + templates)
The LLM writes ~60% of the query budget as diverse `site:linkedin.com/in` queries that cover every archetype and vary the vocabulary (principal, head of school, director academics, trustee, school chain, Sahodaya, school consultant, academic advisor…), mostly anchored to the reference's geography. The rest comes from fixed templates, round-robined across the five archetypes so coverage never depends on the LLM alone. Queries are de-duplicated, and any containing the reference person's own name are dropped.

### 3.3 Similarity and ranking

Each candidate gets three scores in [0, 1]:

| Signal | How | Why it's there |
|---|---|---|
| **Semantic similarity** | Cosine between bge-small embeddings of the persona text and the candidate's headline + organisation + location + snippet, rescaled from the typical [0.55, 0.88] cosine band | Captures professional context beyond exact keywords ("academic advisor to K-12 schools" ≈ "school improvement consultant") |
| **LLM rubric** | The LLM judges candidates in batches of 8 on *ecosystem relevance* (0–10: do they work in or with K-12 schools?) and *influence* (0–10: seniority, ownership, reach across schools), assigns an archetype, writes a one-line reason, and must quote 1–3 evidence phrases | Brings in judgement that keywords miss, e.g. an "edtech director" who sells to corporates vs one who runs school partnerships |
| **Rule features** | Transparent regex signals: role strength (principal 0.95 … teacher 0.35), education context (school, CBSE, vidyalaya, K-12…), multi-school reach ("30+ schools", "group of schools"), same region, matching board; penalties for student/intern/fresher signals or a senior title with no education context | Keeps the ranking explainable and stable when the LLM is unavailable or wrong |

```
llm   = 0.6 × ecosystem + 0.4 × influence
final = 0.30 × semantic + 0.45 × llm + 0.25 × rules        (weights adjustable in the UI)
```

**Cost-aware two-stage ranking:** all candidates are first pre-ranked on semantic + rules (free). Only the top-k (default 60) go to the LLM. Candidates outside that top-k keep their pre-score with a 0.8 factor, so an unreviewed profile can't outrank a reviewed one on keywords alone.

**Guard rails:**
- **Grounded evidence.** An evidence phrase is kept only if it appears verbatim (case and whitespace normalised) in the candidate's source text. Invented quotes are dropped.
- **Ecosystem gate.** If the LLM rates ecosystem relevance below 2.5/10, the final score is capped at 30, however many keywords match.
- **Fallbacks.** No LLM key → heuristic persona + template queries + semantic/rules ranking. No embedding model → TF-IDF cosine. Both are recorded in the run stats.

## 4. Evaluating relevance

The Evaluation tab lets a person tick which of the top 10–50 results they'd actually contact and reports precision@10 and precision@20 from those labels, so relevance is measured by a human rather than by the same model that produced the ranking. Labels are stored in the `labels` table and can be downloaded as CSV.

For the demo run the numbers are: `<fill in after labelling: precision@10 = __, precision@20 = __, profiles found = __, scoring ≥ 60 = __>`.

## 5. Limitations

- **Snippet-level data.** We only see what Google shows: a headline and ~150 characters. Roles can be stale, locations missing, and long careers are invisible.
- **Recall depends on the index.** Profiles that are private, not indexed, or poorly titled won't be found. Google can also return the same few prominent people across queries.
- **Parsing is heuristic.** Unusual titles ("Name | Educator | Speaker") may put the headline in the organisation field. The raw title and snippet are always kept alongside the parsed fields.
- **LLM judgement on thin text.** It can over- or under-rate people with vague headlines. Grounded evidence, the ecosystem gate and the non-LLM signals limit the damage but don't remove it.
- **"Connections" aren't observable.** We can't see who knows whom, so the "relationships with school decision-makers" signal is inferred from role, reach and context rather than measured from the network.
- **Free-tier limits.** Groq rate limits make a run take 1–2 minutes. Serper's free credits allow roughly 100 runs.
- **Streamlit Cloud storage is ephemeral.** The SQLite file resets on restart; JSON exports and committed runs are the durable record.
- **Name collisions** across different people with similar slugs aren't merged; de-duplication is by exact slug.

## 6. Compliant path to production

1. **Licensed data:** LinkedIn's official partner programmes (Sales Navigator Application Platform / Talent APIs) or a licensed B2B data provider, under a contract that covers outreach use. This also gives fuller profiles (experience history, shared connections), which would let the "relationship" signal be measured instead of inferred.
2. **Keep the ranking layer:** the persona → queries → hybrid scoring design is source-agnostic, since `SearchClient` is a small interface.
3. **Privacy:** store only what outreach needs, honour opt-out and deletion requests, set retention limits, and give the team a documented lawful basis for processing (India's DPDP Act 2023).
4. **Learning to rank:** use the Evaluation-tab labels (and later, outreach outcomes like reply rates) to tune the weights, then train a small learning-to-rank model on these features.
5. **Ops:** move from SQLite to Postgres, run discovery as a background job queue, and cache embeddings.

## 7. Setup

See [README.md](README.md). In short: `pip install -r requirements.txt`, put `SERPER_API_KEY` and `GROQ_API_KEY` in `.env` (or Streamlit secrets), then `streamlit run app.py`. Tests: `pytest`.
