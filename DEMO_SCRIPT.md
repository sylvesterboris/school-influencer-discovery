# Demo video outline (7–8 minutes)

Record with the live Streamlit URL. Do one real run first so the embedding model is warm and the search cache is filled; then re-run during the recording so it finishes quickly.

**0:00 – 0:40 · The problem**
Edxso wants people who can open doors to school networks: consultants, former principals, trustees, school-chain founders. Finding them by hand on LinkedIn is slow. This tool takes one strong profile and finds and ranks similar people.

**0:40 – 1:30 · Data approach (say this clearly, reviewers care)**
Show the sidebar note. Data comes from a Google Search API (Serper) using `site:linkedin.com/in` queries. The app never opens LinkedIn, never logs in and never touches a CAPTCHA. Every result keeps the query and timestamp that produced it.

**1:30 – 2:30 · Reference input**
Paste the reference URL → *Fetch public headline* → add the About/Experience text. Click *Discover similar people* and talk over the progress bar: persona → queries → search → AI review.

**2:30 – 3:30 · AI analysis tab**
Walk through the persona: archetypes, ecosystem signals, boards, geography. Then the query table: point out the LLM-written queries vs templates, the coverage of every archetype, and results per query.

**3:30 – 5:30 · Results tab**
Open the top 3 cards. For each one read the score breakdown (semantic / LLM rubric / rules), the archetype, the "Why" line and the green evidence chips, then open *Source* to show the exact snippet and query. Show one lower-ranked profile and explain why it scored lower (e.g. teacher rather than leader, or not school-focused). Use the archetype filter to show only consultants.

**5:30 – 6:30 · Ranking method**
Short explanation: three signals, the formula, two-stage ranking to save LLM calls, evidence must appear in the source text, and the ecosystem gate. Change a weight slider and re-run on cache to show it's tunable.

**6:30 – 7:15 · Evaluation + traceability**
Evaluation tab: label the top 20 and show precision@10 / @20. Traceability tab: the raw hits table.

**7:15 – 8:00 · Export and limitations**
Download the CSV. Close with the main limitation (snippet-level data) and the production path (official LinkedIn partner APIs or licensed data, with this ranking layer on top).
