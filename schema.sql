-- SQLite schema for the School Influencer Discovery Tool.
-- Every candidate can be traced back to the exact search query, result position
-- and retrieval timestamp that produced it.

CREATE TABLE IF NOT EXISTS runs (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    reference_url   TEXT,
    reference_text  TEXT,
    persona_json    TEXT,
    settings_json   TEXT,
    stats_json      TEXT,
    status          TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS queries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    query           TEXT NOT NULL,
    archetype       TEXT,
    source          TEXT,          -- 'llm' or 'template'
    results_count   INTEGER DEFAULT 0
);

-- Raw search API responses, cached so re-runs cost no quota and stay reproducible.
CREATE TABLE IF NOT EXISTS search_cache (
    query_key       TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    response_json   TEXT NOT NULL,
    retrieved_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_hits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL REFERENCES runs(id),
    query           TEXT NOT NULL,
    position        INTEGER,
    url             TEXT NOT NULL,
    title           TEXT,
    snippet         TEXT,
    retrieved_at    TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
    run_id          TEXT NOT NULL REFERENCES runs(id),
    profile_id      TEXT NOT NULL,  -- LinkedIn slug
    url             TEXT NOT NULL,
    name            TEXT,
    headline        TEXT,
    role            TEXT,
    organisation    TEXT,
    location        TEXT,
    snippet         TEXT,
    source_queries  TEXT,           -- JSON array
    retrieved_at    TEXT,
    PRIMARY KEY (run_id, profile_id)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id          TEXT NOT NULL REFERENCES runs(id),
    profile_id      TEXT NOT NULL,
    rank            INTEGER,
    final           REAL,
    semantic        REAL,
    llm_ecosystem   REAL,
    llm_influence   REAL,
    rules           REAL,
    archetype       TEXT,
    explanation     TEXT,
    evidence        TEXT,           -- JSON array of phrases found verbatim in the source text
    signals         TEXT,           -- JSON array
    PRIMARY KEY (run_id, profile_id)
);

-- Human relevance labels, used to measure precision@k in the Evaluation tab.
CREATE TABLE IF NOT EXISTS labels (
    run_id          TEXT NOT NULL,
    profile_id      TEXT NOT NULL,
    relevant        INTEGER NOT NULL,
    labelled_at     TEXT NOT NULL,
    PRIMARY KEY (run_id, profile_id)
);

CREATE INDEX IF NOT EXISTS idx_hits_run ON search_hits(run_id);
CREATE INDEX IF NOT EXISTS idx_scores_run ON scores(run_id, rank);
