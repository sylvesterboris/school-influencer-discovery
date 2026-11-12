"""SQLite persistence: runs, queries, raw hits, candidates, scores, labels and a search cache."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import ROOT
from .models import Candidate, DiscoveryQuery, Persona, RankedCandidate, SearchHit

SCHEMA_PATH = ROOT / "schema.sql"


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_PATH.read_text())
        self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ---- search cache -------------------------------------------------
    def cache_get(self, key: str) -> tuple[dict[str, Any], str] | None:
        row = self._conn.execute(
            "SELECT response_json, retrieved_at FROM search_cache WHERE query_key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["response_json"]), row["retrieved_at"]

    def cache_put(self, key: str, query: str, response: dict[str, Any]) -> str:
        ts = utcnow()
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO search_cache(query_key, query, response_json, retrieved_at) VALUES (?,?,?,?)",
                (key, query, json.dumps(response), ts),
            )
        return ts

    # ---- runs ---------------------------------------------------------
    def create_run(self, run_id: str, reference_url: str, reference_text: str, settings: dict[str, Any]) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT INTO runs(id, created_at, reference_url, reference_text, settings_json, status) VALUES (?,?,?,?,?,?)",
                (run_id, utcnow(), reference_url, reference_text, json.dumps(settings, default=str), "running"),
            )

    def finish_run(self, run_id: str, persona: Persona, stats: dict[str, Any], status: str = "done") -> None:
        with self._tx() as c:
            c.execute(
                "UPDATE runs SET persona_json = ?, stats_json = ?, status = ? WHERE id = ?",
                (json.dumps(persona.to_dict()), json.dumps(stats, default=str), status, run_id),
            )

    def save_queries(self, run_id: str, queries: list[DiscoveryQuery]) -> None:
        with self._tx() as c:
            c.executemany(
                "INSERT INTO queries(run_id, query, archetype, source, results_count) VALUES (?,?,?,?,?)",
                [(run_id, q.query, q.archetype, q.source, q.results_count) for q in queries],
            )

    def save_hits(self, run_id: str, hits: list[SearchHit]) -> None:
        with self._tx() as c:
            c.executemany(
                "INSERT INTO search_hits(run_id, query, position, url, title, snippet, retrieved_at) VALUES (?,?,?,?,?,?,?)",
                [(run_id, h.query, h.position, h.url, h.title, h.snippet, h.retrieved_at) for h in hits],
            )

    def save_candidates(self, run_id: str, candidates: list[Candidate]) -> None:
        with self._tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO candidates(run_id, profile_id, url, name, headline, role, organisation, location,"
                " snippet, source_queries, retrieved_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (run_id, x.profile_id, x.url, x.name, x.headline, x.role, x.organisation, x.location,
                     x.snippet, json.dumps(x.source_queries), x.retrieved_at)
                    for x in candidates
                ],
            )

    def save_scores(self, run_id: str, ranked: list[RankedCandidate]) -> None:
        with self._tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO scores(run_id, profile_id, rank, final, semantic, llm_ecosystem, llm_influence,"
                " rules, archetype, explanation, evidence, signals) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (run_id, r.candidate.profile_id, r.rank, r.scores.final, r.scores.semantic,
                     r.scores.llm_ecosystem, r.scores.llm_influence, r.scores.rules, r.archetype,
                     r.explanation, json.dumps(r.evidence), json.dumps(r.signals))
                    for r in ranked
                ],
            )

    # ---- labels -------------------------------------------------------
    def save_labels(self, run_id: str, labels: dict[str, bool]) -> None:
        ts = utcnow()
        with self._tx() as c:
            c.executemany(
                "INSERT OR REPLACE INTO labels(run_id, profile_id, relevant, labelled_at) VALUES (?,?,?,?)",
                [(run_id, pid, int(v), ts) for pid, v in labels.items()],
            )

    def get_labels(self, run_id: str) -> dict[str, bool]:
        rows = self._conn.execute("SELECT profile_id, relevant FROM labels WHERE run_id = ?", (run_id,)).fetchall()
        return {r["profile_id"]: bool(r["relevant"]) for r in rows}

    def list_runs(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, created_at, reference_url, status FROM runs ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
