"""Serper.dev client (Google Search API).

This is the only place the tool gets profile data from. It reads public search-engine
results; it never requests linkedin.com, never logs in and never touches CAPTCHAs.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Protocol

import requests

from .models import SearchHit
from .storage import Storage

SERPER_URL = "https://google.serper.dev/search"


class SearchClient(Protocol):
    def search(self, query: str, num: int = 10, page: int = 1) -> list[SearchHit]: ...


class SearchError(RuntimeError):
    pass


class SerperClient:
    def __init__(self, api_key: str, storage: Storage | None = None, country: str = "in", timeout: int = 20):
        if not api_key:
            raise SearchError("SERPER_API_KEY is not set")
        self.api_key = api_key
        self.storage = storage
        self.country = country
        self.timeout = timeout
        self.calls_made = 0  # paid API calls (cache hits excluded)

    def _key(self, payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                resp = requests.post(
                    SERPER_URL,
                    headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise SearchError(f"Serper HTTP {resp.status_code}")
                if resp.status_code in (401, 403):
                    raise SearchError("Serper rejected the API key (check SERPER_API_KEY / credits)")
                resp.raise_for_status()
                self.calls_made += 1
                return resp.json()
            except SearchError as e:
                last_err = e
                if "rejected" in str(e):
                    raise
            except requests.RequestException as e:
                last_err = e
            time.sleep(1.5 * (attempt + 1))
        raise SearchError(f"Serper request failed: {last_err}")

    def search(self, query: str, num: int = 10, page: int = 1) -> list[SearchHit]:
        payload = {"q": query, "num": num, "page": page, "gl": self.country, "hl": "en"}
        key = self._key(payload)
        cached = self.storage.cache_get(key) if self.storage else None
        if cached:
            data, retrieved_at = cached
            from_cache = True
        else:
            data = self._post(payload)
            retrieved_at = self.storage.cache_put(key, query, data) if self.storage else ""
            from_cache = False
        hits = []
        for item in data.get("organic", []) or []:
            hits.append(
                SearchHit(
                    query=query,
                    position=int(item.get("position", len(hits) + 1)),
                    url=item.get("link", ""),
                    title=item.get("title", ""),
                    snippet=item.get("snippet", ""),
                    retrieved_at=retrieved_at,
                    from_cache=from_cache,
                )
            )
        return hits
