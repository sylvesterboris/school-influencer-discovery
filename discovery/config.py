"""Settings loaded from environment variables, a local .env file, or Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_env_file(path: Path = ENV_FILE) -> None:
    """Load KEY=value lines from the project's .env, whatever folder the app was started from."""
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=True)
        return
    except ImportError:  # python-dotenv missing: tiny built-in parser
        pass
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ[key.strip()] = value


_load_env_file()
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"


def _secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)  # type: ignore[no-any-return]
    except Exception:
        return default


@dataclass
class Weights:
    """How the three ranking signals are blended into the final score."""

    semantic: float = 0.30
    llm: float = 0.45
    rules: float = 0.25

    def normalised(self) -> "Weights":
        total = self.semantic + self.llm + self.rules
        if total <= 0:
            return Weights()
        return Weights(self.semantic / total, self.llm / total, self.rules / total)


@dataclass
class Settings:
    serper_api_key: str | None = None
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    db_path: str = str(DATA_DIR / "discovery.db")
    country: str = "in"  # Google country code passed to Serper
    max_queries: int = 24
    results_per_query: int = 10
    llm_rerank_top_k: int = 60  # how many pre-ranked candidates the LLM judges
    llm_batch_size: int = 8
    min_final_score: float = 0.0
    weights: Weights = field(default_factory=Weights)

    @property
    def has_search(self) -> bool:
        return bool(self.serper_api_key)

    @property
    def has_llm(self) -> bool:
        return bool(self.groq_api_key)


def get_settings() -> Settings:
    return Settings(
        serper_api_key=_secret("SERPER_API_KEY"),
        groq_api_key=_secret("GROQ_API_KEY"),
        groq_model=_secret("GROQ_MODEL", "llama-3.3-70b-versatile") or "llama-3.3-70b-versatile",
        groq_fallback_model=_secret("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant",
        embedding_model=_secret("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5") or "BAAI/bge-small-en-v1.5",
        db_path=_secret("DB_PATH", str(DATA_DIR / "discovery.db")) or str(DATA_DIR / "discovery.db"),
        country=_secret("SEARCH_COUNTRY", "in") or "in",
    )
