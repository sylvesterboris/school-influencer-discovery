"""Run the full pipeline from the terminal and save the run as JSON.

    python scripts/run_cli.py --url https://www.linkedin.com/in/<slug> --text-file reference.txt
    python scripts/run_cli.py --url https://www.linkedin.com/in/<slug> --fetch   # use the public search snippet

The JSON lands in data/runs/ and can be opened in the app under "Open a saved run".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from discovery.config import get_settings  # noqa: E402
from discovery.llm import GroqClient  # noqa: E402
from discovery.models import ReferenceProfile  # noqa: E402
from discovery.pipeline import lookup_reference, run_discovery  # noqa: E402
from discovery.ranker import Embedder  # noqa: E402
from discovery.search import SerperClient  # noqa: E402
from discovery.storage import Storage  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="Reference LinkedIn profile URL")
    ap.add_argument("--text", default="", help="Reference profile text")
    ap.add_argument("--text-file", help="File containing the reference profile text")
    ap.add_argument("--fetch", action="store_true", help="Prepend the public search snippet for --url")
    ap.add_argument("--max-queries", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=60, help="Profiles reviewed by the LLM")
    ap.add_argument("--show", type=int, default=25)
    args = ap.parse_args()

    s = get_settings()
    if not s.has_search:
        sys.exit("SERPER_API_KEY is missing (set it in .env)")
    s.max_queries, s.llm_rerank_top_k = args.max_queries, args.top_k
    storage = Storage(s.db_path)
    search = SerperClient(s.serper_api_key, storage, s.country)
    llm = GroqClient(s.groq_api_key, s.groq_model, s.groq_fallback_model) if s.has_llm else None

    text = args.text
    if args.text_file:
        text = Path(args.text_file).read_text()
    if args.fetch:
        text = (lookup_reference(args.url, search) + "\n" + text).strip()
    if not text:
        sys.exit("Give the reference profile text with --text / --text-file, or use --fetch")

    result = run_discovery(
        ReferenceProfile(url=args.url, text=text), s, search, llm, Embedder(s.embedding_model), storage,
        progress=lambda msg, frac: print(f"[{frac:4.0%}] {msg}"),
    )
    path = result.save_json()
    print(f"\nPersona: {result.persona.headline}\nStats: {result.stats}\n")
    for r in result.ranked[: args.show]:
        c = r.candidate
        print(f"{r.rank:>3}. {r.scores.final * 100:5.1f}  {c.name} — {c.role or c.headline} @ {c.organisation} "
              f"[{c.location}]\n       {c.url}\n       {r.explanation}")
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
