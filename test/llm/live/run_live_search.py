#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for the Search Providers against real APIs.

This script is intentionally **not** collected by pytest (filename does not
start with ``test_``) and never runs in CI. It reads credentials from a
git-ignored ``.env`` sitting next to this file and exercises both
``BraveSearchProvider`` and ``TavilySearchProvider`` end to end.

Usage:
    1. Copy the template and fill in real values (the real .env is git-ignored):
         cp test/llm/live/.env.example test/llm/live/.env
       Then edit test/llm/live/.env.
    2. Run from the repo root (so the ``isobase`` package is importable):
         python -m test.llm.live.run_live_search          # both providers
         python -m test.llm.live.run_live_search brave    # only the brave section
         python -m test.llm.live.run_live_search tavily   # only the tavily section

Leave a provider's API_KEY blank/absent in .env to skip it.

Expected .env variables (see .env.example):
    BRAVE_SEARCH_API_KEY
    TAVILY_SEARCH_API_KEY

@File   :   run_live_search.py
@Created:   2026/06/21 01:04
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import sys
from os import path
from typing import Any, Dict, Optional

from isobase.llm.tools.search import BaseSearchProvider, BraveSearchProvider, TavilySearchProvider

ENV_PATH = path.join(path.dirname(__file__), ".env")


def _load_env() -> Dict[str, str]:
    """Parses the git-ignored .env file into a dict, or exits with guidance.

    Supports ``KEY=value`` lines with optional ``export`` prefix, ``#``
    comments, blank lines, and surrounding single/double quotes on values.

    Returns:
        A mapping of environment variable names to their string values.
    """
    if not path.exists(ENV_PATH):
        sys.exit(
            f"Missing {ENV_PATH}.\n"
            "Copy the template and fill in real values:\n"
            "  cp test/llm/live/.env.example test/llm/live/.env")

    env: Dict[str, str] = {}
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                env[key] = value
    return env


def _get_api_key(env: Dict[str, str], env_key: str) -> Optional[str]:
    """Retrieves the API key and returns None if absent or contains template placeholders.

    Args:
        env: The parsed .env mapping.
        env_key: The variable name to fetch.

    Returns:
        The valid API key string, or None if the provider should be skipped.
    """
    api_key = env.get(env_key, "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None
    return api_key


def _run_provider_search(label: str, provider: BaseSearchProvider) -> None:
    """Performs standard query verification against a search provider."""
    query = "What is the latest version of Python?"
    print(f"\n{'=' * 60}\n[{label}] web search query\n{'=' * 60}")
    print(f"Query: {query!r}\n")

    # Call the respective search method. Different providers might have different kwargs.
    if label == "BraveSearchProvider":
        results = provider.search(query, count=3)
    else:
        results = provider.search(query, max_results=3, include_raw_content=True)

    if not results:
        print("Status: Success (No results returned)")
        return

    # Check for library-level error outputs captured in the list
    if len(results) == 1 and "error" in results[0]:
        print(f"Status: Failed\nDetails: {results[0]['error']}")
        return

    print(f"Status: Success ({len(results)} results retrieved)\n")
    for idx, item in enumerate(results, 1):
        print(f"Result #{idx}:")
        print(f"  Title:   {item.get('title')}")
        print(f"  URL:     {item.get('url')}")
        snippet = item.get("snippet", "")
        # Standardize representation to handle snippet vs content keys
        print(f"  Snippet: {snippet[:150].strip()}..." if len(snippet) > 150 else f"  Snippet: {snippet}")
        if "raw_content" in item:
            raw = item["raw_content"] or ""
            print(f"  Raw:     {raw[:150].strip()}..." if len(raw) > 150 else f"  Raw:     {raw}")
        print()


def run_brave(api_key: str) -> None:
    """Runs the search test against Brave Search Provider."""
    provider = BraveSearchProvider(api_key=api_key)
    _run_provider_search("BraveSearchProvider", provider)


def run_tavily(api_key: str) -> None:
    """Runs the search test against Tavily Search Provider."""
    provider = TavilySearchProvider(api_key=api_key)
    _run_provider_search("TavilySearchProvider", provider)


def main(only: Optional[str] = None) -> None:
    """Entry point: runs the requested search provider section(s).

    Args:
        only: ``"brave"`` or ``"tavily"`` to run a single provider, or None
            for every search provider whose API key is present in the .env file.
    """
    env = _load_env()
    runners = {
        "brave": ("BRAVE_SEARCH_API_KEY", run_brave),
        "tavily": ("TAVILY_SEARCH_API_KEY", run_tavily),
    }

    selected = [only] if only else list(runners)
    for name in selected:
        if name not in runners:
            sys.exit(f"Unknown search provider '{name}'. Choose from: {list(runners)}")
        env_key, runner = runners[name]
        api_key = _get_api_key(env, env_key)
        if api_key is None:
            print(f"\n[skip] no usable {env_key} in .env for '{name}'")
            continue
        try:
            runner(api_key)
        except Exception as e:
            print(f"\n[{name}] Unexpected runner crash: {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
