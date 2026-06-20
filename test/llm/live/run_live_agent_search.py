#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for an LLM Agent using the SearchTool.

This script is intentionally **not** collected by pytest. It demonstrates the
difference in an LLM's response when answering a time-sensitive question
with and without access to the internet via `SearchTool`.

Usage:
    python -m test.llm.live.run_live_agent_search

@File   :   run_live_agent_search.py
@Created:   2026/06/21
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import sys
from os import path
from typing import Any, Dict, Optional

from isobase.llm import OpenAIChat, AnthropicMessages
from isobase.llm.tools.search import SearchTool, BraveSearchProvider, TavilySearchProvider

ENV_PATH = path.join(path.dirname(__file__), ".env")


def _load_env() -> Dict[str, str]:
    """Parses the git-ignored .env file into a dict."""
    if not path.exists(ENV_PATH):
        sys.exit(f"Missing {ENV_PATH}. Please configure it first.")

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
    api_key = env.get(env_key, "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None
    return api_key


def main():
    env = _load_env()

    # 1. Select an available LLM Provider
    llm_client = None
    llm_name = ""
    if _get_api_key(env, "OPENAI_CHAT_API_KEY"):
        llm_client = OpenAIChat(
            api_key=_get_api_key(env, "OPENAI_CHAT_API_KEY"),
            base_url=env.get("OPENAI_CHAT_BASE_URL", "").strip() or None,
            default_model=env.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo").strip()
        )
        llm_name = "OpenAIChat"
    elif _get_api_key(env, "ANTHROPIC_MESSAGES_API_KEY"):
        llm_client = AnthropicMessages(
            api_key=_get_api_key(env, "ANTHROPIC_MESSAGES_API_KEY"),
            base_url=env.get("ANTHROPIC_MESSAGES_BASE_URL", "").strip() or None,
            default_model=env.get("ANTHROPIC_MESSAGES_MODEL", "claude-3-haiku-20240307").strip()
        )
        llm_name = "AnthropicMessages"
    else:
        sys.exit("[skip] No valid OpenAI or Anthropic API key found in .env.")

    # 2. Select an available Search Provider
    search_provider = None
    search_name = ""
    if _get_api_key(env, "TAVILY_SEARCH_API_KEY"):
        search_provider = TavilySearchProvider(api_key=_get_api_key(env, "TAVILY_SEARCH_API_KEY"))
        search_name = "Tavily"
    elif _get_api_key(env, "BRAVE_SEARCH_API_KEY"):
        search_provider = BraveSearchProvider(api_key=_get_api_key(env, "BRAVE_SEARCH_API_KEY"))
        search_name = "Brave"
    else:
        sys.exit("[skip] No valid Tavily or Brave API key found in .env.")

    print(f"Using LLM: {llm_name}")
    print(f"Using Search Provider: {search_name}")

    query = "What is the current weather in Shanghai today, and what are the major news headlines there right now?"

    print(f"\n{'=' * 60}\nScenario 1: LLM Without Web Search\n{'=' * 60}")
    print(f"Prompt: {query!r}\n")
    print("Generating response (this may contain outdated info or a refusal)...")

    resp_without_search = llm_client.ask(query, stream=False)
    print(f"\nResponse:\n{resp_without_search.content}\n")
    print("-" * 60)

    print(f"\n{'=' * 60}\nScenario 2: LLM With Web Search\n{'=' * 60}")
    print(f"Prompt: {query!r}\n")

    # Reset history to ensure a clean state
    llm_client.messages = []

    # Inject the SearchTool
    # We use force_external=True here to ensure we use our specific selected provider
    # rather than relying on the LLM's built-in blind search.
    search_tool = SearchTool(provider=search_provider, force_external=True)
    llm_client.tool_set.tools = [search_tool]

    print(f"Generating response (The LLM should call the {search_name} search tool automatically)...")
    resp_with_search = llm_client.ask(query, stream=False)

    print(f"\nResponse:\n{resp_with_search.content}\n")

    if llm_client.latest_tool_call_result:
        print(f"\n[Tool Execution Log] The model utilized the following tools: {list(llm_client.latest_tool_call_result.keys())}")
    else:
        print("\n[Tool Execution Log] The model did NOT utilize the search tool.")

    print("=" * 60)


if __name__ == "__main__":
    main()
