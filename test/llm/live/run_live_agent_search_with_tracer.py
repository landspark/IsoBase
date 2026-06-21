#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for an LLM Agent using the SearchTool.

This script is intentionally **not** collected by pytest. It demonstrates the
difference in an LLM's response when answering a time-sensitive question
with and without access to the internet via `SearchTool`.

Usage:
    python -m test.llm.live.run_live_agent_search

@File   :   run_live_agent_search.py
@Created:   2026/06/21 03:14
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import sys
from os import path
from typing import Any, Dict, List, Optional

from isobase.core.logger import LOGGER
from isobase.llm import OpenAIChat, AnthropicMessages
from isobase.llm.entities import SearchResultItem
from isobase.llm.tools.search import SearchTool, BraveSearchProvider, TavilySearchProvider
from isobase.llm.entities import SearchResult
from isobase.llm.tools.search.base import BaseSearchProvider

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

class DummySearchProvider(BaseSearchProvider):
    def search(self, query: str, **kwargs: Any) -> SearchResult:
        LOGGER.info(f"\n[Trace - DummySearch] Executing query: {query!r}")
        return SearchResult(
            success=True,
            results=[
                SearchResultItem(
                    title="Tokyo Weather Mock",
                    url="https://mock.weather/tokyo",
                    snippet="Weather today: Sunny, 21°C."
                ),
                SearchResultItem(
                    title="Tokyo News Mock",
                    url="https://mock.news/tokyo",
                    snippet="News: IsoBase introduces a fast multi-turn search tool wrapper."
                )
            ]
        )

import json

def wrap_client_with_tracer(client_instance: Any):
    """Intercepts and logs the raw messages sent to and received from the API."""
    original_generate = client_instance.generate

    def traced_generate(messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        LOGGER.info("\n" + "="*40 + " RAW OUTBOUND API REQUEST " + "="*40)
        LOGGER.info(f"Model: {kwargs.get('model', client_instance.default_model)}")
        LOGGER.info("Messages Payload:")
        LOGGER.info(json.dumps(messages, indent=2, ensure_ascii=False, default=str))

        if "tools" in kwargs and kwargs["tools"]:
            LOGGER.info("Tools Schemas:")
            LOGGER.info(json.dumps(kwargs["tools"], indent=2, ensure_ascii=False, default=str))
        LOGGER.info("="*106 + "\n")

        response = original_generate(messages=messages, **kwargs)

        LOGGER.info("\n" + "="*40 + " RAW INBOUND API RESPONSE " + "="*40)
        LOGGER.info(f"Success: {response.success} | Status Code: {response.status_code}")
        LOGGER.info(f"Content: {response.content!r}")
        if response.tool_calls:
            LOGGER.info("Parsed Tool Calls:")
            for tc in response.tool_calls:
                LOGGER.info(f"  - ID: {tc.id} | Name: {tc.name} | Args: {tc.arguments}")
        LOGGER.info("="*106 + "\n")
        return response

    client_instance.generate = traced_generate


def _run_agent_scenario(llm_client: Any, search_provider: Any, llm_name: str, search_name: str, query: str):
    """Runs a with/without search scenario for a specific LLM + Search Provider combo."""
    print(f"\n{'=' * 80}")
    print(f"Testing Combination: {llm_name} + {search_name}")
    print(f"{'=' * 80}")

    print(f"\n--- Scenario 1: {llm_name} Without Web Search ---")
    print(f"Prompt: {query!r}\n")
    print("Generating response (this may contain outdated info or a refusal)...")

    resp_without_search = llm_client.ask(query, stream=False)
    print(f"\nResponse:\n{resp_without_search.content}\n")
    print("-" * 60)

    print(f"\n--- Scenario 2: {llm_name} With Web Search ({search_name}) ---")
    print(f"Prompt: {query!r}\n")

    # Reset history to ensure a clean state
    llm_client.messages = []

    # Inject the SearchTool using force_external to ensure the custom provider is used
    search_tool = SearchTool(provider=search_provider, force_external=True)
    llm_client.tool_set.tools = [search_tool]

    print(f"Generating response (The LLM should call the {search_name} search tool automatically)...")
    resp_with_search = llm_client.ask(query, stream=False)

    print(f"\nResponse:\n{resp_with_search.content}\n")

    if llm_client.latest_tool_call_result:
        print(f"\n[Tool Execution Log] The model utilized the following tools: {list(llm_client.latest_tool_call_result.keys())}")
    else:
        print("\n[Tool Execution Log] The model did NOT utilize the search tool.")


def main():
    env = _load_env()
    query = "What is the current weather in Tokyo today, and what are the major news headlines there right now?"
    executed_any = False

    # 1. Test OpenAIChat + Brave Search
    openai_key = _get_api_key(env, "OPENAI_CHAT_API_KEY")
    brave_key = _get_api_key(env, "BRAVE_SEARCH_API_KEY")

    if openai_key:
        executed_any = True
        llm = OpenAIChat(
            api_key=openai_key,
            base_url=env.get("OPENAI_CHAT_BASE_URL", "").strip() or None,
            default_model=env.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo").strip()
        )
        wrap_client_with_tracer(llm)
        search = DummySearchProvider()
        _run_agent_scenario(llm, search, "OpenAIChat", "DummySearch", query)
    else:
        print("\n[skip] Skipping OpenAIChat test due to missing keys in .env.")

    # 2. Test AnthropicMessages + Tavily Search
    anthropic_key = _get_api_key(env, "ANTHROPIC_MESSAGES_API_KEY")

    if anthropic_key:
        executed_any = True
        llm = AnthropicMessages(
            api_key=anthropic_key,
            base_url=env.get("ANTHROPIC_MESSAGES_BASE_URL", "").strip() or None,
            default_model=env.get("ANTHROPIC_MESSAGES_MODEL", "claude-3-haiku-20240307").strip()
        )
        wrap_client_with_tracer(llm)
        search = DummySearchProvider()
        _run_agent_scenario(llm, search, "AnthropicMessages", "DummySearch", query)
    else:
        print("\n[skip] Skipping AnthropicMessages test due to missing keys in .env.")

    if not executed_any:
        print("\nNo tests were executed. Please ensure you have configured pairs of keys in your .env file.")


if __name__ == "__main__":
    main()
