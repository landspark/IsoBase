#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for testing multiple complex tools and callbacks.

Usage:
    python -m test.llm.live.run_agent_complex

@File   :   run_agent_complex.py
@Created:   2026/06/21 03:24
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import sys
from os import path
from typing import Any, Dict, Optional

from isobase.llm import LLMClient, OpenAIChat, AnthropicMessages
from isobase.llm.callbacks import BaseLLMCallback
from isobase.llm.entities import SearchResult, SearchResultItem
from isobase.llm.tools import FunctionTool
from isobase.llm.tools.search import SearchTool
from isobase.llm.tools.search.base import BaseSearchProvider

ENV_PATH = path.join(path.dirname(__file__), ".env")


# --- Environment Loading ---

def _load_env() -> Dict[str, str]:
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


# --- Custom Callback Tracker ---

class ComplexToolsCallback(BaseLLMCallback):
    """Custom callback to capture and output tool lifecycles at the application layer."""

    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        print("\n" + "=" * 25 + " CALLBACK: TOOL STARTED " + "=" * 25)
        print(f"Tool Name: {tool_name}")
        print(f"Arguments: {arguments}")
        print("=" * 74)

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        print("\n" + "=" * 25 + " CALLBACK: TOOL COMPLETED " + "=" * 25)
        print(f"Tool Name: {tool_name}")
        print(f"Raw Output Type: {type(result).__name__}")
        if isinstance(result, SearchResult):
            # Proving we get back the exact dataclass
            print(f"Found {len(result.results)} web results. Success={result.success}")
        else:
            print(f"Output: {result}")
        print("=" * 76 + "\n")

    def on_error(self, error: Exception) -> None:
        print(f"\n[Callback Error] Tool run failed: {error}\n")


# --- 2 Mock Local Function Tools ---

def get_weather(city: str) -> str:
    """Get the current weather for a specific city.

    Args:
        city: The name of the city.
    """
    weather_data = {
        "tokyo": "18°C, Cloudy with light rain",
        "shanghai": "26°C, Sunny and humid",
        "paris": "22°C, Overcast",
        "new york": "24°C, Thunderstorms"
    }
    normalized_city = city.lower().strip()
    return weather_data.get(normalized_city, f"20°C, partly cloudy in {city}")


def calculate_math(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math string to evaluate, like '2 + 2' or '333 - 330'.
    """
    try:
        cleaned = "".join(c for c in expression if c in "0123456789+-*/(). ")
        # In a real app never use eval, but safe for a quick dummy tool smoke test
        res = eval(cleaned, {"__builtins__": None}, {})
        return f"Result: {res}"
    except Exception as e:
        return f"Error evaluating expression '{expression}': {e}"


# --- Mock Search Provider for 2 Search Tools ---

class MockSearchProvider(BaseSearchProvider):
    """Mock provider to perform deterministic local web searches."""

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        normalized = query.lower()
        if "tokyo" in normalized and "tower" in normalized:
            return SearchResult(
                success=True,
                results=[SearchResultItem(
                    title="Tokyo Tower Heights and Stats",
                    url="https://tokyotower.example.com/stats",
                    snippet="Tokyo Tower stands at an official height of 333 meters (1,092 feet)."
                )]
            )
        elif "eiffel" in normalized and "tower" in normalized:
            return SearchResult(
                success=True,
                results=[SearchResultItem(
                    title="Eiffel Tower Official Dimensions",
                    url="https://eiffeltower.example.com/dimensions",
                    snippet="The Eiffel Tower height is 330 meters (1,083 feet) up to the tip."
                )]
            )
        return SearchResult(
            success=True,
            results=[SearchResultItem(
                title=f"Search Result for '{query}'",
                url="https://mocksearch.example.com",
                snippet=f"No mock snippet matches '{query}'."
            )]
        )


# --- Core Runner Scenarios ---

def _run_complex_scenarios(llm_client: LLMClient, prompt: str):
    # Setup Tools
    weather_tool = FunctionTool(mapped_callable=get_weather, name="get_weather")
    math_tool = FunctionTool(mapped_callable=calculate_math, name="calculate_math")
    # force_external=True bypasses Qwen/OpenAI built-in searches to test our custom provider pipeline
    search_tool = SearchTool(provider=MockSearchProvider(), force_external=True)

    llm_client.tool_set.tools = [weather_tool, math_tool, search_tool]
    cb = ComplexToolsCallback()

    print("\n" + "#" * 80)
    print(f"RUNNING NON-STREAMING TEST MODE (stream=False)")
    print("#" * 80)
    llm_client.messages = []  # Clear history

    resp_non_stream = llm_client.ask(prompt, stream=False, callbacks=[cb])
    print(f"\n[Final Output]:\n{resp_non_stream.content}")

    print("\n" + "#" * 80)
    print(f"RUNNING STREAMING TEST MODE (stream=True)")
    print("#" * 80)
    llm_client.messages = []  # Clear history

    print("\n[Response Stream Starts]:\n")
    # Stream mode loop which processes chunks sequentially as they arrive
    for chunk in llm_client.ask(prompt, stream=True, callbacks=[cb]):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        # Note: If extended thinking is active, print it out
        if getattr(chunk, "reasoning_content", None):
            print(f"\033[90m{chunk.reasoning_content}\033[0m", end="", flush=True)

    print("\n\n[Response Stream Ends]\n")


def main():
    env = _load_env()
    prompt = (
        "Please perform the following tasks step-by-step using your available tools, "
        "and compile a final summary:\n"
        "1. Use web search to find the height of Tokyo Tower.\n"
        "2. Use web search to find the height of the Eiffel Tower.\n"
        "3. Use the calculate_math tool to calculate the difference in height "
        "between Tokyo Tower and Eiffel Tower based on your searches.\n"
        "4. Use the get_weather tool to check the weather in Tokyo so I know if it's "
        "a good day to visit."
    )

    # Test OpenAI Client
    openai_key = _get_api_key(env, "OPENAI_CHAT_API_KEY")
    if openai_key:
        llm = OpenAIChat(
            api_key=openai_key,
            base_url=env.get("OPENAI_CHAT_BASE_URL", "").strip() or None,
            default_model=env.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo").strip()
        )
        print(f"\nEvaluating OpenAI Provider...")
        _run_complex_scenarios(llm, prompt)

    # Test Anthropic Client
    anthropic_key = _get_api_key(env, "ANTHROPIC_MESSAGES_API_KEY")
    if anthropic_key:
        llm = AnthropicMessages(
            api_key=anthropic_key,
            base_url=env.get("ANTHROPIC_MESSAGES_BASE_URL", "").strip() or None,
            default_model=env.get("ANTHROPIC_MESSAGES_MODEL", "claude-3-haiku-20240307").strip()
        )
        print(f"\nEvaluating Anthropic Provider...")
        _run_complex_scenarios(llm, prompt)


if __name__ == "__main__":
    main()
