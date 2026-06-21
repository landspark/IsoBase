#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for the LLM providers against real APIs.

This script is intentionally **not** collected by pytest (filename does not
start with ``test_``) and never runs in CI. It reads credentials from a
git-ignored ``.env`` sitting next to this file and exercises both
``OpenAIChat`` and ``AnthropicMessages`` end to end.

Usage:
    1. Copy the template and fill in real values (the real .env is git-ignored):
         cp test/llm/live/.env.example test/llm/live/.env
       Then edit test/llm/live/.env.
    2. Run from the repo root (so the ``isobase`` package is importable):
         python -m test.llm.live.run_llm_basic          # both providers, all scenarios
         python -m test.llm.live.run_llm_basic openai    # only the openai section
         python -m test.llm.live.run_llm_basic anthropic # only the anthropic section

Each provider section runs: non-streaming ask, streaming ask, and a one-tool
round trip. Leave a provider's API_KEY blank/absent in .env to skip it.

Expected .env variables (see .env.example):
    OPENAI_CHAT_BASE_URL / OPENAI_CHAT_API_KEY / OPENAI_CHAT_MODEL
    ANTHROPIC_MESSAGES_BASE_URL / ANTHROPIC_MESSAGES_API_KEY / ANTHROPIC_MESSAGES_MODEL

@File   :   run_llm_basic.py
@Created:   2026/06/07 01:00
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import sys
from os import path
from typing import Any, Dict, Optional

from isobase.llm import LLMClient, AnthropicMessages, OpenAIChat
from isobase.llm.tools import FunctionTool

ENV_PATH = path.join(path.dirname(__file__), ".env")

def __get_weather(city: str) -> tuple:
    """Get the current weather for a city.

    Args:
        city (str): City name
    """
    return True, f"It is 22C and sunny in {city}."


# A trivial tool used to verify the multi-turn tool-calling loop end to end.
WEATHER_TOOL = FunctionTool(mapped_callable=__get_weather, name="get_weather")


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


def _client_kwargs(env: Dict[str, str], prefix: str) -> Optional[Dict[str, Any]]:
    """Builds client kwargs from ``{prefix}_BASE_URL/API_KEY/MODEL`` vars.

    Args:
        env: The parsed .env mapping.
        prefix: The variable-name prefix (e.g. ``"OPENAI_CHAT"``).

    Returns:
        A kwargs dict for the client constructor, or None if no API key is set
        (signalling the section should be skipped).
    """
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None

    kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = env.get(f"{prefix}_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    model = env.get(f"{prefix}_MODEL", "").strip()
    if model:
        kwargs["default_model"] = model
    return kwargs


def _run_scenarios(label: str, client: LLMClient) -> None:
    """Runs the shared scenario battery against an initialized client."""
    print(f"\n{'=' * 60}\n[{label}] non-streaming ask\n{'=' * 60}")
    resp = client.ask("Reply with exactly: pong", stream=False)
    print(f"success={resp.success} status={resp.status_code}")
    print(f"content={resp.content!r}")
    print(f"usage={resp.usage}")

    print(f"\n{'=' * 60}\n[{label}] streaming ask\n{'=' * 60}")
    for chunk in client.ask("Count from 1 to 5, space separated.", stream=True):
        if chunk.content:
            print(chunk.content, end="", flush=True)
    print()

    print(f"\n{'=' * 60}\n[{label}] tool-calling round\n{'=' * 60}")
    tool_resp = client.ask("What's the weather in Paris? Use the tool.",
                           stream=False)
    print(f"content={tool_resp.content!r}")
    print(f"latest_tool_call_result={client.latest_tool_call_result}")


def run_openai(kwargs: Dict[str, Any]) -> None:
    """Runs the scenario battery against the OpenAI-compatible provider."""
    client = OpenAIChat(
        tools=[WEATHER_TOOL],
        **kwargs)
    _run_scenarios("OpenAIChat", client)


def run_anthropic(kwargs: Dict[str, Any]) -> None:
    """Runs the scenario battery against the Anthropic-compatible provider."""
    client = AnthropicMessages(
        tools=[WEATHER_TOOL],
        **kwargs)
    _run_scenarios("AnthropicMessages", client)


def main(only: Optional[str] = None) -> None:
    """Entry point: runs the requested provider section(s).

    Args:
        only: ``"openai"`` or ``"anthropic"`` to run a single section, or None
            for every provider whose API key is present in the .env file.
    """
    env = _load_env()
    runners = {
        "openai": ("OPENAI_CHAT", run_openai),
        "anthropic": ("ANTHROPIC_MESSAGES", run_anthropic),
    }

    selected = [only] if only else list(runners)
    for name in selected:
        if name not in runners:
            sys.exit(f"Unknown provider '{name}'. Choose from: {list(runners)}")
        prefix, runner = runners[name]
        kwargs = _client_kwargs(env, prefix)
        if kwargs is None:
            print(f"\n[skip] no usable {prefix}_API_KEY in .env for '{name}'")
            continue
        runner(kwargs)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
