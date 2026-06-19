# LLM Module

> 中文文档见 [README.zh-cn.md](./README.zh-cn.md)

## Introduction

The `isobase.llm` module is a provider-neutral client layer for Large Language Models. It exposes one consistent surface (`ask` / `generate` / `generate_stream`) across different vendor APIs, returns a standardized `LLMResponse`, and supports multi-turn tool calling, multimodal image input, streaming, and extended thinking. Two providers are interchangeable today: OpenAI Chat Completion compatible (`OpenAIChat`) and Anthropic Messages compatible (`AnthropicMessages`).

The design goal is that swapping providers should not change calling code: tools are defined once in a neutral format and rendered to each vendor's wire schema on demand; every method returns the same `LLMResponse`.

## Features

- **Provider-neutral surface**: `OpenAIChat` and `AnthropicMessages` share the same public methods and return the same `LLMResponse`, so they are drop-in interchangeable.
- **Neutral tool format with auto-generated schemas**: a single `FunctionTool` can automatically extract its `name`, `description`, and JSON `parameters_schema` directly from the signature and docstring of a mapped Python callable. No vendor format is treated as the canonical one, and providers translate the neutral `ToolCall` on the fly.
- **Multi-turn tool calling**: `ask()` runs a bounded recursive tool-calling loop (`max_tool_rounds`), executing Python callables and feeding results back to the model.
- **Streaming and non-streaming**: both paths are implemented for each provider; streaming yields incremental text/reasoning chunks then a final summary carrying tool calls and usage.
- **Extended thinking**: opt-in via `thinking` (e.g. `{"type": "adaptive"}` for Anthropic); thinking text surfaces into `LLMResponse.reasoning_content`.
- **Multimodal image input**: `build_user_message_content` attaches PIL images, rendered to each vendor's image block format.
- **Standardized response & usage**: every call returns `LLMResponse` (success, status_code, content, tool_calls, reasoning_content, usage, raw_response) with `TokenUsage` token accounting.
- **Compatible-gateway resilience**: the Anthropic streaming path tolerates gateways that send a non-conformant `message_start.content: null` (which crashes the SDK's high-level stream helper).

## Quick Start

`isobase.llm` is **not** re-exported at the top level, so import from `isobase.llm` directly.

```python
from isobase.llm import OpenAIChat, AnthropicMessages

# --- OpenAI Chat Completion compatible ---
client = OpenAIChat(api_key="sk-...", default_model="gpt-4o-mini")
resp = client.ask("Say pong.", stream=False)
print(resp.content, resp.usage.total_tokens)

# --- Anthropic Messages compatible (interchangeable) ---
client = AnthropicMessages(api_key="sk-ant-...", default_model="claude-opus-4-8")
resp = client.ask("Say pong.", stream=False)
print(resp.content, resp.usage.total_tokens)
```

### Streaming

```python
for chunk in client.ask("Count from 1 to 5.", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### Tool calling

Tools are defined once in the neutral format and work with either provider. You can let the framework automatically generate the JSON schema from your Python function's signature and docstring, or explicitly provide them to override this behavior.

#### Option A: Automatic Generation (Default)

```python
from isobase.llm.tools import FunctionTool

def get_weather(city: str) -> tuple[bool, str]:
    """Get the current weather for a city.
    
    Args:
        city: The name of the city.
    """
    return True, f"It is 22C and sunny in {city}."  # (is_success, content)

# Automatically extracts name, description, and parameter types from the callable
weather_tool = FunctionTool(get_weather)
```

#### Option B: Explicit Overrides (Manual Schema)

If you need precise control over the metadata and parameters schema exposed to the LLM—or want to decouple the JSON-Schema from the Python function signature—you can explicitly provide `name`, `description`, and `parameters_schema` to override auto-generation:

```python
from isobase.llm.tools import FunctionTool

# Explicitly override metadata and schema manually
weather_tool = FunctionTool(
    mapped_callable=get_weather,
    name="get_weather_override",
    description="Fetch the current weather condition and temperature for a given city.",
    parameters_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The target city name (e.g. 'Paris', 'Tokyo')"
            }
        },
        "required": ["city"]
    }
)
```

With either option, wire the tool into the client:

```python
client = AnthropicMessages(
    api_key="sk-ant-...",
    tools=[weather_tool],
)
resp = client.ask("What's the weather in Paris? Use the tool.")
print(resp.content)                       # final answer after the tool round
print(client.latest_tool_call_result)     # {'get_weather': True}
```

### Extended thinking (Anthropic)

```python
client = AnthropicMessages(api_key="sk-ant-...", thinking={"type": "adaptive"})
resp = client.ask("Solve 27 * 453 step by step.")
print(resp.reasoning_content)  # thinking text
print(resp.content)            # final answer
```

## Module Structure

- `entities.py` — `LLMResponse`, `TokenUsage`, and `ToolCall` dataclasses; the standardized provider-neutral data types used for inputs and returns.
- `providers/base.py` — `BaseLLMClient` abstract interface (`generate`, `generate_stream`) plus `build_user_message_content` for multimodal messages.
- `providers/openai_chat.py` — `OpenAIChat`, the OpenAI Chat Completion compatible client.
- `providers/anthropic_messages.py` — `AnthropicMessages`, the Anthropic Messages compatible client (named after the Messages API, mirroring how `OpenAIChat` is named after the Chat Completion API).
- `tools/base.py` — `FunctionTool`, `ToolSet`; the neutral tool representation and execution engine.

Image helpers live in `isobase/core/image_service.py` (`convert_image_to_data_url` for OpenAI, `convert_image_to_base64` for Anthropic).

Manual live API smoke tests (not run by pytest) live in `test/llm/live/` — copy `.env.example` to `.env`, fill in credentials, and run `python -m test.llm.live.run_live`.

## Status

### Done

- `BaseLLMClient` abstract interface and standardized `LLMResponse` / `TokenUsage`.
- `OpenAIChat` provider: non-streaming, streaming, multi-turn tool calling, multimodal image input, history preservation on error.
- `AnthropicMessages` provider: non-streaming, streaming, multi-turn tool calling, multimodal image input, extended thinking, top-level `system`, mandatory `max_tokens`, and resilience to compatible gateways that send `message_start.content: null`.
- Neutral `FunctionTool` with bidirectional OpenAI/Anthropic schema conversion and a shared execution core (`ToolSet.execute_tool_calls` / `execute_tool_calls_anthropic`).
- Unit tests mocking each SDK (`test/llm/providers/`), plus a manual live runner (`test/llm/live/`).

### Not yet done (future)

- **RAG**: retrieval-augmented generation is not implemented; the `ToolSet` / `LLMResponse` entry points leave room for it.
- **MCP**: Model Context Protocol integration is not implemented (room is left at the same tool entry points).
- **More providers**: e.g. OpenAI Responses API (would be a separate class, not `OpenAIChat`), Gemini, etc.
- **Structured outputs**: provider-native JSON-schema / strict-output modes are not surfaced through the neutral layer yet.
- **Async clients**: only synchronous clients exist today.
- **Prompt caching / batching**: vendor cost-optimization features are not exposed.
- **Automated live tests**: the live runner is manual and human-inspected; it is not wired into pytest (would require credential-guarded opt-in).
