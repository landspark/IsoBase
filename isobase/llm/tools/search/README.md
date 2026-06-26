# Search Tool

> 中文文档见 [README.zh-cn.md](./README.zh-cn.md)

The `isobase.llm.tools.search` package provides an enterprise-ready, dual-architecture web search integration for LLMs. It seamlessly supports both the native search capabilities of modern models and custom fallback mechanisms using third-party search APIs.

## Architecture

The search functionality is built around a **Dual-Architecture** pattern, designed to maximize efficiency and minimize configuration while remaining robust across different LLM providers.

1. **Native/Built-in Search** (Zero Configuration)
   - When an LLM provider natively supports web search (e.g., OpenAI, Qwen, GLM), the `SearchTool` is intercepted by the provider's adapter in `IsoBase` (like `OpenAIChat`).
   - The tool is automatically translated into the provider's specific native wire schema (e.g., `{"type": "web_search"}`).
   - The search is executed server-side by the LLM vendor, saving you tokens, latency, and the need to manage your own search engine API keys.

2. **Custom Search Provider** (Fallback & Enterprise Integration)
   - For LLMs that lack native search support (e.g., DeepSeek, Anthropic) or when strict control over the search engine is required, you can mount a `BaseSearchProvider` into the `SearchTool`.
   - The `SearchTool` acts as a standard custom function (via `FunctionTool`). When the LLM decides to search, `IsoBase` executes the query locally using your chosen provider and returns the results to the LLM.

## Features

- **Decoupled Search Logic**: Implement the `BaseSearchProvider` protocol to inject custom, enterprise-internal search endpoints (like ElasticSearch or internal corporate wikis) without changing any LLM orchestration logic.
- **Built-in Integrations**: 
  - `BraveSearchProvider`: Lightweight REST API search with a generous free tier (2000 requests/month).
  - `TavilySearchProvider`: Deep AI-native search that parses raw web pages and returns clean markdown content.
- **Strict Naming Enforcement**: The tool always presents itself to the LLM as `web_search` (unless explicitly overridden) to prevent model confusion when switching providers.

## Quick Start

### Using Native Search (Recommended for OpenAI / Qwen)

Just instantiate a blank `SearchTool`. The `OpenAIChat` adapter will handle the rest.

```python
from isobase.llm import OpenAIChat
from isobase.llm.tools.search import SearchTool

# Force_external is False by default. The adapter turns this into native {"type": "web_search"}.
search_tool = SearchTool()

client = OpenAIChat(api_key="sk-...", tools=[search_tool])
response = client.ask("What is the current weather in Tokyo?")
print(response.content)
```

### Using a Custom Provider (Required for Anthropic / DeepSeek)

Pass a concrete implementation of `BaseSearchProvider` to the `SearchTool`. Set `force_external=True` if you want to bypass a provider's native search and force the use of your own.

```python
from isobase.llm import AnthropicMessages
from isobase.llm.tools.search import SearchTool, TavilySearchProvider

tavily_provider = TavilySearchProvider(api_key="tvly-...")

# For Anthropic, it acts as a standard tool call handled by IsoBase locally
search_tool = SearchTool(provider=tavily_provider, force_external=True)

client = AnthropicMessages(api_key="sk-ant-...", tools=[search_tool])
response = client.ask("Summarize the latest major news from Japan today.")
print(response.content)
```

## Creating a Custom Enterprise Provider

To inject a custom search backend (e.g., querying your internal company database), simply inherit from `BaseSearchProvider` and implement the `search` method:

```python
from typing import Any, Dict, List
from isobase.llm.entities import SearchResult, SearchResultItem
from isobase.llm.tools.search import BaseSearchProvider, SearchTool

class CorporateWikiSearchProvider(BaseSearchProvider):
    def search(self, query: str, **kwargs: Any) -> SearchResult:
        # Your custom logic to query an internal database
        return SearchResult(
            success=True,
            results=[
                SearchResultItem(
                    title="Internal Company Policy",
                    url="https://wiki.corp.local/policy",
                    snippet="Our remote work policy..."
                )
            ]
        )

# Inject it into the tool!
internal_search = SearchTool(provider=CorporateWikiSearchProvider(), force_external=True)
```
