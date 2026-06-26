#! python3
# -*- encoding: utf-8 -*-
"""Core data structures for LLM responses and usage.

@File   :   entities.py
@Created:   2026/06/05 00:50
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TokenUsage:
    """Represents token consumption for an LLM request.

    Attributes:
        input_tokens: Number of tokens in the input prompt.
        output_tokens: Number of tokens in the generated response.
        total_tokens: Sum of input and output tokens.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    """A provider-neutral representation of a tool/function call requested by the LLM.

    Attributes:
        id: The unique identifier for this tool call.
        name: The name of the tool/function to call.
        arguments: A JSON-serialized string of the arguments.
    """
    id: str
    name: str
    arguments: str


@dataclass
class SearchResultItem:
    """Represents a single standardized search result from a web search provider.

    Attributes:
        title: The title of the search result page.
        url: The absolute URL of the page.
        snippet: A short summary or block of content.
        raw_content: Optional full-page parsed content or markdown.
    """
    title: str
    url: str
    snippet: str
    raw_content: Optional[str] = None


@dataclass
class SearchResult:
    """Represents the complete standardized outcome of a search provider query.

    Attributes:
        success: Whether the search operation succeeded.
        results: A list of SearchResultItem instances.
        error: Optional error message if the search failed.
    """
    success: bool
    results: List[SearchResultItem] = field(default_factory=list)
    error: Optional[str] = None

    def __str__(self) -> str:
        """Standardized string representation optimized for LLM prompt consumption."""
        if not self.success:
            return f"Search failed: {self.error}"

        if not self.results:
            return "No search results found."

        formatted_items = []
        for idx, item in enumerate(self.results, 1):
            block = f"[{idx}] {item.title}\nURL: {item.url}\nSnippet: {item.snippet}"
            if item.raw_content:
                block += f"\nFull Content:\n{item.raw_content}"
            formatted_items.append(block)

        return "\n\n---\n\n".join(formatted_items)


@dataclass
class LLMResponse:
    """Structured response from an LLM provider.

    Attributes:
        success: Whether the request was successful.
        status_code: The HTTP status code or internal error code.
        content: The main text content of the response.
        role: The role of the responder (usually 'assistant').
        usage: Detailed token consumption statistics.
        tool_calls: A list of tool calls requested by the model.
        raw_response: The original response object from the SDK/API.
        reasoning_content: Internal reasoning or chain-of-thought, if available.
    """
    success: bool
    status_code: int
    content: str = ""
    role: str = "assistant"
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Any = None
    reasoning_content: str = ""

    def __str__(self) -> str:
        """Returns a string representation of the response summary."""
        return (f"LLMResponse(success={self.success}, status={self.status_code}, "
                f"content_len={len(self.content)})")
