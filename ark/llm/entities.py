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
