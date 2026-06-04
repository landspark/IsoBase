#! python3
# -*- encoding: utf-8 -*-
"""Core data structures for LLM responses and usage.

@File   :   entities.py
@Created:   2026/06/05 00:50
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional

@dataclass
class TokenUsage:
    """Represents token consumption for a request."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

@dataclass
class LLMResponse:
    """Structured response from an LLM provider."""
    success: bool
    status_code: int
    content: str = ""
    role: str = "assistant"
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None
    reasoning_content: str = ""

    def __str__(self):
        return f"LLMResponse(success={self.success}, status={self.status_code}, content_len={len(self.content)})"
