#! python3
# -*- encoding: utf-8 -*-
"""Search tool engine integration for LLM capabilities.

@File   :   engine.py
@Created:   2026/06/20 23:51
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import Any, Callable, Dict, Optional

from ..base import FunctionTool
from .base import BaseSearchProvider


class SearchTool(FunctionTool):
    """A specialized tool for Web Search capabilities.

    This tool acts as a unified interface for web search. When used with an LLM
    provider that supports native/built-in search (e.g., OpenAI, Qwen, GLM), the
    adapter will intercept this tool and convert it to the provider's native format
    (e.g., `{"type": "web_search"}`), handling the search server-side.

    For providers without native support (e.g., DeepSeek, Anthropic) or when
    `force_external=True` is set, this tool functions as a standard custom function
    calling tool. It can be instantiated with a concrete `BaseSearchProvider` or a
    raw `mapped_callable` to execute the search locally.

    Attributes:
        force_external: If True, bypasses native provider search and forces the
                        use of local execution via standard function calling.
        provider: An optional `BaseSearchProvider` implementation.
    """

    def __init__(self,
                 provider: Optional[BaseSearchProvider] = None,
                 mapped_callable: Optional[Callable] = None,
                 force_external: bool = False,
                 name: str = "web_search",
                 description: str = "",
                 parameters_schema: Optional[Dict[str, Any]] = None,
                 tool_function_callable_kwargs: Dict[str, Any] = None):
        """Initializes a SearchTool.

        Args:
            provider: A concrete implementation of `BaseSearchProvider`. If provided,
                      its `search` method will automatically become the mapped callable.
            mapped_callable: The Python callable to execute if falling back to custom
                             function calling. (Ignored if `provider` is passed).
            force_external: Set to True to force external/local search via the mapped
                            callable or provider, bypassing any native provider search capabilities.
            name: Override the tool name. Defaults to "web_search".
            description: Override the tool description.
            parameters_schema: Override the auto-generated JSON schema.
            tool_function_callable_kwargs: Static kwargs to pass to the callable.

        Raises:
            ValueError: If `force_external=True` but neither `provider` nor `mapped_callable` is provided.
        """
        self.force_external = force_external
        self.provider = provider

        # Prioritize the provider's search method if an instance is passed
        if provider is not None:
            mapped_callable = provider.search

        if force_external and mapped_callable is None:
            raise ValueError(
                "A `provider` or `mapped_callable` must be provided when `force_external=True`."
            )

        # If a callable is resolved, initialize standard FunctionTool behavior
        if mapped_callable:
            super().__init__(
                mapped_callable=mapped_callable,
                name=name,
                description=description,
                parameters_schema=parameters_schema,
                tool_function_callable_kwargs=tool_function_callable_kwargs
            )
        else:
            # When no callable is provided, this tool relies purely on native interception.
            # We initialize the base class minimally just to satisfy the schema generation.
            self.mapped_callable = None
            self.name = name
            self.description = description or "Search the web for up-to-date information."
            self.parameters_schema = parameters_schema or {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query."
                    }
                },
                "required": ["query"]
            }
            self.tool_function_callable_kwargs = tool_function_callable_kwargs or {}

    def execute(self, arguments_json: str) -> Any:
        """Executes the search tool locally.

        This method is only reached if the tool is being executed via standard
        function calling (either because `force_external=True` or the provider
        doesn't support native search).

        Args:
            arguments_json: A JSON string containing arguments from the LLM.

        Returns:
            The result of the callable execution, or an error message string.
        """
        if not self.mapped_callable:
            return ("Error: SearchTool was invoked via local function calling, "
                    "but no `provider` or `mapped_callable` was provided during initialization.")
        return super().execute(arguments_json)
