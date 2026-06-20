#! python3
# -*- encoding: utf-8 -*-
"""Search tool integration for LLM capabilities.

@File   :   search.py
@Created:   2026/06/20 18:33
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/

"""

from typing import Any, Callable, Dict, Optional

from isobase.llm.tools.base import FunctionTool


class SearchTool(FunctionTool):
    """A specialized tool for Web Search capabilities.

    This tool acts as a unified interface for web search. When used with an LLM
    provider that supports native/built-in search (e.g., OpenAI, Qwen, GLM), the
    adapter will intercept this tool and convert it to the provider's native format
    (e.g., `{"type": "web_search"}`), handling the search server-side.

    For providers without native support (e.g., DeepSeek, Anthropic) or when
    `force_external=True` is set, this tool functions as a standard custom function
    calling tool, executing the provided `mapped_callable` locally.

    Attributes:
        force_external: If True, bypasses native provider search and forces the
                        use of the `mapped_callable` via standard function calling.
    """

    def __init__(self,
                 mapped_callable: Optional[Callable] = None,
                 force_external: bool = False,
                 name: Optional[str] = None,
                 description: str = "",
                 parameters_schema: Optional[Dict[str, Any]] = None,
                 tool_function_callable_kwargs: Dict[str, Any] = None):
        """Initializes a SearchTool.

        Args:
            mapped_callable: The Python callable to execute if falling back to custom
                             function calling. Must be provided if `force_external=True`
                             or if the LLM provider lacks native search support.
            force_external: Set to True to force external/local search via the mapped
                            callable, bypassing any native provider search capabilities.
            name: Override the tool name.
            description: Override the tool description.
            parameters_schema: Override the auto-generated JSON schema.
            tool_function_callable_kwargs: Static kwargs to pass to the callable.

        Raises:
            ValueError: If `force_external=True` but no `mapped_callable` is provided.
        """
        self.force_external = force_external

        if force_external and mapped_callable is None:
            raise ValueError(
                "A `mapped_callable` must be provided when `force_external=True`."
            )

        # If a callable is provided, initialize standard FunctionTool behavior
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
            self.name = name or "web_search"
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
                    "but no `mapped_callable` was provided during initialization.")
        return super().execute(arguments_json)
