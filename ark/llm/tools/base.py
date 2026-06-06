#! python3
# -*- encoding: utf-8 -*-
"""Core logic for LLM tool/function calling.

This module defines a provider-neutral representation of a callable tool.
``FunctionTool`` holds the tool's name, description, and a JSON-Schema
parameters object as its single source of truth, and renders that neutral
representation into any vendor's wire format on demand
(``to_openai_schema`` / ``to_anthropic_schema``). The matching ``from_*``
classmethods ingest a vendor schema back into the neutral form, so a single
tool definition can move between providers without loss.

@File   :   base.py
@Created:   2026/06/05 00:33
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from json import loads
from typing import Any, Callable, Dict, List, Optional, Tuple


class FunctionToolParameter():
    """Represents a parameter for a function tool.

    Attributes:
        key: The technical key/name of the parameter.
        param_type: The data type of the parameter (e.g., 'string', 'integer').
        description: A brief description of what the parameter represents.
        required: Whether the parameter must be provided by the LLM.
        value: The current value assigned to this parameter.
    """

    def __init__(self, key: str, param_type: str, description: str,
                 required: bool):
        """Initializes a FunctionToolParameter instance."""
        self.key = key
        self.param_type = param_type
        self.description = description
        self.required = required
        self.value: Any = None

    @property
    def name(self) -> str:
        """Returns the technical name of the parameter."""
        return self.key

    @name.setter
    def name(self, value: str):
        """Sets the technical name of the parameter."""
        self.key = value

    @staticmethod
    def parse_function_parameters(
            parameters_dict: dict) -> List['FunctionToolParameter']:
        """Parses parameter info from a JSON-Schema parameters object.

        Args:
            parameters_dict: A JSON-Schema object with ``properties``/``required``.

        Returns:
            A list of FunctionToolParameter instances.
        """
        properties: Dict[str, dict] = parameters_dict.get("properties", {})
        required_params: list = parameters_dict.get("required", [])

        parameters = []
        for key, value in properties.items():
            param_type = value.get("type", "string")
            description = value.get("description", "")
            required = (key in required_params)
            parameters.append(
                FunctionToolParameter(key=key,
                                      param_type=param_type,
                                      description=description,
                                      required=required))

        return parameters


class FunctionTool():
    """A provider-neutral tool the LLM can call to perform actions.

    The neutral core is ``name`` / ``description`` / ``parameters_schema`` (a
    JSON-Schema object). Vendor wire formats are produced on demand via
    ``to_openai_schema`` / ``to_anthropic_schema`` and ingested via the
    ``from_*`` classmethods. Holding the full JSON-Schema object (rather than a
    flattened parameter list) keeps details such as ``enum`` constraints and
    nested object types intact across conversions.

    Attributes:
        name: The name of the function.
        description: A brief description of the tool's purpose.
        parameters_schema: The JSON-Schema object describing the inputs (the
            shared inner shape used by both OpenAI and Anthropic).
        mapped_callable: The Python function associated with this tool.
        tool_function_callable_kwargs: Extra kwargs for the mapped callable.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 parameters_schema: Dict[str, Any] = None,
                 mapped_callable: Optional[Callable] = None,
                 tool_function_callable_kwargs: Dict[str, Any] = None):
        """Initializes a FunctionTool from neutral fields.

        Args:
            name: The tool/function name.
            description: A human-readable description of the tool.
            parameters_schema: The JSON-Schema inputs object (defaults to an
                empty object schema).
            mapped_callable: The Python callable that backs this tool.
            tool_function_callable_kwargs: Static kwargs to pass to the callable.
        """
        self.name = name
        self.description = description
        self.parameters_schema: Dict[str, Any] = (
            parameters_schema or {"type": "object", "properties": {}})
        self.mapped_callable = mapped_callable
        self.tool_function_callable_kwargs = tool_function_callable_kwargs or {}

    @property
    def parameters(self) -> List[FunctionToolParameter]:
        """Returns the parameters parsed from ``parameters_schema`` on demand."""
        return FunctionToolParameter.parse_function_parameters(
            self.parameters_schema)

    @classmethod
    def from_openai_schema(
            cls,
            tool_definition_dict: Dict[str, Any],
            tool_function_mapper: Dict[str, Callable] = None,
            tool_function_callable_kwargs: Dict[str, Any] = None
    ) -> 'FunctionTool':
        """Builds a FunctionTool from an OpenAI-style tool definition.

        Args:
            tool_definition_dict: OpenAI ``{"type": "function", "function": {...}}``.
            tool_function_mapper: A map of tool names to Python callables.
            tool_function_callable_kwargs: Static kwargs to pass to callables.

        Returns:
            A FunctionTool instance.
        """
        function_dict = tool_definition_dict.get("function", {})
        name = function_dict.get("name", "")
        return cls(
            name=name,
            description=function_dict.get("description", ""),
            parameters_schema=function_dict.get("parameters"),
            mapped_callable=(tool_function_mapper or {}).get(name),
            tool_function_callable_kwargs=tool_function_callable_kwargs,
        )

    @classmethod
    def from_anthropic_schema(
            cls,
            tool_definition_dict: Dict[str, Any],
            tool_function_mapper: Dict[str, Callable] = None,
            tool_function_callable_kwargs: Dict[str, Any] = None
    ) -> 'FunctionTool':
        """Builds a FunctionTool from an Anthropic-style tool definition.

        Args:
            tool_definition_dict: Anthropic ``{"name", "description", "input_schema"}``.
            tool_function_mapper: A map of tool names to Python callables.
            tool_function_callable_kwargs: Static kwargs to pass to callables.

        Returns:
            A FunctionTool instance.
        """
        name = tool_definition_dict.get("name", "")
        return cls(
            name=name,
            description=tool_definition_dict.get("description", ""),
            parameters_schema=tool_definition_dict.get("input_schema"),
            mapped_callable=(tool_function_mapper or {}).get(name),
            tool_function_callable_kwargs=tool_function_callable_kwargs,
        )

    def to_openai_schema(self) -> Dict[str, Any]:
        """Renders this tool as an OpenAI-style tool definition.

        Returns:
            An OpenAI ``{"type": "function", "function": {...}}`` dictionary.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        """Renders this tool as an Anthropic-style tool definition.

        Returns:
            An Anthropic ``{"name", "description", "input_schema"}`` dictionary.
            ``input_schema`` always carries ``"type": "object"`` as Anthropic
            requires.
        """
        schema = self.parameters_schema
        if "type" not in schema:
            schema = {**schema, "type": "object"}
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }

    def execute(self, arguments_json: str) -> Any:
        """Executes the mapped callable with the provided JSON arguments.

        Args:
            arguments_json: A JSON string containing arguments from the LLM.

        Returns:
            The result of the callable execution, or an error message string.

        Raises:
            ValueError: If no callable is mapped for this tool.
        """
        if not self.mapped_callable:
            raise ValueError(f"No callable mapped for tool: {self.name}")

        try:
            args = loads(arguments_json)
            if self.tool_function_callable_kwargs:
                args.update(self.tool_function_callable_kwargs)
            return self.mapped_callable(**args)
        except Exception as e:
            return f"Error executing tool '{self.name}': {e}"

    def __str__(self) -> str:
        """Returns a string representation of the tool."""
        callable_name = (self.mapped_callable.__name__
                         if self.mapped_callable else "None")
        return (f"FunctionTool(name={self.name}, "
                f"mapped_callable={callable_name})")


class ToolSet():
    """Manages a collection of FunctionTools and orchestrates their execution.

    Attributes:
        tools: A list of managed FunctionTool instances.
    """

    def __init__(self, tools: List[FunctionTool] = None):
        """Initializes a ToolSet instance."""
        self.tools = tools or []

    def get_openai_schema(self) -> List[Dict[str, Any]]:
        """Returns the list of tool definitions in OpenAI schema format."""
        return [t.to_openai_schema() for t in self.tools]

    def get_anthropic_schema(self) -> List[Dict[str, Any]]:
        """Returns the list of tool definitions in Anthropic schema format."""
        return [t.to_anthropic_schema() for t in self.tools]

    def _execute_one(self, func_name: str,
                     arguments_json: str) -> Tuple[bool, Any]:
        """Executes a single named tool with the given JSON arguments.

        This is the provider-neutral execution core shared by all
        ``execute_tool_calls_*`` formatters.

        Args:
            func_name: The name of the tool to execute.
            arguments_json: A JSON string of arguments from the LLM.

        Returns:
            A tuple of ``(is_success, tool_output)``. ``tool_output`` is the
            raw callable result (or an error message when the tool is missing).
        """
        tool = next((t for t in self.tools if t.name == func_name), None)
        if tool is None:
            return False, f"Tool '{func_name}' not found."

        result = tool.execute(arguments_json)
        # Handle (is_success, content) tuple or raw content.
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        return True, result

    def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        """Executes tool calls and returns OpenAI ``role: "tool"`` messages.

        Args:
            tool_calls: A list of standardized tool calls as returned by the LLM
                (``[{"id", "type": "function", "function": {"name", "arguments"}}]``).

        Returns:
            A tuple containing:
                - A list of response messages with the role 'tool'.
                - A dictionary mapping tool names to their execution success.
        """
        messages = []
        latest_results = {}

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            is_success, tool_output = self._execute_one(
                func_name, tc["function"]["arguments"])
            latest_results[func_name] = is_success

            messages.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "name": func_name,
                "content": str(tool_output)
            })

        return messages, latest_results

    def execute_tool_calls_anthropic(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        """Executes tool calls and returns Anthropic ``tool_result`` blocks.

        Accepts the same standardized tool-call shape as ``execute_tool_calls``
        so both providers share one execution path; only the result formatting
        differs. The returned blocks are meant to be wrapped in a single
        ``{"role": "user", "content": [...]}`` message.

        Args:
            tool_calls: A list of standardized tool calls
                (``[{"id", "type": "function", "function": {"name", "arguments"}}]``).

        Returns:
            A tuple containing:
                - A list of ``tool_result`` content blocks (``is_error`` is set
                  to ``True`` for failed executions).
                - A dictionary mapping tool names to their execution success.
        """
        blocks = []
        latest_results = {}

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            is_success, tool_output = self._execute_one(
                func_name, tc["function"]["arguments"])
            latest_results[func_name] = is_success

            block: Dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": tc["id"],
                "content": str(tool_output),
            }
            if not is_success:
                block["is_error"] = True
            blocks.append(block)

        return blocks, latest_results

    def __bool__(self) -> bool:
        """Returns True if the ToolSet contains at least one tool."""
        return len(self.tools) > 0
