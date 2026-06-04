#! python3
# -*- encoding: utf-8 -*-
"""Core logic for LLM tool/function calling.

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
        """Parses parameter info from a function definition dictionary.

        Args:
            parameters_dict: The 'parameters' dictionary from a tool definition.

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
    """Represents a tool that the LLM can call to perform actions.

    Attributes:
        tool_definition_dict: The original OpenAI-style tool definition.
        name: The name of the function.
        description: A brief description of the tool's purpose.
        parameters: A list of parsed FunctionToolParameter objects.
        mapped_callable: The Python function associated with this tool.
        tool_function_callable_kwargs: Extra kwargs for the mapped callable.
    """

    def __init__(self,
                 tool_definition_dict: Dict[str, Any],
                 tool_function_mapper: Dict[str, Callable] = None,
                 tool_function_callable_kwargs: Dict[str, Any] = None):
        """Initializes a FunctionTool with a definition and a callable mapper.

        Args:
            tool_definition_dict: The OpenAI-style tool definition dictionary.
            tool_function_mapper: A map of tool names to Python callables.
            tool_function_callable_kwargs: Static kwargs to pass to callables.
        """
        self.tool_definition_dict = tool_definition_dict
        function_dict = tool_definition_dict.get("function", {})
        self.name = function_dict.get("name", "")
        self.description = function_dict.get("description", "")
        self.parameters = FunctionToolParameter.parse_function_parameters(
            function_dict.get("parameters", {}))

        tool_function_mapper = tool_function_mapper or {}
        self.mapped_callable = tool_function_mapper.get(self.name)
        self.tool_function_callable_kwargs = tool_function_callable_kwargs or {}

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
        return [t.tool_definition_dict for t in self.tools]

    def execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
        """Executes a list of tool calls and returns formatted messages.

        Args:
            tool_calls: A list of tool calls as returned by the LLM.

        Returns:
            A tuple containing:
                - A list of response messages with the role 'tool'.
                - A dictionary mapping tool names to their execution success.
        """
        messages = []
        latest_results = {}

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            tool = next((t for t in self.tools if t.name == func_name), None)

            if tool:
                result = tool.execute(tc["function"]["arguments"])
                # Handle (is_success, content) tuple or raw content string.
                if isinstance(result, tuple) and len(result) == 2:
                    is_success, tool_output = result
                else:
                    is_success, tool_output = True, str(result)

                latest_results[func_name] = is_success
            else:
                tool_output = f"Tool '{func_name}' not found."
                latest_results[func_name] = False

            messages.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "name": func_name,
                "content": str(tool_output)
            })

        return messages, latest_results

    def __bool__(self) -> bool:
        """Returns True if the ToolSet contains at least one tool."""
        return len(self.tools) > 0
