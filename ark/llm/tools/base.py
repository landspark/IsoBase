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
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints

from ark.llm.entities import ToolCall

class FunctionTool:
    """A provider-neutral tool the LLM can call to perform actions.

    The neutral core is ``name`` / ``description`` / ``parameters_schema`` (a
    JSON-Schema object). It also holds the Python ``mapped_callable`` to execute.

    Attributes:
        name: The name of the function.
        description: A brief description of the tool's purpose.
        parameters_schema: The JSON-Schema object describing the inputs.
        mapped_callable: The Python function associated with this tool.
        tool_function_callable_kwargs: Extra kwargs for the mapped callable.
    """

    def __init__(self,
                 mapped_callable: Callable,
                 name: Optional[str] = None,
                 description: str = "",
                 parameters_schema: Optional[Dict[str, Any]] = None,
                 tool_function_callable_kwargs: Dict[str, Any] = None):
        """Initializes a FunctionTool from neutral fields.

        While auto-generation of metadata (name, description, parameters_schema) is
        the default behavior using Python's `inspect` and typing hints, developers
        can explicitly pass these arguments to override and bypass the automatic
        generation mechanism. This is useful for customizing or fine-tuning the tool
        schema without modifying the underlying Python function, or when precise
        JSON-Schema definition is required.

        Args:
            mapped_callable: The Python callable that backs this tool.
            name: The tool/function name (defaults to callable's name).
                  Explicitly passing this overrides the name.
            description: A human-readable description (defaults to the first docstring line).
                         Explicitly passing this overrides the description.
            parameters_schema: The JSON-Schema inputs object. If omitted, this attempts to
                               auto-generate the schema using `inspect.signature` and typing hints.
                               Explicitly passing a dictionary overrides and bypasses auto-generation.
            tool_function_callable_kwargs: Static kwargs to pass to the callable.
        """
        self.mapped_callable = mapped_callable
        self.name = name or mapped_callable.__name__

        if description:
            self.description = description
        elif mapped_callable.__doc__:
            # Use the first non-empty line of the docstring as the default description
            lines = [line.strip() for line in mapped_callable.__doc__.split('\n') if line.strip()]
            self.description = lines[0] if lines else ""
        else:
            self.description = ""

        if parameters_schema is not None:
            self.parameters_schema = parameters_schema
        else:
            self.parameters_schema = self.__generate_schema_from_callable(mapped_callable)

        self.tool_function_callable_kwargs = tool_function_callable_kwargs or {}

    @classmethod
    def __generate_schema_from_callable(cls, func: Callable) -> Dict[str, Any]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # Mapping of Python types to JSON Schema types
        type_mapping = {
            str: "string", int: "integer", float: "number",
            bool: "boolean", list: "array", dict: "object", type(None): "null"
        }

        # Best-effort docstring parameter parsing (Google/Sphinx style)
        param_descriptions = {}
        if func.__doc__:
            # Look for lines like "param_name: description" or "param_name (type): description"
            # under an "Args:" section.
            doc_lines = func.__doc__.split('\n')
            in_args_section = False
            for line in doc_lines:
                line = line.strip()
                if line.startswith("Args:") or line.startswith("Parameters:"):
                    in_args_section = True
                    continue
                if in_args_section:
                    if not line or line.endswith(":"): # Empty line or next section
                        # A very crude check to stop parsing args if we hit another section like "Returns:"
                        if re.match(r"^[A-Z][a-z]+:$", line):
                            break
                        continue

                    # Match "param_name: description" or "param_name (type): description"
                    match = re.match(r"^([a-zA-Z0-9_]+)\s*(?:\([^)]+\))?:\s*(.*)$", line)
                    if match:
                        param_descriptions[match.group(1)] = match.group(2)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ["self", "cls"]:
                continue

            param_type = type_hints.get(param_name, Any)
            json_type = type_mapping.get(param_type, "string") # fallback to string

            param_schema: Dict[str, Any] = {"type": json_type}
            if param_name in param_descriptions:
                param_schema["description"] = param_descriptions[param_name]

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        return schema

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

    def __execute_one(self, func_name: str,
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
        self, tool_calls: List[ToolCall]
    ) -> Tuple[List[Any], Dict[str, bool]]:
        """Executes a list of tool calls.

        Args:
            tool_calls: A list of ToolCall objects to execute.

        Returns:
            A tuple containing:
                - A list of execution results (raw outputs from the callables).
                - A dictionary mapping tool names to their execution success.
        """
        results = []
        latest_results = {}

        for tc in tool_calls:
            is_success, tool_output = self.__execute_one(tc.name, tc.arguments)
            latest_results[tc.name] = is_success
            results.append(tool_output)

        return results, latest_results

    def __bool__(self) -> bool:
        """Returns True if the ToolSet contains at least one tool."""
        return len(self.tools) > 0
