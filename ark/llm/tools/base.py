#! python3
# -*- encoding: utf-8 -*-
"""Core logic for LLM tool/function calling.

@File   :   base.py
@Created:   2026/06/05 00:33
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from typing import Any, Callable, Dict, List, Optional
from json import loads

class FunctionToolParameter():
    """Represents a parameter for a function tool."""

    def __init__(self, key: str, param_type: str, description: str, required: bool):
        self.key = key
        self.param_type = param_type
        self.description = description
        self.required = required
        self.value: Any = None

    @property
    def name(self) -> str:
        return self.key
    
    @name.setter
    def name(self, value: str):
        self.key = value

    @staticmethod
    def parse_function_parameters(parameters_dict: dict) -> List['FunctionToolParameter']:
        """Parses parameters information from a function definition dictionary."""
        properties: Dict[str, dict] = parameters_dict.get("properties", {})
        required_params: list = parameters_dict.get("required", [])
        
        parameters = []
        for key, value in properties.items():
            param_type = value.get("type", "string")
            description = value.get("description", "")
            required = (key in required_params)
            parameters.append(FunctionToolParameter(
                key=key, 
                param_type=param_type, 
                description=description, 
                required=required
            ))

        return parameters

class FunctionTool():
    """Represents a tool that the LLM can call."""

    def __init__(
        self, 
        tool_definition_dict: Dict[str, Any], 
        tool_function_mapper: Dict[str, Callable] = None, 
        tool_function_callable_kwargs: Dict[str, Any] = None
    ):
        """
        Initializes a Function Tool.
        
        Args:
            tool_definition_dict: The OpenAI-style tool definition.
            tool_function_mapper: Map of function names to Python callables.
            tool_function_callable_kwargs: Additional kwargs for the callables.
        """
        self.tool_definition_dict = tool_definition_dict
        function_dict = tool_definition_dict.get("function", {})
        self.name = function_dict.get("name", "")
        self.description = function_dict.get("description", "")
        self.parameters = FunctionToolParameter.parse_function_parameters(
            function_dict.get("parameters", {})
        )
        
        tool_function_mapper = tool_function_mapper or {}
        self.mapped_callable = tool_function_mapper.get(self.name)
        self.tool_function_callable_kwargs = tool_function_callable_kwargs or {}

    def execute(self, arguments_json: str) -> Any:
        """Executes the mapped callable with the provided JSON arguments."""
        if not self.mapped_callable:
            raise ValueError(f"No callable mapped for tool: {self.name}")
        
        try:
            args = loads(arguments_json)
            if self.tool_function_callable_kwargs:
                args.update(self.tool_function_callable_kwargs)
            return self.mapped_callable(**args)
        except Exception as e:
            return f"Error executing tool '{self.name}': {e}"

    def __str__(self):
        callable_name = self.mapped_callable.__name__ if self.mapped_callable else "None"
        return f"FunctionTool(name={self.name}, mapped_callable={callable_name})"
