#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_base.py
@Created:   2026/06/19 03:06
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from typing import Any, Dict, List, Optional
from isobase.llm.tools.base import FunctionTool, ToolSet

def test_function_tool_auto_schema_basic():
    """Tests basic auto-generation of schema from a simple function."""
    def calculate_sum(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    tool = FunctionTool(mapped_callable=calculate_sum)

    assert tool.name == "calculate_sum"
    assert tool.description == "Adds two numbers."
    assert tool.parameters_schema == {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }

def test_function_tool_auto_schema_with_docstring_params():
    """Tests auto-generation extracting parameter descriptions from docstrings."""
    def get_user_info(user_id: int, include_email: bool = False):
        """Fetches user information from the database.

        Args:
            user_id: The unique identifier for the user.
            include_email (bool): Whether to include the email in the response.
        """
        pass

    tool = FunctionTool(mapped_callable=get_user_info)

    assert tool.name == "get_user_info"
    assert tool.description == "Fetches user information from the database."
    assert tool.parameters_schema == {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "integer",
                "description": "The unique identifier for the user."
            },
            "include_email": {
                "type": "boolean",
                "description": "Whether to include the email in the response."
            }
        },
        "required": ["user_id"] # include_email has a default, so it's not required
    }

def test_function_tool_auto_schema_missing_types_and_docs():
    """Tests behavior when types and param docs are missing."""
    def mysterious_function(x, y="default"):
        pass

    tool = FunctionTool(mapped_callable=mysterious_function)

    assert tool.name == "mysterious_function"
    assert tool.description == ""
    assert tool.parameters_schema == {
        "type": "object",
        "properties": {
            "x": {"type": "string"}, # Defaults to string if no type hint
            "y": {"type": "string"}
        },
        "required": ["x"]
    }

def test_function_tool_auto_schema_complex_types():
    """Tests type mapping for lists and dicts."""
    def process_data(items: list, metadata: dict, flag: bool):
        """Processes complex data structures."""
        pass

    tool = FunctionTool(mapped_callable=process_data)

    assert tool.parameters_schema["properties"]["items"]["type"] == "array"
    assert tool.parameters_schema["properties"]["metadata"]["type"] == "object"
    assert tool.parameters_schema["properties"]["flag"]["type"] == "boolean"
    assert tool.parameters_schema["required"] == ["items", "metadata", "flag"]

def test_toolset_execute_one_not_found():
    """ToolSet handles missing tools gracefully."""
    ts = ToolSet([])
    is_success, output = ts._ToolSet__execute_one("missing_tool", "{}")
    assert is_success is False
    assert "not found" in output
