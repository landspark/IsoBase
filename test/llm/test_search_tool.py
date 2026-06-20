#! python3
# -*- encoding: utf-8 -*-
"""Tests for SearchTool and its integration with providers.

@File   :   test_search_tool.py
@Created:   2026/06/20 18:36
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import json
from isobase.llm.tools.search import SearchTool
from isobase.llm.tools.base import ToolSet
from isobase.llm.providers.openai_chat import OpenAIChat
from isobase.llm.providers.anthropic_messages import AnthropicMessages

def test_search_tool_initialization_with_callable():
    def mock_search(query: str):
        return f"Results for {query}"

    tool = SearchTool(mapped_callable=mock_search, force_external=True)
    assert tool.force_external is True
    assert tool.mapped_callable is not None
    assert tool.name == "mock_search"

    result = tool.execute(json.dumps({"query": "test"}))
    assert result == "Results for test"

def test_search_tool_initialization_without_callable():
    tool = SearchTool(force_external=False)
    assert tool.force_external is False
    assert tool.mapped_callable is None
    assert tool.name == "web_search"
    assert tool.parameters_schema["type"] == "object"

    # Executing local callable when there is none should return error msg
    result = tool.execute(json.dumps({"query": "test"}))
    assert "Error: SearchTool was invoked via local function calling" in result

def test_openai_schema_generation():
    tool_native = SearchTool(force_external=False)

    def mock_search(query: str): pass
    tool_custom = SearchTool(mapped_callable=mock_search, force_external=True)

    client = OpenAIChat(api_key="test", tools=ToolSet([tool_native, tool_custom]))

    # We test via the private method for schema verification
    schemas = client._OpenAIChat__build_tools_schema()

    # The first tool (force_external=False) should be built-in web_search
    assert schemas[0] == {"type": "web_search"}

    # The second tool (force_external=True) should be a standard function
    assert schemas[1]["type"] == "function"
    assert schemas[1]["function"]["name"] == "mock_search"

def test_anthropic_schema_generation():
    tool_native = SearchTool(force_external=False)

    def mock_search(query: str): pass
    tool_custom = SearchTool(mapped_callable=mock_search, force_external=True)

    client = AnthropicMessages(api_key="test", tools=ToolSet([tool_native, tool_custom]))

    # Anthropic doesn't have a specific type for search yet, it should fallback
    schemas = client._AnthropicMessages__build_tools_schema()

    assert schemas[0]["name"] == "web_search"
    assert schemas[0]["input_schema"]["type"] == "object"
    assert schemas[1]["name"] == "mock_search"
    assert schemas[1]["input_schema"]["type"] == "object"
