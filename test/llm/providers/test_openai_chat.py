#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_openai_chat.py
@Created:   2026/06/05 00:13
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from isobase.llm.providers.openai_chat import OpenAIChat
from isobase.llm import LLMResponse
from isobase.llm.tools import FunctionTool

@pytest.fixture
def mock_openai_chat():
    """Fixture to mock the OpenAI client."""
    with patch("isobase.llm.providers.openai_chat.OpenAI") as mock:
        yield mock

def test_openai_chat_init(mock_openai_chat):
    """Tests initialization and system instruction setup."""
    client = OpenAIChat(api_key="test-key", instructions="System rules")
    assert len(client.messages) == 1
    assert client.messages[0]["content"] == "System rules"

def test_openai_chat_atomic_completion(mock_openai_chat):
    """Tests the atomic generate method."""
    mock_instance = mock_openai_chat.return_value
    m = MagicMock()
    m.choices = [MagicMock(message=MagicMock(content="Atomic Response", tool_calls=None, reasoning_content=None))]
    m.usage = None
    mock_instance.chat.completions.create.return_value = m
    
    client = OpenAIChat(api_key="test-key")
    response = client.generate(messages=[{"role": "user", "content": "Hello"}])
    
    assert response.success is True
    assert response.content == "Atomic Response"
    # Atomic call should NOT affect internal messages
    assert len(client.messages) == 0

def test_openai_chat_ask_success(mock_openai_chat):
    """Tests high-level ask method with a simple response."""
    mock_instance = mock_openai_chat.return_value
    m = MagicMock()
    m.choices = [MagicMock(message=MagicMock(content="Hello", tool_calls=None, reasoning_content=None))]
    m.usage = None
    mock_instance.chat.completions.create.return_value = m
    
    client = OpenAIChat(api_key="test-key")
    response = client.ask("Hi", stream=False)
    
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert response.content == "Hello"
    assert len(client.messages) == 2

def test_openai_chat_ask_stream_success(mock_openai_chat):
    """Tests high-level ask method with streaming output."""
    mock_instance = mock_openai_chat.return_value
    
    # Mock streaming chunks
    c1 = MagicMock()
    # Explicitly set reasoning_content to avoid Mock truthiness
    c1.choices = [MagicMock(delta=MagicMock(content="Hello", tool_calls=None, reasoning_content=None))]
    c1.usage = None
    
    c2 = MagicMock()
    c2.choices = []
    c2.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    
    mock_instance.chat.completions.create.return_value = [c1, c2]
    
    client = OpenAIChat(api_key="test-key")
    response_gen = client.ask("Hi", stream=True)
    
    chunks = list(response_gen)
    # Expected: 1. Chunk with "Hello", 2. Final Chunk with usage
    assert len(chunks) == 2
    assert chunks[0].content == "Hello"
    assert chunks[1].content == ""
    assert chunks[1].usage.total_tokens == 15
    assert len(client.messages) == 2

def test_openai_chat_multi_turn_tool(mock_openai_chat):
    """Tests recursive tool calling in ask loop."""
    def tool_a(x: int): return True, f"Done {x}"
    tool_def = FunctionTool(
        name="tool_a",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        mapped_callable=tool_a
    )

    client = OpenAIChat(api_key="test-key", tools=[tool_def])
    mock_instance = mock_openai_chat.return_value
    
    # Round 1: Model calls tool
    m1 = MagicMock()
    tc = MagicMock()
    tc.id = "c1"
    tc.type = "function"
    tc.function.name = "tool_a"
    tc.function.arguments = '{"x": 1}'
    tc.model_dump.return_value = {"id": "c1", "type": "function", "function": {"name": "tool_a", "arguments": '{"x": 1}'}}
    m1.raw_response = m1
    m1.choices = [MagicMock(message=MagicMock(content=None, tool_calls=[tc], reasoning_content=None))]
    m1.usage = None
    
    # Round 2: Model gives final answer
    m2 = MagicMock()
    m2.raw_response = m2
    m2.choices = [MagicMock(message=MagicMock(content="Finished", tool_calls=None, reasoning_content=None))]
    m2.usage = None
    
    mock_instance.chat.completions.create.side_effect = [m1, m2]
    
    response = client.ask("Run tool", stream=False)
    assert response.content == "Finished"
    assert mock_instance.chat.completions.create.call_count == 2
    assert client.latest_tool_call_result["tool_a"] is True

def test_openai_chat_safe_state_on_error(mock_openai_chat):
    """Verifies that conversation history is preserved on API failure."""
    mock_instance = mock_openai_chat.return_value
    mock_instance.chat.completions.create.side_effect = Exception("API Down")
    
    client = OpenAIChat(api_key="test-key", instructions="System")
    initial_len = len(client.messages)
    
    response = client.ask("Fail me", stream=False)
    assert response.success is False
    assert len(client.messages) == initial_len
