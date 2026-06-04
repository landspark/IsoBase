#! python3
# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from ark.llm.providers.openai_chat import OpenAIChat
from ark.llm import LLMResponse

@pytest.fixture
def mock_openai_chat():
    with patch("ark.llm.providers.openai_chat.OpenAI") as mock:
        yield mock

def test_openai_chat_init(mock_openai_chat):
    client = OpenAIChat(api_key="test-key", instructions="System rules")
    assert len(client.messages) == 1
    assert client.messages[0]["content"] == "System rules"

def test_openai_chat_ask_success(mock_openai_chat):
    mock_instance = mock_openai_chat.return_value
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="Hello", tool_calls=None))]
    mock_chunk.usage = None
    mock_instance.chat.completions.create.return_value = [mock_chunk]
    
    client = OpenAIChat(api_key="test-key")
    response = client.ask("Hi")
    
    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert response.content == "Hello"
    assert len(client.messages) == 2

def test_openai_chat_multi_turn_tool(mock_openai_chat):
    def tool_a(x: int): return True, f"Done {x}"
    tool_def = {
        "type": "function",
        "function": {"name": "tool_a", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}
    }
    
    client = OpenAIChat(api_key="test-key", tools=[tool_def], tool_function_mapper={"tool_a": tool_a})
    mock_instance = mock_openai_chat.return_value
    
    # Round 1: Model calls tool
    tc = MagicMock()
    tc.index = 0
    tc.id = "c1"
    tc.function.name = "tool_a"
    tc.function.arguments = '{"x": 1}'
    m1 = MagicMock(choices=[MagicMock(delta=MagicMock(content=None, tool_calls=[tc]))], usage=None)
    
    # Round 2: Model gives final answer
    m2 = MagicMock(choices=[MagicMock(delta=MagicMock(content="Finished", tool_calls=None))], usage=None)
    
    mock_instance.chat.completions.create.side_effect = [[m1], [m2]]
    
    response = client.ask("Run tool")
    assert response.content == "Finished"
    assert mock_instance.chat.completions.create.call_count == 2
    assert client.latest_tool_call_result["tool_a"] is True

def test_openai_chat_safe_state_on_error(mock_openai_chat):
    mock_instance = mock_openai_chat.return_value
    mock_instance.chat.completions.create.side_effect = Exception("API Down")
    
    client = OpenAIChat(api_key="test-key", instructions="System")
    initial_len = len(client.messages)
    
    response = client.ask("Fail me")
    assert response.success is False
    assert len(client.messages) == initial_len # Should NOT have user/error messages added
