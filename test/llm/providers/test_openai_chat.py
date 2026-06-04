#! python3
# -*- coding: utf-8 -*-
from unittest.mock import MagicMock, patch, ANY
import pytest
from PIL import Image
from ark.llm.providers.openai_chat import OpenAIChat

@pytest.fixture
def mock_openai_client():
    with patch("ark.llm.providers.openai_chat.OpenAI") as mock:
        yield mock

def test_openai_chat_init(mock_openai_client):
    client = OpenAIChat(
        api_key="test-key", 
        base_url="https://test.api", 
        default_model="test-model",
        instructions="You are a helpful assistant"
    )
    mock_openai_client.assert_called_once_with(api_key="test-key", base_url="https://test.api")
    assert client.default_model == "test-model"
    assert len(client.messages) == 1
    assert client.messages[0]["role"] == "system"

def test_openai_chat_ask_simple(mock_openai_client):
    mock_instance = mock_openai_client.return_value
    # Mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock(delta=MagicMock(content="Hello", tool_calls=None))]
    mock_instance.chat.completions.create.return_value = [mock_chunk]
    
    client = OpenAIChat(api_key="test-key")
    success, status, content = client.ask("Hi")
    
    assert success is True
    assert status == 200
    assert content == "Hello"
    assert len(client.messages) == 2 # User and Assistant

def test_openai_chat_multimodal_content():
    prompt = "What is this?"
    mock_img = MagicMock(spec=Image.Image)
    
    with patch("ark.llm.providers.openai_chat.convert_image_to_data_url", return_value="data:image/png;base64,xxx"):
        content = OpenAIChat.build_user_message_content(prompt, [mock_img])
    
    assert isinstance(content, list)
    assert content[0]["text"] == prompt
    assert content[1]["image_url"]["url"] == "data:image/png;base64,xxx"

def test_openai_chat_tool_execution(mock_openai_client):
    # Mock a tool function
    def my_tool(x):
        return True, f"Result is {x}"
    
    tool_def = {
        "type": "function",
        "function": {
            "name": "my_tool",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"]
            }
        }
    }
    
    client = OpenAIChat(
        api_key="test-key",
        tools=[tool_def],
        tool_function_mapper={"my_tool": my_tool}
    )
    
    mock_instance = mock_openai_client.return_value
    
    # Mock first response with tool call
    mock_tool_call = MagicMock()
    mock_tool_call.index = 0
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "my_tool"
    mock_tool_call.function.arguments = '{"x": 10}'
    
    mock_chunk_1 = MagicMock()
    mock_chunk_1.choices = [MagicMock(delta=MagicMock(content=None, tool_calls=[mock_tool_call]))]
    
    # Mock second response (after tool)
    mock_chunk_2 = MagicMock()
    mock_chunk_2.choices = [MagicMock(delta=MagicMock(content="The result is 10", tool_calls=None))]
    
    mock_instance.chat.completions.create.side_effect = [[mock_chunk_1], [mock_chunk_2]]
    
    success, status, content = client.ask("Use tool")
    
    assert success is True
    assert content == "The result is 10"
    assert client.latest_tool_call_result["my_tool"] is True
