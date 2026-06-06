#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_anthropic_messages.py
@Created:   2026/06/06 21:15
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from unittest.mock import MagicMock, patch
import pytest
from PIL import Image
from ark.llm.providers.anthropic_messages import AnthropicMessages
from ark.llm import LLMResponse
from ark.llm.tools import FunctionTool, ToolSet


@pytest.fixture
def mock_anthropic():
    """Fixture to mock the Anthropic client."""
    with patch("ark.llm.providers.anthropic_messages.Anthropic") as mock:
        yield mock


def _text_block(text):
    """Builds a mock Anthropic text content block."""
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _thinking_block(thinking):
    """Builds a mock Anthropic thinking content block."""
    b = MagicMock()
    b.type = "thinking"
    b.thinking = thinking
    return b


def _tool_use_block(block_id, name, tool_input):
    """Builds a mock Anthropic tool_use content block."""
    b = MagicMock()
    b.type = "tool_use"
    b.id = block_id
    b.name = name
    b.input = tool_input
    return b


def _message(content, input_tokens=0, output_tokens=0):
    """Builds a mock Anthropic Message response."""
    m = MagicMock()
    m.content = content
    if input_tokens or output_tokens:
        m.usage = MagicMock(input_tokens=input_tokens,
                            output_tokens=output_tokens)
    else:
        m.usage = None
    return m


def _delta_event(delta_type, **fields):
    """Builds a mock content_block_delta stream event."""
    e = MagicMock()
    e.type = "content_block_delta"
    e.delta = MagicMock(type=delta_type, **fields)
    return e


def _set_stream(mock_instance, events, final_message):
    """Wires client.messages.stream() to a context manager mock."""
    stream_obj = MagicMock()
    stream_obj.__iter__.return_value = iter(events)
    stream_obj.get_final_message.return_value = final_message

    cm = MagicMock()
    cm.__enter__.return_value = stream_obj
    cm.__exit__.return_value = False
    mock_instance.messages.stream.return_value = cm


# --- Schema conversion (provider-neutral tool format) -----------------------

def test_function_tool_to_anthropic_schema():
    """OpenAI-shaped tools render to Anthropic's flat schema."""
    openai_def = {
        "type": "function",
        "function": {
            "name": "tool_a",
            "description": "does a",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        },
    }
    tool = FunctionTool.from_openai_schema(openai_def)
    schema = tool.to_anthropic_schema()
    assert schema["name"] == "tool_a"
    assert schema["description"] == "does a"
    assert schema["input_schema"]["type"] == "object"
    assert schema["input_schema"]["properties"]["x"]["type"] == "integer"
    assert schema["input_schema"]["required"] == ["x"]


def test_anthropic_to_openai_roundtrip():
    """Anthropic schema ingests and renders back to OpenAI shape."""
    anthropic_def = {
        "name": "tool_b",
        "description": "does b",
        "input_schema": {
            "type": "object",
            "properties": {"y": {"type": "string"}},
            "required": ["y"],
        },
    }
    tool = FunctionTool.from_anthropic_schema(anthropic_def)
    openai_schema = tool.to_openai_schema()
    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["name"] == "tool_b"
    assert openai_schema["function"]["parameters"]["properties"]["y"]["type"] == "string"


def test_toolset_get_anthropic_schema():
    """ToolSet exposes the Anthropic schema list."""
    openai_def = {
        "type": "function",
        "function": {"name": "t", "parameters": {"type": "object", "properties": {}}},
    }
    ts = ToolSet([FunctionTool.from_openai_schema(openai_def)])
    schemas = ts.get_anthropic_schema()
    assert len(schemas) == 1
    assert schemas[0]["name"] == "t"
    assert "input_schema" in schemas[0]


def test_execute_tool_calls_anthropic_blocks():
    """Anthropic tool execution returns tool_result blocks."""
    def tool_a(x):
        return True, f"got {x}"

    openai_def = {
        "type": "function",
        "function": {"name": "tool_a", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    }
    ts = ToolSet([FunctionTool.from_openai_schema(openai_def, {"tool_a": tool_a})])
    calls = [{"id": "tu1", "type": "function",
              "function": {"name": "tool_a", "arguments": '{"x": 5}'}}]
    blocks, results = ts.execute_tool_calls_anthropic(calls)

    assert results["tool_a"] is True
    assert blocks[0]["type"] == "tool_result"
    assert blocks[0]["tool_use_id"] == "tu1"
    assert blocks[0]["content"] == "got 5"
    assert "is_error" not in blocks[0]


# --- Client behavior --------------------------------------------------------

def test_init_system_not_in_messages(mock_anthropic):
    """System instructions stay top-level, never in the message list."""
    client = AnthropicMessages(api_key="test-key", instructions="System rules")
    assert client.instructions == "System rules"
    assert len(client.messages) == 0


def test_atomic_completion(mock_anthropic):
    """The atomic chat_completion method parses text and does not touch history."""
    mock_instance = mock_anthropic.return_value
    mock_instance.messages.create.return_value = _message(
        [_text_block("Atomic Response")], input_tokens=3, output_tokens=2)

    client = AnthropicMessages(api_key="test-key")
    response = client.chat_completion(messages=[{"role": "user", "content": "Hi"}])

    assert response.success is True
    assert response.content == "Atomic Response"
    assert response.usage.total_tokens == 5
    assert len(client.messages) == 0


def test_ask_success(mock_anthropic):
    """High-level ask with a simple text response."""
    mock_instance = mock_anthropic.return_value
    mock_instance.messages.create.return_value = _message(
        [_text_block("Hello")], input_tokens=4, output_tokens=1)

    client = AnthropicMessages(api_key="test-key")
    response = client.ask("Hi", stream=False)

    assert isinstance(response, LLMResponse)
    assert response.success is True
    assert response.content == "Hello"
    assert len(client.messages) == 2


def test_ask_thinking_into_reasoning(mock_anthropic):
    """Thinking blocks surface into reasoning_content."""
    mock_instance = mock_anthropic.return_value
    mock_instance.messages.create.return_value = _message(
        [_thinking_block("step by step"), _text_block("Answer")],
        input_tokens=5, output_tokens=3)

    client = AnthropicMessages(api_key="test-key", thinking={"type": "adaptive"})
    response = client.ask("Solve", stream=False)

    assert response.content == "Answer"
    assert response.reasoning_content == "step by step"
    # thinking config is forwarded to the API
    _, call_kwargs = mock_instance.messages.create.call_args
    assert call_kwargs["thinking"] == {"type": "adaptive"}


def test_ask_stream_success(mock_anthropic):
    """Streaming ask yields incremental text then a final usage summary."""
    mock_instance = mock_anthropic.return_value
    events = [
        _delta_event("text_delta", text="Hel"),
        _delta_event("text_delta", text="lo"),
    ]
    final = _message([_text_block("Hello")], input_tokens=10, output_tokens=5)
    _set_stream(mock_instance, events, final)

    client = AnthropicMessages(api_key="test-key")
    chunks = list(client.ask("Hi", stream=True))

    text_chunks = [c for c in chunks if c.content]
    assert "".join(c.content for c in text_chunks) == "Hello"
    assert len(client.messages) == 2


def test_multi_turn_tool(mock_anthropic):
    """Recursive tool calling: tool_use round then a final answer."""
    def tool_a(x):
        return True, f"Done {x}"

    tool_def = {
        "type": "function",
        "function": {"name": "tool_a", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}},
    }
    client = AnthropicMessages(api_key="test-key", tools=[tool_def],
                               tool_function_mapper={"tool_a": tool_a})
    mock_instance = mock_anthropic.return_value

    # Round 1: model calls the tool. Round 2: final answer.
    m1 = _message([_tool_use_block("tu1", "tool_a", {"x": 1})],
                  input_tokens=5, output_tokens=2)
    m2 = _message([_text_block("Finished")], input_tokens=6, output_tokens=2)
    mock_instance.messages.create.side_effect = [m1, m2]

    response = client.ask("Run tool", stream=False)

    assert response.content == "Finished"
    assert mock_instance.messages.create.call_count == 2
    assert client.latest_tool_call_result["tool_a"] is True


def test_safe_state_on_error(mock_anthropic):
    """Conversation history is preserved when the API fails."""
    mock_instance = mock_anthropic.return_value
    mock_instance.messages.create.side_effect = Exception("API Down")

    client = AnthropicMessages(api_key="test-key", instructions="System")
    initial_len = len(client.messages)

    response = client.ask("Fail me", stream=False)
    assert response.success is False
    assert len(client.messages) == initial_len


def test_build_user_message_content_with_image(mock_anthropic):
    """Multimodal content uses Anthropic base64 image blocks."""
    img = Image.new("RGB", (2, 2), color="red")
    content = AnthropicMessages.build_user_message_content("describe", [img])

    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["type"] == "base64"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"]
