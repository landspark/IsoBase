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
from anthropic.types import (
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
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


def _start_event(content=None, input_tokens=10):
    """Builds a real message_start event.

    ``content`` defaults to None to mirror the Anthropic-*compatible* gateways
    that send ``message.content: null`` — the framing that crashed the SDK's
    stream accumulator. We feed real event models (not mocks) so the test
    exercises the actual normalization + ``accumulate_event`` path.
    """
    message = Message.construct(
        id="m1", type="message", role="assistant", model="x",
        content=content, stop_reason=None, stop_sequence=None,
        usage={"input_tokens": input_tokens, "output_tokens": 0})
    return RawMessageStartEvent.construct(type="message_start", message=message)


def _block_start_event(index, content_block):
    """Builds a real content_block_start event for the given block dict."""
    return RawContentBlockStartEvent.construct(
        type="content_block_start", index=index, content_block=content_block)


def _delta_event(index, delta):
    """Builds a real content_block_delta event from a delta dict."""
    return RawContentBlockDeltaEvent.construct(
        type="content_block_delta", index=index, delta=delta)


def _block_stop_event(index):
    """Builds a real content_block_stop event."""
    return RawContentBlockStopEvent.construct(
        type="content_block_stop", index=index)


def _message_delta_event(output_tokens):
    """Builds a real message_delta event carrying final output token usage."""
    return RawMessageDeltaEvent.construct(
        type="message_delta",
        delta={"stop_reason": "end_turn", "stop_sequence": None},
        usage={"output_tokens": output_tokens})


def _stop_event():
    """Builds a real message_stop event."""
    return RawMessageStopEvent.construct(type="message_stop")


def _set_stream(mock_instance, events):
    """Wires client.messages.create(stream=True) to yield raw events."""
    mock_instance.messages.create.return_value = iter(events)

from ark.llm.entities import ToolCall

def test_execute_tool_calls_raw_outputs():
    """Tool execution returns raw outputs now."""
    def tool_a(x):
        return True, f"got {x}"

    tool_def = FunctionTool(
        name="tool_a",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        mapped_callable=tool_a
    )
    ts = ToolSet([tool_def])
    calls = [ToolCall(id="tu1", name="tool_a", arguments='{"x": 5}')]

    # ToolSet now only returns raw tool outputs.
    outputs, results = ts.execute_tool_calls(calls)

    assert results["tool_a"] is True
    assert outputs[0] == "got 5"


# --- Client behavior --------------------------------------------------------

def test_init_system_not_in_messages(mock_anthropic):
    """System instructions stay top-level, never in the message list."""
    client = AnthropicMessages(api_key="test-key", instructions="System rules")
    assert client.instructions == "System rules"
    assert len(client.messages) == 0


def test_atomic_completion(mock_anthropic):
    """The atomic generate method parses text and does not touch history."""
    mock_instance = mock_anthropic.return_value
    mock_instance.messages.create.return_value = _message(
        [_text_block("Atomic Response")], input_tokens=3, output_tokens=2)

    client = AnthropicMessages(api_key="test-key")
    response = client.generate(messages=[{"role": "user", "content": "Hi"}])

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
    """Streaming ask yields incremental text from raw events.

    The message_start event carries ``content=None`` on purpose: this is the
    compatible-gateway framing that crashed the SDK's high-level accumulator,
    so the test pins that we now consume raw events and survive it.
    """
    mock_instance = mock_anthropic.return_value
    events = [
        _start_event(content=None),
        _block_start_event(0, {"type": "text", "text": ""}),
        _delta_event(0, {"type": "text_delta", "text": "Hel"}),
        _delta_event(0, {"type": "text_delta", "text": "lo"}),
        _block_stop_event(0),
        _message_delta_event(output_tokens=5),
        _stop_event(),
    ]
    _set_stream(mock_instance, events)

    client = AnthropicMessages(api_key="test-key")
    chunks = list(client.ask("Hi", stream=True))

    text_chunks = [c for c in chunks if c.content]
    assert "".join(c.content for c in text_chunks) == "Hello"
    assert len(client.messages) == 2


def test_ask_stream_tool_call(mock_anthropic):
    """Streaming tool_use is reconstructed from raw events and executed.

    Verifies tool_use input is rebuilt from input_json_delta fragments (the
    raw-event path) and that the tool round runs to a final streamed answer.
    """
    def tool_a(x):
        return True, f"Done {x}"

    tool_def = FunctionTool(
        name="tool_a",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        mapped_callable=tool_a
    )
    client = AnthropicMessages(api_key="test-key", tools=[tool_def])
    mock_instance = mock_anthropic.return_value

    # Round 1: a tool_use block whose input streams in as partial JSON; the SDK
    # accumulator reassembles it (we don't hand-parse partial_json ourselves).
    round1 = [
        _start_event(content=None),
        _block_start_event(0, {"type": "tool_use", "id": "tu1",
                               "name": "tool_a", "input": {}}),
        _delta_event(0, {"type": "input_json_delta", "partial_json": '{"x":'}),
        _delta_event(0, {"type": "input_json_delta", "partial_json": ' 1}'}),
        _block_stop_event(0),
        _message_delta_event(output_tokens=2),
        _stop_event(),
    ]
    # Round 2: final text answer.
    round2 = [
        _start_event(content=None),
        _block_start_event(0, {"type": "text", "text": ""}),
        _delta_event(0, {"type": "text_delta", "text": "Done"}),
        _block_stop_event(0),
        _message_delta_event(output_tokens=2),
        _stop_event(),
    ]
    mock_instance.messages.create.side_effect = [iter(round1), iter(round2)]

    chunks = list(client.ask("Run tool", stream=True))

    assert "".join(c.content for c in chunks if c.content) == "Done"
    assert client.latest_tool_call_result["tool_a"] is True
    assert mock_instance.messages.create.call_count == 2


def test_multi_turn_tool(mock_anthropic):
    """Recursive tool calling: tool_use round then a final answer."""
    def tool_a(x):
        return True, f"Done {x}"

    tool_def = FunctionTool(
        name="tool_a",
        parameters_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        mapped_callable=tool_a
    )
    client = AnthropicMessages(api_key="test-key", tools=[tool_def])
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
