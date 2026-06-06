#! python3
# -*- encoding: utf-8 -*-
"""Anthropic Messages API client with structured responses and tool support.

``AnthropicMessages`` is the concrete provider for the Anthropic Messages API
(``/v1/messages``) and any Anthropic-compatible gateway (via ``base_url``). It
mirrors the public surface of :class:`~ark.llm.providers.openai_chat.OpenAIChat`
(``chat_completion`` / ``chat_completion_stream`` / ``ask`` /
``build_user_message_content``) and returns the standardized ``LLMResponse``,
so the two providers are interchangeable.

The class is named after the API it targets (the *Messages* API), matching the
convention behind ``OpenAIChat`` (the OpenAI *Chat Completion* API). Tool
calling, and the room left for RAG / MCP, route through the same ``ToolSet`` /
``LLMResponse`` entry points the OpenAI client uses.

Key Anthropic-specific handling versus OpenAI:
    - ``system`` is a top-level string, not a message role; ``self.messages``
      never contains a system entry.
    - ``max_tokens`` is mandatory.
    - Tool results are ``tool_result`` blocks inside a ``user`` message.
    - Tool schemas are flat ``{name, description, input_schema}``.
    - Images are ``{"type": "image", "source": {"type": "base64", ...}}``.
    - Extended thinking is opt-in via ``thinking`` (e.g. ``{"type": "adaptive"}``);
      on a tool round the assistant turn echoes back the raw content blocks so
      thinking signatures are preserved.

@File   :   anthropic_messages.py
@Created:   2026/06/06 21:13
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from inspect import signature
from json import dumps
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from anthropic import Anthropic, BadRequestError
from PIL import Image as PILImage
from ark.core.image_service import convert_image_to_base64
from ark.core.logger import LOGGER
from ark.llm.entities import LLMResponse, TokenUsage
from ark.llm.providers.base import BaseLLMClient
from ark.llm.tools import FunctionTool, ToolSet


class AnthropicMessages(BaseLLMClient):
    """Client for the Anthropic Messages API and compatible endpoints.

    Supports structured ``LLMResponse``, safe history mutation, multi-turn tool
    calling, multimodal image input, streaming, and extended thinking.

    Attributes:
        client: The underlying Anthropic client instance.
        default_model: The default model ID to use for requests.
        instructions: System instructions defining the agent's behavior (sent as
            the top-level ``system`` parameter, never as a message).
        conversation_mode: Whether to maintain a message history.
        max_tool_rounds: Maximum number of recursive tool-calling rounds.
        max_tokens: The default ``max_tokens`` for requests (required by the API).
        thinking: Optional extended-thinking config (e.g. ``{"type": "adaptive"}``).
        messages: The current conversation history (never contains a system entry).
        tool_set: A collection of available tools.
        latest_tool_call_result: Results of the last tool executions.
        anthropic_args_dict: Filtered kwargs for the Anthropic SDK.
    """

    def __init__(self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "claude-opus-4-8",
        instructions: str = "",
        messages: List[Dict[str, Any]] = None,
        tools: List[Dict[str, Any]] = None,
        tool_function_mapper: Dict[str, Callable] = None,
        tool_function_callable_kwargs: Dict[str, Any] = None,
        conversation_mode: bool = True,
        max_tool_rounds: int = 5,
        max_tokens: int = 4096,
        thinking: Optional[Dict[str, Any]] = None,
        **kwargs: Any):
        """Initializes an AnthropicMessages client.

        Args:
            api_key: The API key for authentication.
            base_url: The base URL for the API endpoint (for compatible gateways).
            default_model: The default model ID.
            instructions: System instructions for the model (top-level ``system``).
            messages: Initial list of conversation messages (no system entry).
            tools: A list of tool definitions in OpenAI format (converted to
                Anthropic format on the wire), keeping parity with OpenAIChat.
            tool_function_mapper: Map of tool names to Python callables.
            tool_function_callable_kwargs: Static kwargs for tool callables.
            conversation_mode: If True, history is preserved across asks.
            max_tool_rounds: Limits recursive tool calls per ask.
            max_tokens: Default maximum output tokens (required by the API).
            thinking: Optional extended-thinking config; omitted when None.
            **kwargs: Additional arguments for the Anthropic API.
        """
        self.client = Anthropic(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.instructions = instructions
        self.conversation_mode = conversation_mode
        self.max_tool_rounds = max_tool_rounds
        self.max_tokens = max_tokens
        self.thinking = thinking

        # System lives in the top-level `system` param, so it is never stored
        # as a message entry (unlike OpenAIChat).
        self.messages = messages or []

        # Tools are supplied in OpenAI format for parity with OpenAIChat, then
        # rendered to Anthropic's flat schema when calling the API.
        self.tool_set = ToolSet([
            FunctionTool.from_openai_schema(
                t, tool_function_mapper, tool_function_callable_kwargs)
            for t in (tools or []) if t.get("type") == "function"
        ])
        self.latest_tool_call_result = {}

        create_params = signature(self.client.messages.create).parameters
        self.anthropic_args_dict = {
            k: v for k, v in kwargs.items() if k in create_params
        }

        LOGGER.info(f"AnthropicMessages initialized (model: {default_model})")

    @classmethod
    def build_user_message_content(
        cls, prompt: str,
        images: Optional[List[PILImage.Image]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Builds user message content in Anthropic format.

        Overrides the OpenAI-shaped base implementation to emit Anthropic
        content blocks: a ``text`` block plus base64 ``image`` blocks.

        Args:
            prompt: The text prompt provided by the user.
            images: An optional list of PIL Image objects to attach.

        Returns:
            A string if only text is provided, or a list of Anthropic content
            blocks for multimodal requests.

        Raises:
            TypeError: If any item in the images list is not a PIL Image.
        """
        if not images:
            return prompt

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if not isinstance(img, PILImage.Image):
                raise TypeError("Each image must be a PIL.Image.Image instance.")
            media_type, image_base64 = convert_image_to_base64(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            })
        return content

    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: bool = False,
            **kwargs: Any) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """Orchestrates a chat interaction, handling history and tool calls.

        Args:
            prompt (str): The user's input text.
            images (Optional[List[PILImage.Image]]): Optional images for multimodal input.
            stream (bool, optional): Whether to use streaming. Defaults to False.
            **kwargs: Additional parameters for the model.

        Returns:
            An LLMResponse (non-stream) or Iterator[LLMResponse] (stream).
        """
        if stream:
            return self.__ask_loop_stream(prompt, images, **kwargs)
        else:
            return self.__ask_loop(prompt, images, **kwargs)

    def chat_completion(self,
            messages: List[Dict[str, Any]],
            model: Optional[str] = None,
            **kwargs: Any
        ) -> LLMResponse:
        """Sends a single non-streaming Messages request.

        Args:
            messages: Conversation context (Anthropic message format).
            model: Specific model ID to use.
            **kwargs: Extra parameters for the API.

        Returns:
            An LLMResponse containing the model's output.
        """
        try:
            request_args = self.__build_request_args(model, **kwargs)
            message = self.client.messages.create(messages=messages,
                                                  **request_args)
            content, tool_calls, reasoning = self.__parse_content(
                message.content)
            usage = self.__extract_usage(message.usage)
            return LLMResponse(
                success=True, status_code=200,
                content=content, usage=usage,
                tool_calls=tool_calls, reasoning_content=reasoning,
                raw_response=message
            )
        except Exception as e:
            return LLMResponse(success=False, status_code=500, content=f"Error: {e}")

    def chat_completion_stream(
            self, messages: List[Dict[str, Any]],
            model: Optional[str] = None, **kwargs: Any
        ) -> Iterator[LLMResponse]:
        """Sends a streaming Messages request.

        Args:
            messages: Conversation context (Anthropic message format).
            model: Specific model ID to use.
            **kwargs: Extra parameters for the API.

        Yields:
            LLMResponse objects containing incremental chunks, then a final
            summary carrying aggregated tool calls and usage.
        """
        request_args = self.__build_request_args(model, **kwargs)

        with self.client.messages.stream(messages=messages,
                                         **request_args) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue

                delta = event.delta
                delta_type = getattr(delta, "type", "")
                if delta_type == "text_delta":
                    yield LLMResponse(
                        success=True, status_code=200,
                        content=delta.text, raw_response=event)
                elif delta_type == "thinking_delta":
                    yield LLMResponse(
                        success=True, status_code=200,
                        content="", reasoning_content=delta.thinking,
                        raw_response=event)

            # Aggregate the full message for tool logic and usage.
            final_message = stream.get_final_message()

        _, tool_calls, _ = self.__parse_content(final_message.content)
        usage = self.__extract_usage(final_message.usage)

        # Final yield with empty content to avoid duplicating streamed text,
        # carrying the aggregated tool calls, usage, and raw message.
        yield LLMResponse(
            success=True, status_code=200,
            content="", tool_calls=tool_calls,
            reasoning_content="", usage=usage,
            raw_response=final_message
        )

    def __ask_loop(self,
                  prompt: str,
                  images: Optional[List[PILImage.Image]] = None,
                  **kwargs: Any) -> LLMResponse:
        """Internal loop for non-streaming interaction.

        Args:
            prompt: User prompt.
            images: Multimodal inputs.
            **kwargs: API arguments.

        Returns:
            Final LLMResponse after all tool calls.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            current_messages = self.__prepare_messages(user_content)

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                response = self.chat_completion(messages=current_messages,
                                               **kwargs)
                if not response.success:
                    return response

                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens
                total_usage.total_tokens += response.usage.total_tokens
                final_reasoning += response.reasoning_content

                if not response.tool_calls:
                    final_content = response.content
                    break

                # Echo the assistant turn back verbatim (raw content blocks
                # preserve thinking signatures and tool_use blocks).
                current_messages.append({
                    "role": "assistant",
                    "content": response.raw_response.content
                })

                tool_blocks, tool_results = (
                    self.tool_set.execute_tool_calls_anthropic(
                        response.tool_calls))
                current_messages.append({
                    "role": "user",
                    "content": tool_blocks
                })
                self.latest_tool_call_result.update(tool_results)

            if final_content:
                current_messages.append({
                    "role": "assistant",
                    "content": final_content
                })
            if self.conversation_mode:
                self.messages = current_messages

            return LLMResponse(
                success=True, status_code=200,
                content=final_content, usage=total_usage,
                reasoning_content=final_reasoning
            )

        except Exception as e:
            return self.__handle_ask_exception(e)

    def __ask_loop_stream(self,
                         prompt: str,
                         images: Optional[List[PILImage.Image]] = None,
                         **kwargs: Any) -> Iterator[LLMResponse]:
        """Internal loop for streaming interaction.

        Args:
            prompt: User prompt.
            images: Multimodal inputs.
            **kwargs: API arguments.

        Yields:
            Incremental chunks and a final summary per round.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            current_messages = self.__prepare_messages(user_content)

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                tool_calls = []
                current_round_content = ""
                current_round_reasoning = ""
                final_message = None

                for chunk_resp in self.chat_completion_stream(
                        messages=current_messages, **kwargs):

                    if chunk_resp.tool_calls:
                        tool_calls = chunk_resp.tool_calls
                    if chunk_resp.usage.total_tokens > 0:
                        total_usage.input_tokens += chunk_resp.usage.input_tokens
                        total_usage.output_tokens += chunk_resp.usage.output_tokens
                        total_usage.total_tokens += chunk_resp.usage.total_tokens
                    if chunk_resp.raw_response is not None and (
                            chunk_resp.usage.total_tokens > 0):
                        # The final summary chunk carries the aggregated message.
                        final_message = chunk_resp.raw_response

                    if chunk_resp.content or chunk_resp.reasoning_content:
                        current_round_content += chunk_resp.content
                        current_round_reasoning += chunk_resp.reasoning_content
                        yield chunk_resp

                final_reasoning += current_round_reasoning

                if not tool_calls:
                    final_content = current_round_content
                    break

                # Echo the assistant turn back verbatim from the aggregated
                # message (preserves thinking signatures and tool_use blocks).
                current_messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })

                tool_blocks, tool_results = (
                    self.tool_set.execute_tool_calls_anthropic(tool_calls))
                current_messages.append({
                    "role": "user",
                    "content": tool_blocks
                })
                self.latest_tool_call_result.update(tool_results)

            if final_content:
                current_messages.append({
                    "role": "assistant",
                    "content": final_content
                })
            if self.conversation_mode:
                self.messages = current_messages

        except Exception as e:
            yield self.__handle_ask_exception(e)

    def __build_request_args(self, model: Optional[str],
                             **kwargs: Any) -> Dict[str, Any]:
        """Assembles the keyword arguments for a Messages API call.

        Args:
            model: Specific model ID, or None to use the default.
            **kwargs: Per-call overrides merged over the client defaults.

        Returns:
            A dict of API arguments (model, max_tokens, optional system/tools/
            thinking, plus filtered defaults and overrides).
        """
        args: Dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": self.max_tokens,
        }
        if self.instructions:
            args["system"] = self.instructions
        if self.tool_set:
            args["tools"] = self.tool_set.get_anthropic_schema()
        if self.thinking:
            args["thinking"] = self.thinking
        args.update(self.anthropic_args_dict)
        args.update(kwargs)
        return args

    def __prepare_messages(
            self, user_content: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Helper to prepare the message list for ask loops.

        Unlike OpenAIChat, the system prompt is never injected here; it is
        passed via the top-level ``system`` API parameter instead.
        """
        messages = self.messages.copy() if self.conversation_mode else []
        messages.append({"role": "user", "content": user_content})
        return messages

    def __parse_content(
            self, content_blocks: List[Any]
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        """Parses Anthropic response content blocks.

        Args:
            content_blocks: The ``content`` list from a Message response.

        Returns:
            A tuple of ``(text, tool_calls, reasoning)``, where ``tool_calls``
            uses the standardized OpenAI shape so the shared tool machinery and
            ``LLMResponse`` consumers work unchanged across providers.
        """
        text = ""
        reasoning = ""
        tool_calls = []
        for block in (content_blocks or []):
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text += block.text
            elif block_type == "thinking":
                reasoning += getattr(block, "thinking", "") or ""
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": dumps(block.input or {}),
                    },
                })
        return text, tool_calls, reasoning

    def __extract_usage(self, usage: Any) -> TokenUsage:
        """Extracts TokenUsage from an Anthropic usage object.

        Args:
            usage: The usage object on a Message (or None).

        Returns:
            A populated TokenUsage instance. ``total_tokens`` is computed as
            input + output since Anthropic does not return a combined field.
        """
        if not usage:
            return TokenUsage()
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

    def __handle_ask_exception(self, e: Exception) -> LLMResponse:
        """Helper to handle exceptions in ask loops."""
        LOGGER.error(f"ask failed: {e}")
        status = 400 if isinstance(e, BadRequestError) else 500
        return LLMResponse(success=False, status_code=status, content=f"Unexpected error: {e}")
