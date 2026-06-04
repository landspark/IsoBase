#! python3
# -*- encoding: utf-8 -*-
"""OpenAI Chat Completion client with structured responses and multi-turn tool support.

@File   :   openai_chat.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openai import (
    BadRequestError,
    OpenAI,
)
from PIL import Image as PILImage

from ark.core.logger import LOGGER
from ark.llm.entities import LLMResponse, TokenUsage
from ark.llm.providers.base import BaseLLMClient
from ark.llm.tools import FunctionTool, ToolSet


class OpenAIChat(BaseLLMClient):
    """Optimized client for OpenAI Chat Completion compatible APIs.

    Supports structured LLMResponse, safe history mutation, and multi-turn
    tool calling.

    Attributes:
        client: The underlying OpenAI client instance.
        default_model: The default model ID to use for requests.
        instructions: System instructions defining the agent's behavior.
        conversation_mode: Whether to maintain a message history.
        max_tool_rounds: Maximum number of recursive tool-calling rounds.
        messages: The current conversation history.
        tool_set: A collection of available tools.
        latest_tool_call_result: results of the last tool executions.
        openai_args_dict: Filtered kwargs for the OpenAI SDK.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 default_model: str = "gpt-3.5-turbo",
                 instructions: str = "",
                 messages: List[Dict[str, str]] = None,
                 tools: List[Dict[str, Any]] = None,
                 tool_function_mapper: Dict[str, Callable] = None,
                 tool_function_callable_kwargs: Dict[str, Any] = None,
                 conversation_mode: bool = True,
                 max_tool_rounds: int = 5,
                 **kwargs: Any):
        """Initializes an OpenAIChat client.

        Args:
            api_key: The API key for authentication.
            base_url: The base URL for the API endpoint.
            default_model: The default model ID.
            instructions: System instructions for the model.
            messages: Initial list of conversation messages.
            tools: A list of tool definitions in OpenAI format.
            tool_function_mapper: Map of tool names to Python callables.
            tool_function_callable_kwargs: Static kwargs for tool callables.
            conversation_mode: If True, history is preserved across asks.
            max_tool_rounds: Limits recursive tool calls per ask.
            **kwargs: Additional arguments for the OpenAI client or API.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.instructions = instructions
        self.conversation_mode = conversation_mode
        self.max_tool_rounds = max_tool_rounds

        self.messages = messages or []
        if not self.messages and self.instructions:
            self.messages = [{"role": "system", "content": self.instructions}]

        # Use the centralized ToolSet for management.
        self.tool_set = ToolSet([
            FunctionTool(t, tool_function_mapper, tool_function_callable_kwargs)
            for t in (tools or []) if t.get("type") == "function"
        ])
        self.latest_tool_call_result = {}

        create_params = signature(self.client.chat.completions.create).parameters
        self.openai_args_dict = {
            k: v for k, v in kwargs.items() if k in create_params
        }

        LOGGER.info(f"OpenAIChat initialized (model: {default_model})")

    def _extract_usage(self, usage: Any) -> TokenUsage:
        """Extracts TokenUsage from an OpenAI usage object.

        Args:
            usage: The usage object returned by the OpenAI SDK.

        Returns:
            A populated TokenUsage instance.
        """
        if not usage:
            return TokenUsage()
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0))

    def _parse_completion(
            self,
            completion: Any) -> Tuple[str, List[Dict[str, Any]], str, TokenUsage]:
        """Parses a standard ChatCompletion object.

        Args:
            completion: The completion object from the OpenAI SDK.

        Returns:
            A tuple of (content, tool_calls, reasoning, usage).
        """
        choice = completion.choices[0]
        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            tool_calls = [{
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            } for tc in choice.message.tool_calls]
        reasoning = getattr(choice.message, "reasoning_content", "")
        usage = self._extract_usage(completion.usage)
        return content, tool_calls, reasoning, usage

    def _parse_stream(
            self,
            stream: Any) -> Tuple[str, List[Dict[str, Any]], str, TokenUsage]:
        """Parses a streaming ChatCompletion response.

        Args:
            stream: The stream generator from the OpenAI SDK.

        Returns:
            A tuple of (content, tool_calls, reasoning, usage) aggregated
            from all chunks.
        """
        content = ""
        tool_calls = []
        reasoning = ""
        usage = TokenUsage()

        for chunk in stream:
            if chunk.usage:
                usage = self._extract_usage(chunk.usage)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                content += delta.content
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                reasoning += delta.reasoning_content
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    while len(tool_calls) <= tc_chunk.index:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        })
                    tc = tool_calls[tc_chunk.index]
                    if tc_chunk.id:
                        tc["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            tc["function"]["name"] = tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tc["function"][
                                "arguments"] += tc_chunk.function.arguments
        return content, tool_calls, reasoning, usage

    def chat_completion(self,
                        messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        **kwargs: Any) -> LLMResponse:
        """Sends a single chat completion request.

        Args:
            messages: Conversation context.
            model: Specific model ID to use.
            **kwargs: Extra parameters for the API.

        Returns:
            An LLMResponse containing the model's output.
        """
        try:
            target_model = model or self.default_model
            args = {**self.openai_args_dict, **kwargs}
            completion = self.client.chat.completions.create(
                model=target_model, messages=messages, **args)
            content, tool_calls, reasoning, usage = self._parse_completion(
                completion)
            return LLMResponse(True,
                               200,
                               content,
                               usage=usage,
                               tool_calls=tool_calls,
                               reasoning_content=reasoning,
                               raw_response=completion)
        except Exception as e:
            return LLMResponse(False, 500, f"Error: {e}")

    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: bool = True,
            **kwargs: Any) -> LLMResponse:
        """Orchestrates a chat interaction, handling history and tool calls.

        Args:
            prompt: The user's input text.
            images: Optional list of images for multimodal input.
            stream: Whether to use streaming for API calls.
            **kwargs: Additional parameters for the model.

        Returns:
            An LLMResponse containing the final answer and usage metadata.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)

            # Use local messages to prevent state pollution on error.
            current_messages = self.messages.copy(
            ) if self.conversation_mode else []
            if not current_messages and self.instructions:
                current_messages.append({
                    "role": "system",
                    "content": self.instructions
                })
            current_messages.append({"role": "user", "content": user_content})

            tools_defs = (self.tool_set.get_openai_schema()
                          if self.tool_set else None)

            final_content = ""
            final_reasoning = ""
            total_usage = TokenUsage()

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                completion_args = {
                    "messages": current_messages,
                    "model": self.default_model,
                    "tools": tools_defs,
                    "stream": stream,
                    **self.openai_args_dict,
                    **kwargs
                }

                response = self.client.chat.completions.create(
                    **completion_args)
                if stream:
                    content, tool_calls, reasoning, usage = self._parse_stream(
                        response)
                else:
                    content, tool_calls, reasoning, usage = self._parse_completion(
                        response)

                final_content = content
                final_reasoning += reasoning
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                total_usage.total_tokens += usage.total_tokens

                if not tool_calls:
                    # Final response obtained.
                    break

                # Handle Tool Calls via centralized execute_tool_calls.
                assistant_msg = {"role": "assistant", "tool_calls": tool_calls}
                if content:
                    assistant_msg["content"] = content
                current_messages.append(assistant_msg)

                tool_messages, tool_results = self.tool_set.execute_tool_calls(
                    tool_calls)
                current_messages.extend(tool_messages)
                self.latest_tool_call_result.update(tool_results)

            # Loop finished or max rounds reached.
            if final_content:
                current_messages.append({
                    "role": "assistant",
                    "content": final_content
                })

            # Commit changes to self.messages only on success.
            if self.conversation_mode:
                self.messages = current_messages

            return LLMResponse(True,
                               200,
                               final_content,
                               usage=total_usage,
                               reasoning_content=final_reasoning)

        except Exception as e:
            LOGGER.error(f"ask failed: {e}")
            status = 400 if isinstance(e, BadRequestError) else 500
            return LLMResponse(False, status, f"Unexpected error: {e}")
