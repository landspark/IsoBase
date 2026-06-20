#! python3
# -*- encoding: utf-8 -*-
"""OpenAI Chat Completion client with structured responses and multi-turn tool support.

@File   :   openai_chat.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from openai import (
    BadRequestError,
    OpenAI,
    Stream,
)
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from PIL import Image as PILImage

from isobase.core.logger import LOGGER
from isobase.llm.entities import LLMResponse, TokenUsage, ToolCall
from isobase.llm.providers.base import BaseLLMClient
from isobase.llm.tools import FunctionTool, SearchTool, ToolSet


class OpenAIChat(BaseLLMClient):
    """Optimized client for OpenAI Chat Completion compatible APIs.

    Supports structured LLMResponse, safe history mutation, and multi-turn
    tool calling via internal orchestration loops.

    Attributes:
        client: The underlying OpenAI client instance.
        default_model: The default model ID to use for requests.
        instructions: System instructions defining the agent's behavior.
        conversation_mode: Whether to maintain a message history.
        max_tool_rounds: Maximum number of recursive tool-calling rounds.
        messages: The current conversation history.
        tool_set: A collection of available tools.
        latest_tool_call_result: Results of the last tool executions.
        openai_args_dict: Filtered kwargs for the OpenAI SDK.
    """

    def __init__(self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-5.4-mini",
        instructions: str = "",
        messages: List[Dict[str, str]] = None,
        tools: Union[ToolSet, List[FunctionTool], None] = None,
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
            tools: An optional ToolSet or list of FunctionTool objects.
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

        if isinstance(tools, ToolSet):
            self.tool_set = tools
        else:
            self.tool_set = ToolSet(tools or [])
        self.latest_tool_call_result = {}

        create_params = signature(self.client.chat.completions.create).parameters
        self.openai_args_dict = {
            k: v for k, v in kwargs.items() if k in create_params
        }

        LOGGER.info(f"OpenAIChat initialized (model: {default_model})")

    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: bool = False,
            **kwargs: Any) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """Orchestrates a chat interaction, handling history and tool calls.

        Args:
            prompt (str): The user's input text.
            images (Optional[List[PILImage.Image]]): Optional list of images for multimodal input.
            stream (bool, optional): Whether to use streaming for the interaction. Defaults to False.
            **kwargs: Additional parameters for the model.

        Returns:
            An LLMResponse (non-stream) or Iterator[LLMResponse] (stream).
        """
        if stream:
            return self.__ask_loop_stream(prompt, images, **kwargs)
        else:
            return self.__ask_loop(prompt, images, **kwargs)

    def generate(self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            **kwargs: Any
        ) -> LLMResponse:
        """Sends a single non-streaming chat completion request.

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
            completion: ChatCompletion = self.client.chat.completions.create(
                model=target_model, messages=messages, stream=False, **args)
            content, tool_calls, reasoning, usage = self.__parse_completion(
                completion)
            return LLMResponse(
                success=True, status_code=200,
                content=content, usage=usage,
                tool_calls=tool_calls, reasoning_content=reasoning,
                raw_response=completion
            )
        except Exception as e:
            return LLMResponse(success=False, status_code=500, content=f"Error: {e}")

    def generate_stream(
            self, messages: List[Dict[str, str]],
            model: Optional[str] = None, **kwargs: Any
        ) -> Iterator[LLMResponse]:
        """Sends a streaming chat completion request.

        Args:
            messages: Conversation context.
            model: Specific model ID to use.
            **kwargs: Extra parameters for the API.

        Yields:
            LLMResponse objects containing chunks or final results.
        """
        target_model = model or self.default_model
        args = {**self.openai_args_dict, **kwargs}

        stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=target_model, messages=messages,
            stream=True, stream_options={"include_usage": True},
            **args
        )

        content = ""
        tool_calls = []
        reasoning = ""
        usage = TokenUsage()

        for chunk in stream:
            chunk_usage = self.__extract_usage(chunk.usage) if chunk.usage else None
            if chunk_usage:
                usage = chunk_usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            delta_content = delta.content or ""
            delta_reasoning = delta.reasoning_content or ""

            if delta_content:
                content += delta_content
            if delta_reasoning:
                reasoning += delta_reasoning

            # Handle tool calls in stream.
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
                            tc["function"]["arguments"] += tc_chunk.function.arguments

            # Yield incremental text chunks.
            if delta_content or delta_reasoning:
                yield LLMResponse(
                    success=True, status_code=200,
                    content=delta_content, reasoning_content=delta_reasoning,
                    raw_response=chunk
                )

        # Final yield with full aggregated data for logic/tools, but empty content
        # to avoid duplication in the ask loop.
        final_tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"]
            ) for tc in tool_calls
        ]

        yield LLMResponse(
            success=True, status_code=200,
            content="", tool_calls=final_tool_calls,
            reasoning_content="",
            usage=usage
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
            tools_defs = self.__build_tools_schema()

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                response = self.generate(messages=current_messages,
                                               tools=tools_defs,
                                               **kwargs)
                if not response.success:
                    return response

                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens
                total_usage.total_tokens += response.usage.total_tokens
                final_reasoning += (response.reasoning_content or "")

                if not response.tool_calls:
                    final_content = response.content
                    break

                # Handle Tool Calls.
                assistant_msg = self.__build_assistant_message(
                    content=response.content,
                    tool_calls=response.raw_response.choices[0].message.tool_calls
                )
                current_messages.append(assistant_msg)

                tool_outputs, tool_results = self.tool_set.execute_tool_calls(
                    response.tool_calls)

                tool_messages = []
                for tc, output in zip(response.tool_calls, tool_outputs):
                    tool_messages.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": tc.name,
                        "content": str(output)
                    })
                current_messages.extend(tool_messages)
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
            Incremental chunks and final summary.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            current_messages = self.__prepare_messages(user_content)
            tools_defs = self.__build_tools_schema()

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                tool_calls = []
                current_round_content = ""
                current_round_reasoning = ""

                # Iterate through chunks from the streaming completion.
                for chunk_resp in self.generate_stream(
                        messages=current_messages, tools=tools_defs, **kwargs):
                    
                    if chunk_resp.tool_calls:
                        tool_calls = chunk_resp.tool_calls
                    if chunk_resp.usage:
                        total_usage.input_tokens += chunk_resp.usage.input_tokens
                        total_usage.output_tokens += chunk_resp.usage.output_tokens
                        total_usage.total_tokens += chunk_resp.usage.total_tokens
                    
                    # chunk_resp.content is incremental in chunks, empty in final.
                    if chunk_resp.content or chunk_resp.reasoning_content or chunk_resp.usage.total_tokens > 0:
                        current_round_content += chunk_resp.content
                        current_round_reasoning += chunk_resp.reasoning_content
                        # Forward the increment or final summary to the user.
                        yield chunk_resp

                final_reasoning += current_round_reasoning

                if not tool_calls:
                    final_content = current_round_content
                    break

                # Handle Tool Calls.
                assistant_msg = self.__build_assistant_message(
                    content=current_round_content,
                    tool_calls=tool_calls
                )
                current_messages.append(assistant_msg)

                tool_outputs, tool_results = self.tool_set.execute_tool_calls(
                    tool_calls)

                tool_messages = []
                for tc, output in zip(tool_calls, tool_outputs):
                    tool_messages.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": tc.name,
                        "content": str(output)
                    })
                current_messages.extend(tool_messages)
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

    def __extract_usage(self, usage: Optional[CompletionUsage]) -> TokenUsage:
        """Extracts TokenUsage from an OpenAI usage object.

        Args:
            usage (Optional[CompletionUsage]): The usage object returned by the OpenAI SDK.

        Returns:
            A populated TokenUsage instance.
        """
        if not usage:
            return TokenUsage()
        else:
            return TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

    def __parse_completion(
            self, completion: ChatCompletion
        ) -> Tuple[str, List[ToolCall], str, TokenUsage]:
        """Parses a standard ChatCompletion instance.

        Args:
            completion (ChatCompletion): The ChatCompletion instance from the OpenAI SDK.

        Returns:
            A tuple of (content, tool_calls, reasoning, usage).
        """
        choice = completion.choices[0]
        content = choice.message.content or ""
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.type == "function":
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        )
                    )
        reasoning = choice.message.reasoning_content or ""
        usage = self.__extract_usage(completion.usage)
        return content, tool_calls, reasoning, usage

    def __build_tools_schema(self) -> Optional[List[Dict[str, Any]]]:
        """Builds tool schemas in OpenAI function format.

        Returns:
            A list of OpenAI-format tool schemas, or None if no tools are defined.
        """
        if not self.tool_set or not self.tool_set.tools:
            return None

        schemas = []
        for tool in self.tool_set.tools:
            # Avoid direct import to prevent circular dependency if they're in same tree.
            if isinstance(tool, SearchTool) and not tool.force_external:
                schemas.append({"type": "web_search"})
            else:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema,
                    }
                })
        return schemas

    def __prepare_messages(self, user_content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Helper to prepare the message list for ask loops."""
        messages = self.messages.copy() if self.conversation_mode else []
        if not messages and self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": user_content})
        return messages

    def __build_assistant_message(
        self,
        content: Optional[str],
        tool_calls: List[Union[Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Constructs an assistant message containing content and tool calls.

        This method structures the assistant's turn into the exact format required
        by OpenAI's API so it can be appended to the conversation history
        (`current_messages` and ultimately `self.messages`) for multi-turn routing.
        Handles both SDK ChoiceMessageToolCall objects and raw dictionaries.

        Args:
            content: Text content from the assistant (or None).
            tool_calls: List of tool call objects or raw dictionaries.

        Returns:
            The structured assistant message dictionary.
        """
        assistant_msg: Dict[str, Any] = {"role": "assistant", "tool_calls": []}

        for tc in tool_calls:
            if isinstance(tc, dict):
                # Handle raw dictionary tool calls (from __ask_loop_stream)
                tc_type = tc.get("type", "function")
                if tc_type == "function":
                    assistant_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
                else:
                    assistant_msg["tool_calls"].append(tc)
            else:
                # Handle SDK ChoiceMessageToolCall objects (from __ask_loop)
                if tc.type == "function":
                    assistant_msg["tool_calls"].append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
                else:
                    # Pass through native tools verbatim
                    assistant_msg["tool_calls"].append(tc.model_dump())

        if content:
            assistant_msg["content"] = content

        return assistant_msg

    def __handle_ask_exception(self, e: Exception) -> LLMResponse:
        """Helper to handle exceptions in ask loops."""
        LOGGER.error(f"ask failed: {e}")
        status = 400 if isinstance(e, BadRequestError) else 500
        return LLMResponse(success=False, status_code=status, content=f"Unexpected error: {e}")
