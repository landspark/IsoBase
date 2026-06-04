#! python3
# -*- encoding: utf-8 -*-
"""OpenAI Chat Completion compatible interface with tool calling and history support.

@File   :   openai_chat.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.

from typing import Any, List, Dict, Optional, Union, Tuple, Callable
from json import loads
from inspect import signature
from PIL import Image as PILImage

from openai import (
    OpenAI,
    APIError,
    RateLimitError,
    BadRequestError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
)

from ark.llm.providers.base import BaseLLMClient
from ark.llm.tools import FunctionTool
from ark.core.logger import LOGGER
from ark.core.image_service import convert_image_to_data_url

class OpenAIChat(BaseLLMClient):
    """
    Client for OpenAI Chat Completion compatible APIs, supporting history and tools.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        default_model: str = "gpt-3.5-turbo",
        instructions: str = "",
        messages: List[Dict[str, str]] = None,
        tools: List[Dict[str, Any]] = None,
        tool_function_mapper: Dict[str, Callable] = None,
        tool_function_callable_kwargs: Dict[str, Any] = None,
        conversation_mode: bool = True,
        **kwargs: Any
    ):
        """
        Initializes the OpenAI Chat client.

        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            default_model: Default model to use.
            instructions: System instructions to define the agent behavior.
            messages: Initial conversation history.
            tools: List of OpenAI-style tool definitions.
            tool_function_mapper: Map of tool names to Python callables.
            tool_function_callable_kwargs: Additional kwargs for tool callables.
            conversation_mode: If True, retains conversation history.
            **kwargs: Additional parameters for the OpenAI client and chat completions.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.instructions = instructions
        self.conversation_mode = conversation_mode
        
        # Initialize messages
        if messages:
            self.messages = messages
        elif self.instructions:
            self.messages = [{"role": "system", "content": self.instructions}]
        else:
            self.messages = []

        # Initialize tools
        tools = tools or []
        tool_function_mapper = tool_function_mapper or {}
        tool_function_callable_kwargs = tool_function_callable_kwargs or {}
        
        self.function_tools = [
            FunctionTool(
                tool_definition_dict=t, 
                tool_function_mapper=tool_function_mapper,
                tool_function_callable_kwargs=tool_function_callable_kwargs
            ) for t in tools if t.get("type") == "function"
        ]
        self.latest_tool_call_result = {}

        # Filter kwargs for chat completion
        create_params = signature(self.client.chat.completions.create).parameters
        self.openai_args_dict = {k: v for k, v in kwargs.items() if k in create_params}

        LOGGER.info(f"Initialized OpenAIChat (model: {default_model}, conversation_mode: {conversation_mode})")

    @staticmethod
    def build_user_message_content(
        prompt: str,
        images: Optional[List[PILImage.Image]] = None,
    ) -> Union[str, List[Dict[str, Any]]]:
        """Builds message content, supporting multimodal input if images are provided."""
        if not images:
            return prompt

        content = [{"type": "text", "text": prompt}]
        for img in images:
            if not isinstance(img, PILImage.Image):
                raise TypeError("Each image must be a PIL.Image.Image instance.")
            content.append({
                "type": "image_url",
                "image_url": {"url": convert_image_to_data_url(img)},
            })
        return content

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """Low-level call to OpenAI API."""
        target_model = model or self.default_model
        merged_kwargs = {**self.openai_args_dict, **kwargs}
        return self.client.chat.completions.create(
            model=target_model,
            messages=messages,
            **merged_kwargs
        )

    def ask(
        self, 
        prompt: str, 
        images: Optional[List[PILImage.Image]] = None,
        stream: bool = True,
        **kwargs: Any
    ) -> Tuple[bool, int, str]:
        """
        High-level interface for chat, handling history, tools, and streaming.

        Returns:
            Tuple[bool, int, str]: (is_success, status_code, content)
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            
            # Manage message history
            if self.conversation_mode:
                messages = self.messages.copy()
            else:
                messages = []
                if self.instructions:
                    messages.append({"role": "system", "content": self.instructions})
            
            messages.append({"role": "user", "content": user_content})

            tools_definitions = [t.tool_definition_dict for t in self.function_tools] if self.function_tools else None

            # First Call
            response_content = ""
            current_tool_calls = []
            
            completion_args = {
                "messages": messages,
                "model": self.default_model,
                "tools": tools_definitions,
                "stream": stream,
                **self.openai_args_dict,
                **kwargs
            }

            if stream:
                response = self.client.chat.completions.create(**completion_args)
                for chunk in response:
                    if not chunk.choices: continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        response_content += delta.content
                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            while len(current_tool_calls) <= tc_chunk.index:
                                current_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            tc = current_tool_calls[tc_chunk.index]
                            if tc_chunk.id: tc["id"] = tc_chunk.id
                            if tc_chunk.function:
                                if tc_chunk.function.name: tc["function"]["name"] = tc_chunk.function.name
                                if tc_chunk.function.arguments: tc["function"]["arguments"] += tc_chunk.function.arguments
            else:
                response = self.client.chat.completions.create(**completion_args)
                msg = response.choices[0].message
                response_content = msg.content or ""
                if msg.tool_calls:
                    current_tool_calls = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]

            # Handle Tool Calls
            if current_tool_calls:
                messages.append({"role": "assistant", "tool_calls": current_tool_calls})
                
                for tc in current_tool_calls:
                    func_name = tc["function"]["name"]
                    tool = next((t for t in self.function_tools if t.name == func_name), None)
                    
                    if tool:
                        result = tool.execute(tc["function"]["arguments"])
                        # Logic from legacy: result might be (is_success, value)
                        if isinstance(result, tuple) and len(result) == 2:
                            is_success, tool_output = result
                        else:
                            is_success, tool_output = True, str(result)
                        
                        self.latest_tool_call_result[func_name] = is_success
                    else:
                        tool_output = f"Tool '{func_name}' not found."
                        self.latest_tool_call_result[func_name] = False
                    
                    messages.append({
                        "tool_call_id": tc["id"],
                        "role": "tool",
                        "name": func_name,
                        "content": str(tool_output)
                    })

                # Second Call after tools
                second_response = self.client.chat.completions.create(
                    messages=messages,
                    model=self.default_model,
                    tools=tools_definitions,
                    stream=stream,
                    **self.openai_args_dict,
                    **kwargs
                )
                
                final_content = ""
                if stream:
                    for chunk in second_response:
                        if chunk.choices[0].delta.content:
                            final_content += chunk.choices[0].delta.content
                else:
                    final_content = second_response.choices[0].message.content or ""
                
                response_content = final_content

            # Update History
            if response_content:
                messages.append({"role": "assistant", "content": response_content})
            
            if self.conversation_mode:
                self.messages = messages

            return True, 200, response_content

        except RateLimitError:
            return True, 429, "Rate Limit Error"
        except BadRequestError as e:
            return True, 400, f"Bad Request: {e}"
        except APITimeoutError:
            return False, 500, "API Timeout"
        except AuthenticationError:
            return False, 401, "Authentication Failed"
        except InternalServerError:
            return False, 500, "Internal Server Error"
        except APIError as e:
            return False, 500, f"API Error: {e}"
        except Exception as e:
            LOGGER.error(f"Unexpected error: {e}")
            return False, 500, f"Unexpected error: {e}"
