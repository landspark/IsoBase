#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for LLM providers.

@File   :   base.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

from PIL import Image as PILImage

from isobase.core.image_service import convert_image_to_data_url
from isobase.llm.entities import LLMResponse


class BaseLLMClient(ABC):
    """Abstract base class for all LLM provider clients.

    Ensures a consistent interface for interacting with various LLM APIs.
    """

    @abstractmethod
    def generate(self,
                 messages: List[Dict[str, str]],
                 model: Optional[str] = None,
                 **kwargs: Any) -> LLMResponse:
        """Sends a non-streaming generation request.

        Args:
            messages: A list of message objects representing the conversation.
            model: The specific model ID to use for this request.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An LLMResponse object containing the structured result.
        """
        pass

    @abstractmethod
    def generate_stream(self,
                        messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        **kwargs: Any) -> Iterator[LLMResponse]:
        """Sends a streaming generation request.

        Args:
            messages: A list of message objects representing the conversation.
            model: The specific model ID to use for this request.
            **kwargs: Additional provider-specific parameters.

        Yields:
            LLMResponse objects containing incremental content chunks.
        """
        pass

    @abstractmethod
    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: bool = False,
            **kwargs: Any) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """Orchestrates a chat interaction, handling history and tool calls.

        Args:
            prompt: The user's input text.
            images: Optional list of images for multimodal input.
            stream: Whether to use streaming for the interaction.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An LLMResponse (non-stream) or Iterator[LLMResponse] (stream).
        """
        pass

    @classmethod
    def build_user_message_content(
        cls, prompt: str,
        images: Optional[List[PILImage.Image]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Builds user message content, supporting multimodal input.

        Standardizes the input into an OpenAI-compatible format if images
        are provided.

        Args:
            prompt: The text prompt provided by the user.
            images: An optional list of PIL Image objects to attach.

        Returns:
            A string if only text is provided, or a list of content dictionaries
            for multimodal requests.

        Raises:
            TypeError: If any item in the images list is not a PIL Image.
        """
        if not images:
            return prompt

        content = [{"type": "text", "text": prompt}]
        for img in images:
            if not isinstance(img, PILImage.Image):
                raise TypeError("Each image must be a PIL.Image.Image instance.")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": convert_image_to_data_url(img)
                },
            })
        return content
