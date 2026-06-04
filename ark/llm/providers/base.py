#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for LLM providers.

@File   :   base.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
from ark.llm.entities import LLMResponse

class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM provider clients.
    Ensures a consistent interface for interacting with various LLM APIs.
    """

    @abstractmethod
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Sends a chat completion request to the LLM provider.

        Args:
            messages (List[Dict[str, str]]): A list of message objects.
            model (Optional[str]): The model to use.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse: Structured response object.
        """
        pass
