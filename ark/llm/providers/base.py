#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for LLM providers.

@File   :   base.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

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
    ) -> Any:
        """
        Sends a chat completion request to the LLM provider.

        Args:
            messages (List[Dict[str, str]]): A list of message objects representing the conversation history.
            model (Optional[str]): The model to use for completion.
            **kwargs: Additional provider-specific parameters (e.g., temperature, max_tokens).

        Returns:
            Any: The response from the LLM provider.
        """
        pass
