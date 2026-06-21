#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   __init__.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from isobase.llm.providers.base import BaseLLMClient
from isobase.llm.providers.openai_chat import OpenAIChat
from isobase.llm.providers.anthropic_messages import AnthropicMessages

# 现代 Python 3.12+ 的类型别名写法
type LLMClient = BaseLLMClient

__all__ = ["BaseLLMClient", "LLMClient", "OpenAIChat", "AnthropicMessages"]
