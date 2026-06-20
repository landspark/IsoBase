#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for Search Providers.

@File   :   base.py
@Created:   2026/06/20 23:50
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSearchProvider(ABC):
    """Abstract base class for search engine adapters used by SearchTool.

    Enterprise users can subclass this to inject their own internal search
    engines (e.g., ElasticSearch, internal corporate Bing endpoints).
    """

    @abstractmethod
    def search(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """Executes a search query.

        Args:
            query: The search term provided by the LLM or user.
            **kwargs: Extra parameters (e.g., limit, depth, region) supported by the provider.

        Returns:
            A list of dictionary objects representing the search results.
            Standardized format typically includes 'title', 'url', and 'snippet' or 'content'.
        """
        pass
