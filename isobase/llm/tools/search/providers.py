#! python3
# -*- encoding: utf-8 -*-
"""Concrete implementations of Search Providers.

@File   :   providers.py
@Created:   2026/06/20 23:51
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import httpx
from typing import Any, Dict, List, Optional
from tavily import TavilyClient

from .base import BaseSearchProvider


class BraveSearchProvider(BaseSearchProvider):
    """Search provider implementation for the Brave Search API.

    Brave Search API offers a generous free tier (2,000 requests/month) and fast
    responses, making it a great option for production usage.
    """

    def __init__(self, api_key: str):
        """Initializes the Brave Search Provider.

        Args:
            api_key: The subscription token for the Brave Search API.
        """
        self.api_key = api_key
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, count: int = 5, **kwargs: Any) -> List[Dict[str, Any]]:
        """Executes a search query against the Brave Search API.

        Args:
            query: The search query.
            count: Number of results to return (default 5).
            **kwargs: Extra parameters to pass to the Brave Search API.

        Returns:
            A list of dictionaries containing title, url, and snippet.
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        params = {"q": query, "count": count, **kwargs}

        try:
            with httpx.Client() as client:
                response = client.get(self.endpoint, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()

            results = []
            if "web" in data and "results" in data["web"]:
                for item in data["web"]["results"]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("description", "")
                    })
            return results

        except Exception as e:
            return [{"error": f"Brave Search failed: {str(e)}"}]


class TavilySearchProvider(BaseSearchProvider):
    """Search provider implementation for the Tavily API.

    Tavily is an AI-native search engine that provides clean, markdown-formatted
    web page content rather than just snippets, making it ideal for deep LLM research.
    """

    def __init__(self, api_key: str):
        """Initializes the Tavily Search Provider.

        Args:
            api_key: The API key for Tavily.
        """
        self.api_key = api_key

    def search(self, query: str, search_depth: str = "basic", max_results: int = 5, include_raw_content: bool = True, **kwargs: Any) -> List[Dict[str, Any]]:
        """Executes a search query against the Tavily API.

        Args:
            query: The search query.
            search_depth: The depth of the search ("basic" or "advanced").
            max_results: The maximum number of search results to return.
            include_raw_content: Whether to fetch the full parsed page content.
            **kwargs: Additional arguments for the TavilyClient.

        Returns:
            A list of search results.
        """
        try:
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_raw_content=include_raw_content,
                **kwargs
            )

            results = []
            for item in response.get("results", []):
                result_entry = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")
                }
                if include_raw_content and "raw_content" in item:
                    result_entry["raw_content"] = item["raw_content"]
                results.append(result_entry)

            return results

        except Exception as e:
            return [{"error": f"Tavily Search failed: {str(e)}"}]
