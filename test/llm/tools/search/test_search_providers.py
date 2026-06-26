#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_search_providers.py
@Created:   2026/06/21 00:03
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import json
import pytest
from isobase.llm.tools.search import SearchTool, BaseSearchProvider, BraveSearchProvider, TavilySearchProvider
from unittest.mock import patch, MagicMock

def test_brave_search_provider_success():
    provider = BraveSearchProvider(api_key="test-key")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "web": {
            "results": [
                {"title": "Brave 1", "url": "https://brave.com/1", "description": "Snippet 1"},
                {"title": "Brave 2", "url": "https://brave.com/2", "description": "Snippet 2"}
            ]
        }
    }

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.return_value = mock_response

        results = provider.search("test query", count=2)

        assert results.success is True
        assert len(results.results) == 2
        assert results.results[0].title == "Brave 1"
        assert results.results[0].url == "https://brave.com/1"
        assert results.results[0].snippet == "Snippet 1"

        assert results.results[1].title == "Brave 2"
        assert results.results[1].url == "https://brave.com/2"
        assert results.results[1].snippet == "Snippet 2"

        mock_instance.get.assert_called_once_with(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"Accept": "application/json", "X-Subscription-Token": "test-key"},
            params={"q": "test query", "count": 2}
        )

def test_brave_search_provider_error():
    provider = BraveSearchProvider(api_key="test-key")

    with patch("httpx.Client") as mock_client:
        mock_instance = mock_client.return_value.__enter__.return_value
        mock_instance.get.side_effect = Exception("Network Error")

        results = provider.search("test query")

        assert results.success is False
        assert results.error == "Brave Search failed: Network Error"


def test_tavily_search_provider_success():
    provider = TavilySearchProvider(api_key="test-key")

    with patch("isobase.llm.tools.search.providers.TavilyClient") as mock_tavily_client:
        mock_client_instance = MagicMock()
        mock_tavily_client.return_value = mock_client_instance

        mock_client_instance.search.return_value = {
            "results": [
                {"title": "Tavily 1", "url": "https://tavily.com/1", "content": "Snippet 1", "raw_content": "Raw 1"},
            ]
        }

        results = provider.search("test query", include_raw_content=True)

        assert results.success is True
        assert len(results.results) == 1
        assert results.results[0].title == "Tavily 1"
        assert results.results[0].url == "https://tavily.com/1"
        assert results.results[0].snippet == "Snippet 1"
        assert results.results[0].raw_content == "Raw 1"

        mock_tavily_client.assert_called_once_with(api_key="test-key")
        mock_client_instance.search.assert_called_once_with(
            query="test query",
            search_depth="basic",
            max_results=5,
            include_raw_content=True
        )
