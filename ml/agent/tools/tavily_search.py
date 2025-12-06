"""Tavily search tool implementation.

This module wraps the Tavily search API to provide a simple query interface
returning summarized results for use in ReAct trajectories.
"""

from typing import Any, Dict, List
from tavily import TavilyClient
from tavily.tavily import json

from ml.config.settings import settings
from ml.agent.tools.decorator import dspy_tool


def _client() -> TavilyClient:
    if settings.TAVILY_API_KEY is None:
        raise RuntimeError("tavily_api_key is not configured")
    return TavilyClient(api_key=settings.TAVILY_API_KEY)


@dspy_tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for the given query.

    Use this tool when the user asks a question that requires searching the web for information about the university, events, or other general topics.
    This is a fallback tool that should be used only if other tools do not provide the information.

    Parameters
    ----------
    query: str
        The query to search for.
    max_results: int
        The maximum number of results to return.

    Returns
    -------
    str
        A JSON string containing a dictionary with a summary of the search results and the sources.
        The dictionary has the following structure:
        {
            "summary": [{
                "title": str,
                "snippet": str,
                "source": str
            }]
        }
    """
    try:
        client = _client()
        response = client.search(query=query, max_results=max_results)
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"}, ensure_ascii=False)

    results: List[Dict[str, Any]] = response.get("results", [])
    summary: List[Dict[str, str]] = []

    for item in results:
        title = item.get("title") or ""
        source = item.get("url") or ""
        snippet = item.get("content") or item.get("snippet") or ""
        summary.append({
            "title": title if title else "",
            "snippet": snippet,
            "source": source
        })

    return json.dumps({"summary": summary}, ensure_ascii=False)
