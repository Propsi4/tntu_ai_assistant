"""Web fetching and readability extraction tool.

Provides HTTP fetching via httpx and content extraction using BeautifulSoup
and readability-lxml to produce clean text for analysis.
"""

from __future__ import annotations

import httpx
from ml.agent.tools.decorator import dspy_tool
import json
from ml.agent.utils import html_to_markdown


@dspy_tool
def web_fetch(url: str, timeout_seconds: float = 15.0) -> str:
    """
    Fetch a URL and return extracted readable text.

    Use this tool when the user asks a question that requires fetching a URL and extracting the text.

    Parameters
    ----------
    url : str
        The URL to fetch.

    Returns
    -------
    str
        A JSON string containing the extracted text and metadata.
        The dictionary has the following structure:
        {
            "content": str (Markdown content of the HTML),
            "source": str (the URL of the fetched content)
        }
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    with httpx.Client(follow_redirects=True, headers=headers, timeout=timeout_seconds) as client:
        response = client.get(url)
        response.raise_for_status()
        html = response.text

    text = html_to_markdown(html)

    return json.dumps({
        "content": text,
        "source": url
    }, ensure_ascii=False)
