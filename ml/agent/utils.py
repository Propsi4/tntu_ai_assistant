"""Utility functions for the TNTU assistant agent."""

from playwright.async_api import async_playwright
import asyncio
from typing import Union, Dict, Any, List
from langchain_core.messages import ToolMessage
import json
import requests
from bs4 import BeautifulSoup
import re
from markdownify import markdownify as md


async def is_url_available(url: str) -> bool:
    """
    Check if the URL is available.

    Parameters
    ----------
    url : str
        The URL to check.

    Returns
    -------
    bool
        True if the URL is available, False otherwise.
    """
    try:
        response = requests.head(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


async def fetch_rendered_html(url: str, wait_selector: str | None = None, timeout_ms: int = 30000) -> str:
    """
    Extract the rendered HTML from the given URL.

    Parameters:
        url: str
            The URL to extract the rendered HTML from.
        wait_selector: str | None, optional
            The CSS selector to wait for when there's a known anchor element.
        timeout_ms: int, optional
            The overall timeout for waits in milliseconds.

    Returns:
        str: The rendered HTML from the given URL.
    """
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_load_state("load", timeout=timeout_ms)

        if wait_selector:
            await page.wait_for_selector(wait_selector, timeout=timeout_ms)

        await page.wait_for_load_state("networkidle", timeout=timeout_ms)

        html = await page.content()
        await browser.close()
        return html


def html_to_markdown(html_content: str) -> str:
    """
    Convert HTML to clean Markdown suitable for LLM analysis.

    Parameters:
        html_content: str
            The HTML content to convert to Markdown.

    Returns:
        str: The Markdown content.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    for hidden in soup(["script", "style", "meta", "noscript", "head", "iframe", "svg"]):
        hidden.decompose()

    # 2. Convert to Markdown
    # We use 'ATX' style (e.g., # Header) as it is token-efficient for LLMs.
    text = md(str(soup), heading_style="ATX", strip=['a', 'img'])

    # 3. Remove URLs & Link artifacts
    # Pattern explanation: matches http/https URLs and removes them
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # Remove any remaining Markdown link syntax [text](url) -> text
    # Note: markdownify's strip=['a'] handles tags, but this catches loose markdown.
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 4. Filter Special Symbols
    # Keep alphanumeric, whitespace, and standard punctuation.
    # Remove emojis and obscure symbols which consume tokens without adding meaning.
    text = re.sub(r'[^\w\s.,!?;:\-\'"()#*]', '', text)

    # 5. Final Whitespace Cleanup
    # Replace multiple newlines with a single newline to save context window space.
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse internal spacing

    return text.strip()


def _run_coro_anywhere(coro) -> any:
    """
    Safely run a coroutine in an event loop.

    Parameters:
        coro: coroutine
            The coroutine to run.

    Returns:
        any: The result of the coroutine.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # In Jupyter or any already-running loop:
        # If inside a synchronous function, schedule and wait for result using a Future.
        # Prefer nest_asyncio if available for re-entrancy, but keep it optional.
        try:
            import nest_asyncio  # optional
            nest_asyncio.apply()
        except Exception:
            pass
        # If running in main thread with IPython, we can use asyncio.ensure_future and gather via loop.create_task
        # However, we cannot block the event loop; instead, use a small helper to bridge sync->async:
        # IPython supports top-level await; but here we return a blocking result for sync API by running a nested loop safely.
        # After nest_asyncio.apply(), run_until_complete becomes legal.
        return asyncio.get_event_loop().run_until_complete(coro)
    else:
        # No loop is running; safe to use asyncio.run
        return asyncio.run(coro)


def tool_messages_from_flat_json(data: Union[str, Dict[str, Any]]) -> List[ToolMessage]:
    """
    Convert a flat numbered JSON to a list of ToolMessages.

    Parameters:
        data: Union[str, Dict[str, Any]]
            The data to convert to a list of ToolMessages.

    Returns:
        List[ToolMessage]: The list of ToolMessages.
    """
    if isinstance(data, str):
        data = json.loads(data)

    # Collect all numeric suffixes
    indices = set()
    for k in data.keys():
        if "_" in k and k.rsplit("_", 1)[-1].isdigit():
            indices.add(int(k.rsplit("_", 1)[-1]))
    if not indices:
        return []

    messages: List[ToolMessage] = []
    for i in sorted(indices):
        thought = data.get(f"thought_{i}")
        tool_name = data.get(f"tool_name_{i}")
        tool_args = data.get(f"tool_args_{i}")
        observation = data.get(f"observation_{i}")

        # Skip finish tool calls
        if tool_name is not None and str(tool_name).strip().lower() == "finish":
            continue

        # Normalize args/obs types
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except Exception:
                pass
        if isinstance(observation, str):
            # If observation looks like JSON, parse; otherwise keep as plain text
            obs_parsed = None
            try:
                obs_parsed = json.loads(observation)
            except Exception:
                pass
        else:
            obs_parsed = observation

        # Pack content for ToolMessage; you can adjust fields to your needs
        content = {
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "observation": obs_parsed if obs_parsed is not None else observation,
            "index": i,
        }

        # ToolMessage typically requires a tool_call_id; generate a stable one per index
        tool_call_id = f"tool_call_{i}"
        messages.append(ToolMessage(content=content, name=tool_name, tool_call_id=tool_call_id))

    return messages
