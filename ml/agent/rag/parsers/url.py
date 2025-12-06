"""URL parser implementation using BeautifulSoup."""

from typing import Union
from bs4 import BeautifulSoup
from ml.agent.rag.parsers.base import BaseParser
from ml.agent.utils import fetch_rendered_html, is_url_available


class UrlParser(BaseParser):
    """Parser for URLs."""

    async def parse(self, source: Union[str, bytes]) -> str:
        """
        Parse URL content and extract text.

        Parameters
        ----------
        source : Union[str, bytes]
            The URL string.

        Returns
        -------
        str
            Extracted text from the URL.
        """
        # Ensure source is a string
        if isinstance(source, bytes):
            url = source.decode("utf-8")
        else:
            url = source

        # Test if url is available
        if not await is_url_available(url):
            raise ValueError(f"URL {url} is not available")

        html = await fetch_rendered_html(url)

        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (
            phrase.strip() for line in lines for phrase in line.split("  ")
        )
        # Drop blank lines
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text
