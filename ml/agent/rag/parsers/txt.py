"""Plain text parser implementation."""

from typing import Union
from ml.agent.rag.parsers.base import BaseParser


class TxtParser(BaseParser):
    """Parser for plain text documents."""

    async def parse(self, source: Union[str, bytes]) -> str:
        """
        Parse text content.

        Parameters
        ----------
        source : Union[str, bytes]
            The text content as bytes or file path.

        Returns
        -------
        str
            The text content.
        """
        if isinstance(source, bytes):
            return source.decode("utf-8")
        else:
            # If it's a file path
            with open(source, "r", encoding="utf-8") as f:
                return f.read()
