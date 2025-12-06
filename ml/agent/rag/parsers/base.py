"""Base parser interface for extracting text from various sources."""

from abc import ABC, abstractmethod
from typing import Union


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    async def parse(self, source: Union[str, bytes]) -> str:
        """
        Parse the source and extract text content.

        Parameters
        ----------
        source : Union[str, bytes]
            The source to parse. Can be a file path, URL, or raw bytes.

        Returns
        -------
        str
            The extracted text content.
        """
        pass
