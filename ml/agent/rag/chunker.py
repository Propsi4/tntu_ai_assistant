"""Text chunking logic."""

from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ml.config.settings import settings


class Chunker:
    """Chunker for splitting text into smaller segments."""

    def __init__(
        self,
        chunk_size: int = settings.DOCUMENT_MAX_TOKENS,
        chunk_overlap: int = 100
    ):
        """
        Initialize the chunker.

        Parameters
        ----------
        chunk_size : int
            The maximum number of tokens per chunk.
        chunk_overlap : int
            The number of overlapping characters/tokens between chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=False,
        )

    def chunk(self, text: str) -> List[str]:
        """
        Split text into chunks.

        Parameters
        ----------
        text : str
            The text to split.

        Returns
        -------
        List[str]
            List of text chunks.
        """
        return self.splitter.split_text(text)
