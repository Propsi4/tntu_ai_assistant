"""Factory for creating parsers based on content type."""

from typing import Dict, Type
from ml.agent.rag.parsers.base import BaseParser
from ml.agent.rag.parsers.pdf import PdfParser
from ml.agent.rag.parsers.docx import DocxParser
from ml.agent.rag.parsers.pptx import PptxParser
from ml.agent.rag.parsers.txt import TxtParser
from ml.agent.rag.parsers.url import UrlParser


class ParserFactory:
    """Factory class to get the appropriate parser."""

    _parsers: Dict[str, Type[BaseParser]] = {
        "application/pdf": PdfParser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxParser,  # noqa: E501
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": PptxParser,  # noqa: E501
        "text/plain": TxtParser,
        "url": UrlParser,
    }

    @classmethod
    def get_parser(cls, content_type: str) -> BaseParser:
        """
        Get a parser instance for the given content type.

        Parameters
        ----------
        content_type : str
            The MIME type of the content or 'url'.

        Returns
        -------
        BaseParser
            An instance of the appropriate parser.

        Raises
        ------
        ValueError
            If no parser is found for the content type.
        """
        parser_cls = cls._parsers.get(content_type)
        if not parser_cls:
            msg = f"No parser found for content type: {content_type}"
            raise ValueError(msg)
        return parser_cls()
