"""DOCX parser implementation using python-docx."""

import io
from typing import Union
from docx import Document
from ml.agent.rag.parsers.base import BaseParser


class DocxParser(BaseParser):
    """Parser for DOCX documents."""

    async def parse(self, source: Union[str, bytes]) -> str:
        """
        Parse DOCX content and extract text.

        Parameters
        ----------
        source : Union[str, bytes]
            The DOCX content as bytes or file path.

        Returns
        -------
        str
            Extracted text from the DOCX.
        """
        if isinstance(source, bytes):
            docx_file = io.BytesIO(source)
        else:
            docx_file = source  # type: ignore

        doc = Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
