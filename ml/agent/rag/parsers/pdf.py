"""PDF parser implementation using pypdf."""

import io
from typing import Union
from pypdf import PdfReader
from ml.agent.rag.parsers.base import BaseParser


class PdfParser(BaseParser):
    """Parser for PDF documents."""

    async def parse(self, source: Union[str, bytes]) -> str:
        """
        Parse PDF content and extract text.

        Parameters
        ----------
        source : Union[str, bytes]
            The PDF content as bytes or file path.

        Returns
        -------
        str
            Extracted text from the PDF.
        """
        if isinstance(source, bytes):
            pdf_file = io.BytesIO(source)
        else:
            # If it's a string path, we read it as bytes.
            # However, typical usage via UploadFile provides bytes.
            # For file paths, pypdf can handle them directly but we wrap in
            # BytesIO for consistency if needed or pass path directly if
            # pypdf supports it (it does).
            pdf_file = source  # type: ignore

        reader = PdfReader(pdf_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)

        return "\n".join(text)
