"""Tests for RAG parsers."""

import pytest
from unittest.mock import patch, MagicMock
from ml.agent.rag.parsers.pdf import PdfParser
from ml.agent.rag.parsers.docx import DocxParser
from ml.agent.rag.parsers.pptx import PptxParser
from ml.agent.rag.parsers.txt import TxtParser
from ml.agent.rag.parsers.url import UrlParser
from ml.agent.rag.parsers.factory import ParserFactory


@pytest.fixture
def mock_pdf_reader():
    """Mock PdfReader."""
    with patch("ml.agent.rag.parsers.pdf.PdfReader") as mock:
        yield mock


@pytest.fixture
def mock_docx_document():
    """Mock docx Document."""
    with patch("ml.agent.rag.parsers.docx.Document") as mock:
        yield mock


@pytest.fixture
def mock_pptx_presentation():
    """Mock pptx Presentation."""
    with patch("ml.agent.rag.parsers.pptx.Presentation") as mock:
        yield mock


@pytest.mark.asyncio
async def test_pdf_parser(mock_pdf_reader):
    """Test PDF parser."""
    # Setup mock
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Content"
    mock_pdf_reader.return_value.pages = [mock_page]

    parser = PdfParser()
    content = b"%PDF-1.4..."
    text = await parser.parse(content)

    assert text == "PDF Content"


@pytest.mark.asyncio
async def test_docx_parser(mock_docx_document):
    """Test DOCX parser."""
    # Setup mock
    mock_para = MagicMock()
    mock_para.text = "DOCX Content"
    mock_docx_document.return_value.paragraphs = [mock_para]

    parser = DocxParser()
    content = b"PK..."
    text = await parser.parse(content)

    assert text == "DOCX Content"


@pytest.mark.asyncio
async def test_pptx_parser(mock_pptx_presentation):
    """Test PPTX parser."""
    # Setup mock
    mock_shape = MagicMock()
    mock_shape.text = "PPTX Content"
    mock_slide = MagicMock()
    mock_slide.shapes = [mock_shape]
    mock_pptx_presentation.return_value.slides = [mock_slide]

    parser = PptxParser()
    content = b"PK..."
    text = await parser.parse(content)

    assert text == "PPTX Content"


@pytest.mark.asyncio
async def test_txt_parser():
    """Test TXT parser."""
    parser = TxtParser()
    content = b"TXT Content"
    text = await parser.parse(content)

    assert text == "TXT Content"


@pytest.mark.asyncio
async def test_url_parser():
    """Test URL parser."""
    # Setup mock
    parser = UrlParser()
    text = await parser.parse("https://example.com")

    assert text == "Example DomainExample DomainThis domain is for use in documentation examples without needing permission. Avoid use in operations.Learn more"


def test_parser_factory():
    """Test ParserFactory."""
    assert isinstance(
        ParserFactory.get_parser("application/pdf"),
        PdfParser
    )
    assert isinstance(
        ParserFactory.get_parser("text/plain"),
        TxtParser
    )

    with pytest.raises(ValueError):
        ParserFactory.get_parser("invalid/type")
