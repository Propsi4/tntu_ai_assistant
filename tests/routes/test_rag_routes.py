"""Tests for RAG routes."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from ml.main import app
from ml.agent.rag.schemas import DocumentResponse

client = TestClient(app)


@pytest.fixture
def mock_rag_service():
    """Mock the RAG service."""
    with patch(
        "ml.routes.rag_routes.rag_service", new_callable=AsyncMock
    ) as mock:
        yield mock


def test_index_document(mock_rag_service):
    """Test indexing a document."""
    # Mock return value
    mock_doc = DocumentResponse(
        id=1,
        content="Test content",
        metadata={"source": "test"},
        created_at="2023-01-01T00:00:00Z"
    )
    mock_rag_service.index_document.return_value = mock_doc

    response = client.post(
        "/api/v1/rag/index",
        json={"content": "Test content", "metadata": {"source": "test"}}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["id"] == 1
    assert data["content"] == "Test content"

    mock_rag_service.index_document.assert_called_once_with(
        "Test content", {"source": "test"}
    )


def test_upload_document(mock_rag_service):
    """Test uploading a document."""
    # Mock return value
    mock_doc = DocumentResponse(
        id=1,
        content="Parsed content",
        metadata={"source": "test.txt"},
        created_at="2023-01-01T00:00:00Z"
    )
    mock_rag_service.process_and_index_document.return_value = [mock_doc]

    files = {'file': ('test.txt', b'test content', 'text/plain')}
    response = client.post("/api/v1/rag/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Parsed content"

    mock_rag_service.process_and_index_document.assert_called_once()


def test_index_url(mock_rag_service):
    """Test indexing a URL."""
    # Mock return value
    mock_doc = DocumentResponse(
        id=1,
        content="URL content",
        metadata={"source": "http://example.com"},
        created_at="2023-01-01T00:00:00Z"
    )
    mock_rag_service.process_and_index_url.return_value = [mock_doc]

    response = client.post(
        "/api/v1/rag/index-url",
        json={"url": "http://example.com"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "URL content"

    mock_rag_service.process_and_index_url.assert_called_once_with(
        "http://example.com"
    )


def test_list_documents(mock_rag_service):
    """Test listing documents."""
    # Mock return value
    mock_docs = [
        DocumentResponse(
            id=1,
            content="Doc 1",
            metadata={},
            created_at="2023-01-01T00:00:00Z"
        ),
        DocumentResponse(
            id=2,
            content="Doc 2",
            metadata={},
            created_at="2023-01-01T00:00:00Z"
        )
    ]

    mock_rag_service.list_documents.return_value = mock_docs
    mock_rag_service.count_documents.return_value = 2

    response = client.get("/api/v1/rag/documents?limit=10&offset=0")

    assert response.status_code == 200
    data = response.json()
    # Ensure schema matches DocumentListResponse
    # items, total, limit, offset
    assert len(data["items"]) == 2
    assert data["total"] == 2
    assert data["limit"] == 10
    assert data["offset"] == 0

    mock_rag_service.list_documents.assert_called_once_with(
        limit=10, offset=0
    )
    mock_rag_service.count_documents.assert_called_once()


def test_delete_document(mock_rag_service):
    """Test deleting a document."""
    mock_rag_service.deindex_document.return_value = True

    response = client.delete("/api/v1/rag/document/1")

    assert response.status_code == 200
    assert response.json()["status"] == "success"

    mock_rag_service.deindex_document.assert_called_once_with(1)


def test_delete_document_not_found(mock_rag_service):
    """Test deleting a non-existent document."""
    mock_rag_service.deindex_document.return_value = False

    response = client.delete("/api/v1/rag/document/999")

    assert response.status_code == 404
