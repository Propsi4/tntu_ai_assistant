"""Tests for chat history routes."""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from ml.main import app
from langchain_core.messages import HumanMessage, AIMessage

client = TestClient(app)


@pytest.fixture
def mock_history_manager():
    """Mock the history manager."""
    with patch("ml.routes.chat_history_routes.history_manager") as mock:
        yield mock


def test_get_messages_for_session(mock_history_manager):
    """Test retrieving messages for a session."""
    # Setup mock return value
    mock_messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there")
    ]
    mock_history_manager.get_messages_for_session = AsyncMock(
        return_value=mock_messages
    )

    response = client.get("/api/v1/chat-history/messages/session123")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "session123"
    assert data["total_messages"] == 2
    assert len(data["messages"]) == 2
    assert data["messages"][0]["content"] == "Hello"
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["content"] == "Hi there"
    assert data["messages"][1]["role"] == "assistant"


def test_get_messages_error(mock_history_manager):
    """Test error handling when retrieving messages."""
    mock_history_manager.get_messages_for_session = AsyncMock(
        side_effect=Exception("DB Error")
    )

    response = client.get("/api/v1/chat-history/messages/session123")

    assert response.status_code == 500
    assert "Failed to retrieve messages" in response.json()["detail"]


def test_delete_session_history_success(mock_history_manager):
    """Test successful deletion of session history."""
    mock_history_manager.clear_session = AsyncMock(return_value=True)

    response = client.delete("/api/v1/chat-history/delete/session123")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_history_manager.clear_session.assert_called_once_with("session123")


def test_delete_session_history_failure(mock_history_manager):
    """Test failure when deleting session history."""
    mock_history_manager.clear_session = AsyncMock(return_value=False)

    response = client.delete("/api/v1/chat-history/delete/session123")

    assert response.status_code == 500
    assert response.json()["status"] == "error"


def test_delete_session_history_exception(mock_history_manager):
    """Test exception handling when deleting session history."""
    mock_history_manager.clear_session = AsyncMock(
        side_effect=Exception("Delete failed")
    )

    response = client.delete("/api/v1/chat-history/delete/session123")

    assert response.status_code == 500
    assert "Failed to delete history" in response.json()["detail"]
