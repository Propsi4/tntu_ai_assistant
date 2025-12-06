"""Tests for agent routes."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient
from ml.main import app
import dspy

client = TestClient(app)


@pytest.fixture
def mock_agent():
    """Mock the agent."""
    with patch("ml.routes.agent_routes.agent") as mock:
        mock.is_healthy.return_value = True
        # Setup default behavior for non-streaming call
        mock.return_value.response = "Mocked response"
        mock.return_value.trajectory = []
        yield mock


@pytest.fixture
def mock_history_manager():
    """Mock the history manager."""
    with patch("ml.routes.agent_routes.history_manager") as mock:
        mock.get_messages_for_session = AsyncMock(return_value=[])
        mock.save_messages = AsyncMock()
        yield mock


@pytest.fixture
def mock_dspy_streamify():
    """Mock dspy.streamify."""
    with patch("dspy.streamify") as mock:
        yield mock


def test_agent_chat_validation_error(mock_agent, mock_history_manager):
    """Test validation error for agent chat."""
    # Missing session_id
    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "Hello"}
    )
    assert response.status_code == 422

    # Missing message
    response = client.post(
        "/api/v1/agent/chat",
        json={"session_id": "123"}
    )
    assert response.status_code == 422


def test_agent_chat_success(mock_agent, mock_history_manager):
    """Test successful agent chat."""
    # Setup mock return
    mock_prediction = MagicMock()
    mock_prediction.response = "Hello there!"
    mock_prediction.trajectory = {}
    mock_agent.return_value = mock_prediction

    response = client.post(
        "/api/v1/agent/chat",
        json={"message": "Hello", "session_id": "123"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Hello there!"
    assert data["session_id"] == "123"
    assert data["status"] == "success"


def test_chat_stream(mock_agent, mock_history_manager, mock_dspy_streamify):
    """Test streaming agent chat."""
    # Mock streamify to return a callable that returns an async generator
    async def async_gen():
        # Yield token chunks
        chunk1 = MagicMock(spec=dspy.streaming.StreamResponse)
        chunk1.chunk = "Hello"
        yield chunk1

        chunk2 = MagicMock(spec=dspy.streaming.StreamResponse)
        chunk2.chunk = " world"
        yield chunk2

        # Yield final prediction
        prediction = MagicMock(spec=dspy.Prediction)
        prediction.response = "Hello world"
        prediction.trajectory = {}
        yield prediction

    mock_stream_agent = MagicMock()
    mock_stream_agent.return_value = async_gen()
    mock_dspy_streamify.return_value = mock_stream_agent

    response = client.post(
        "/api/v1/agent/chat/stream",
        json={"message": "Hello", "session_id": "123"}
    )

    assert response.status_code == 200

    # Verify streaming content
    content = response.text
    assert "data:" in content
    assert "Hello" in content
    assert "world" in content
    assert "complete" in content
