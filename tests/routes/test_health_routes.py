"""Tests for health check routes."""

from fastapi.testclient import TestClient
from ml.main import app

client = TestClient(app)


def test_health_check():
    """Test the global health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_agent_health():
    """Test the agent health check endpoint."""
    response = client.get("/api/v1/agent/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
