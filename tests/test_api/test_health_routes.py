# Test health routes module - unit tests for the health check endpoint.
# Tests API status reporting and response format.

import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.main import app


class TestHealthRoutes:
    """Tests for the health check API endpoint."""

    def setup_method(self):
        """Set up test client for each test method."""
        self.client = TestClient(app)

    @patch("src.api.routes.health.VectorStore")
    @patch("src.api.routes.health.LeadService")
    def test_health_check_returns_200(self, mock_lead_service_cls, mock_vector_store_cls):
        """Test that health check returns 200 with correct structure."""
        # Configure mocks
        mock_vector_store_cls.return_value.get_document_count.return_value = 42
        mock_lead_service_cls.return_value.get_lead_count.return_value = 5

        # Make the health check request
        response = self.client.get("/api/health")

        # Verify successful response
        assert response.status_code == 200

        # Verify response structure
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "vector_store_count" in data
        assert "leads_count" in data

    @patch("src.api.routes.health.VectorStore")
    @patch("src.api.routes.health.LeadService")
    def test_health_check_includes_metrics(self, mock_lead_service_cls, mock_vector_store_cls):
        """Test that health response includes vector store and leads counts."""
        # Configure specific counts
        mock_vector_store_cls.return_value.get_document_count.return_value = 100
        mock_lead_service_cls.return_value.get_lead_count.return_value = 15

        response = self.client.get("/api/health")
        data = response.json()

        # Verify metrics are present
        assert data["vector_store_count"] == 100
        assert data["leads_count"] == 15
