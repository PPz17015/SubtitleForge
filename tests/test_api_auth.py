"""
Integration tests for API Authentication.

Coverage:
- verify_api_key correctly validates against API_KEY env var
- Rejects invalid/missing keys when API_KEY is configured
- Allows all requests when no API_KEY is configured
- Protected endpoints return 401 for unauthorized requests
- Health endpoint is accessible without authentication
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient

from api.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test the health check endpoint (no auth required)."""

    def test_health_check_accessible(self, client):
        """Health endpoint should be accessible without API key."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "gpu_available" in data

    def test_health_check_returns_version(self, client):
        """Health endpoint returns current version."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "1.0.0"


class TestSettingsEndpoint:
    """Test the settings endpoint (no auth required)."""

    def test_settings_accessible(self, client):
        """Settings endpoint should be accessible."""
        response = client.get("/api/v1/settings")
        assert response.status_code == 200

        data = response.json()
        assert "default_source_lang" in data
        assert "default_target_lang" in data
        assert data["default_source_lang"] == "ja"
        assert data["default_target_lang"] == "vi"


class TestAPIKeyValidation:
    """Test API key validation logic."""

    def test_no_api_key_configured_allows_all(self, client):
        """When API_KEY env var is not set, all requests should be allowed."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove API_KEY if it exists
            os.environ.pop("API_KEY", None)

            response = client.post(
                "/api/v1/subtitle",
                json={
                    "video_path": "/nonexistent/video.mp4",
                    "source_language": "ja",
                    "target_language": "vi"
                }
            )
            # Should not be 401 (may be other errors)
            assert response.status_code != 401

    def test_valid_api_key_accepted(self, client):
        """Valid API key should be accepted."""
        test_key = "test-secret-key-12345"

        with patch.dict(os.environ, {"API_KEY": test_key}):
            response = client.post(
                "/api/v1/subtitle",
                json={
                    "video_path": "/nonexistent/video.mp4",
                    "source_language": "ja",
                    "target_language": "vi"
                },
                params={"x_api_key": test_key}
            )
            # Should not be 401 (may fail for other reasons like file not found)
            assert response.status_code != 401

    def test_invalid_api_key_rejected(self, client):
        """Invalid API key should be rejected with 401."""
        with patch.dict(os.environ, {"API_KEY": "correct-key"}):
            response = client.post(
                "/api/v1/subtitle",
                json={
                    "video_path": "/some/video.mp4",
                    "source_language": "ja",
                    "target_language": "vi"
                },
                params={"x_api_key": "wrong-key"}
            )
            assert response.status_code == 401
            assert "Invalid" in response.json()["detail"]

    def test_missing_api_key_rejected(self, client):
        """Missing API key should be rejected when API_KEY is configured."""
        with patch.dict(os.environ, {"API_KEY": "configured-key"}):
            response = client.post(
                "/api/v1/subtitle",
                json={
                    "video_path": "/some/video.mp4",
                    "source_language": "ja",
                    "target_language": "vi"
                }
                # No x_api_key param
            )
            assert response.status_code == 401

    def test_empty_api_key_rejected(self, client):
        """Empty API key should be rejected when API_KEY is configured."""
        with patch.dict(os.environ, {"API_KEY": "configured-key"}):
            response = client.post(
                "/api/v1/subtitle",
                json={
                    "video_path": "/some/video.mp4",
                    "source_language": "ja",
                    "target_language": "vi"
                },
                params={"x_api_key": ""}
            )
            assert response.status_code == 401


class TestProtectedEndpoints:
    """Test that protected endpoints enforce authentication."""

    @pytest.fixture(autouse=True)
    def setup_api_key(self):
        """Set up API key for all tests in this class."""
        with patch.dict(os.environ, {"API_KEY": "test-protected-key"}):
            yield

    def test_create_job_requires_auth(self, client):
        """POST /subtitle requires authentication."""
        response = client.post(
            "/api/v1/subtitle",
            json={
                "video_path": "/video.mp4",
                "source_language": "ja",
                "target_language": "vi"
            }
        )
        assert response.status_code == 401

    def test_create_job_with_auth(self, client):
        """POST /subtitle with valid key should not return 401."""
        response = client.post(
            "/api/v1/subtitle",
            json={
                "video_path": "/nonexistent/video.mp4",
                "source_language": "ja",
                "target_language": "vi"
            },
            params={"x_api_key": "test-protected-key"}
        )
        # Should pass auth (may fail for other reasons like file not found)
        assert response.status_code != 401

    def test_job_status_accessible_without_auth(self, client):
        """GET /subtitle/{job_id} should be accessible (it doesn't use Depends)."""
        response = client.get("/api/v1/subtitle/nonexistent-job")
        # Should return 404, not 401
        assert response.status_code == 404

    def test_delete_job_requires_auth(self, client):
        """DELETE /subtitle/{job_id} requires authentication."""
        response = client.delete("/api/v1/subtitle/some-job-id")
        assert response.status_code == 401


class TestOutputFormatParameter:
    """Test that the renamed output_format parameter works correctly."""

    def test_download_nonexistent_job(self, client):
        """Test download endpoint with nonexistent job returns 404."""
        response = client.get("/api/v1/subtitle/nonexistent-id/download")
        assert response.status_code == 404

    def test_download_with_format_param(self, client):
        """
        Test that the output_format parameter is properly accepted.

        This verifies the fix where `format` (Python built-in shadow)
        was renamed to `output_format`.
        """
        response = client.get(
            "/api/v1/subtitle/nonexistent-id/download",
            params={"output_format": "vtt"}
        )
        # Should be 404 (job not found), not 422 (validation error)
        assert response.status_code == 404


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_app_info(self, client):
        """Root endpoint returns app info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "SubtitleForge Pro API"
        assert data["version"] == "1.0.0"
        assert "docs" in data


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
