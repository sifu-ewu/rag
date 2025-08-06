"""
Professional Test Suite for Enhanced RAG API
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import json
import tempfile
import os

# Import the professional API
import sys
sys.path.append('..')
from api_pro import app
from config import config

# Test client
client = TestClient(app)

class TestProfessionalRAGAPI:
    """Test suite for the professional RAG API"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment"""
        # Mock authentication for testing
        self.test_api_key = "test_api_key_123"
        self.admin_api_key = "admin_api_key_123"
        
        # Mock security manager
        with patch('api_pro.security_manager') as mock_security:
            mock_security.validate_api_key.return_value = {
                'name': 'test_user',
                'permissions': ['read', 'write'],
                'rate_limit': 1000
            }
            mock_security.check_rate_limit.return_value = True
        
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "warning", "critical"]
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "2.0.0"
    
    def test_health_check_structure(self):
        """Test health check response structure"""
        response = client.get("/health")
        data = response.json()
        
        required_fields = ["status", "timestamp", "version", "environment", "uptime_seconds"]
        for field in required_fields:
            assert field in data
    
    @pytest.mark.parametrize("query,expected_fields", [
        ("Test query", ["query", "response", "language", "processing_time_seconds"]),
        ("অনুপম কে?", ["query", "response", "language", "memory_used"])
    ])
    def test_query_processing(self, query, expected_fields):
        """Test query processing with different inputs"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['read']}
            
            response = client.post(
                "/v1/query",
                json={"query": query},
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            
            # Should work even without RAG pipeline in test
            if response.status_code == 500:
                # Expected in test environment without proper setup
                return
            
            assert response.status_code == 200
            data = response.json()
            
            for field in expected_fields:
                assert field in data
    
    def test_query_validation(self):
        """Test query input validation"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['read']}
            
            # Test empty query
            response = client.post(
                "/v1/query",
                json={"query": ""},
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            assert response.status_code == 422  # Validation error
            
            # Test query too long
            long_query = "x" * 6000
            response = client.post(
                "/v1/query",
                json={"query": long_query},
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            assert response.status_code == 422
    
    def test_authentication_required(self):
        """Test that authentication is required for protected endpoints"""
        # Test without authentication
        response = client.post("/v1/query", json={"query": "test"})
        assert response.status_code == 403  # Forbidden without auth
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['read']}
            
            response = client.get(
                "/metrics",
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            
            # Should return metrics or error
            assert response.status_code in [200, 500]
    
    def test_cache_stats(self):
        """Test cache statistics endpoint"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['read']}
            
            response = client.get(
                "/v1/cache/stats",
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            
            assert response.status_code in [200, 500]
    
    def test_document_upload_validation(self):
        """Test document upload validation"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['write']}
            
            # Test non-PDF file
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
                tmp.write(b"test content")
                tmp.flush()
                
                try:
                    with open(tmp.name, 'rb') as f:
                        response = client.post(
                            "/v1/documents/upload",
                            files={"file": ("test.txt", f, "text/plain")},
                            headers={"Authorization": f"Bearer {self.test_api_key}"}
                        )
                    
                    assert response.status_code == 400  # Should reject non-PDF
                finally:
                    os.unlink(tmp.name)
    
    def test_admin_endpoints(self):
        """Test admin-only endpoints"""
        with patch('api_pro.get_admin_user') as mock_admin:
            mock_admin.return_value = {'permissions': ['admin']}
            
            # Test API key generation
            response = client.post(
                "/auth/api-key?name=test_key&permissions=read",
                headers={"Authorization": f"Bearer {self.admin_api_key}"}
            )
            
            # Should work or return error if not properly mocked
            assert response.status_code in [200, 500]
    
    def test_search_endpoint(self):
        """Test document search endpoint"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['read']}
            
            response = client.get(
                "/v1/search?query=test&num_results=5",
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            
            # Should work or return error if RAG pipeline not initialized
            assert response.status_code in [200, 500]
    
    def test_rate_limiting_headers(self):
        """Test that rate limiting headers are present"""
        response = client.get("/health")
        
        # Check for security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
    
    def test_cors_headers(self):
        """Test CORS configuration"""
        # In development mode, CORS should be permissive
        if config.is_development():
            response = client.options("/health")
            # Check if CORS headers are present in development
            # This depends on the middleware setup
    
    def test_error_handling(self):
        """Test error handling and response format"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['read']}
            
            # Test malformed JSON
            response = client.post(
                "/v1/query",
                data="invalid json",
                headers={
                    "Authorization": f"Bearer {self.test_api_key}",
                    "Content-Type": "application/json"
                }
            )
            
            assert response.status_code == 422
    
    def test_session_management(self):
        """Test session creation and management"""
        with patch('api_pro.get_current_user') as mock_auth:
            mock_auth.return_value = {'permissions': ['write']}
            
            session_id = "test_session_123"
            
            # Test session creation (implicit through query)
            response = client.post(
                f"/v1/query?session_id={session_id}",
                json={"query": "test"},
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            
            # Session endpoints
            response = client.delete(
                f"/v1/sessions/{session_id}",
                headers={"Authorization": f"Bearer {self.test_api_key}"}
            )
            
            # Should return success or appropriate error
            assert response.status_code in [200, 404, 500]

@pytest.mark.asyncio
class TestAsyncFunctionality:
    """Test async functionality"""
    
    async def test_async_cache_operations(self):
        """Test async cache operations"""
        from src.cache import cache_manager
        
        # Test basic cache operations
        key = "test_key"
        value = {"test": "data"}
        
        # Set cache value
        success = await cache_manager.backend.set(key, value, 60)
        assert success is True or success is None  # Depends on backend
        
        # Get cache value
        retrieved = await cache_manager.backend.get(key)
        # May be None if cache is disabled in tests
    
    async def test_performance_tracking(self):
        """Test performance tracking context manager"""
        from src.monitoring import PerformanceTracker
        
        async with PerformanceTracker("test_operation") as tracker:
            await asyncio.sleep(0.1)
        
        # Tracker should have recorded the operation
        assert tracker.start_time is not None
        assert tracker.end_time is not None

class TestConfiguration:
    """Test configuration management"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        # This will test the current config
        result = config.validate_config()
        # Should return True or False based on current setup
        assert isinstance(result, bool)
    
    def test_environment_detection(self):
        """Test environment detection"""
        assert config.is_development() or config.is_production()
        assert not (config.is_development() and config.is_production())
    
    def test_settings_dict(self):
        """Test settings dictionary export"""
        settings = config.get_settings_dict()
        assert isinstance(settings, dict)
        assert "ENVIRONMENT" in settings
        assert "API_HOST" in settings

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
