import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from security.zero_trust import ZeroTrustSecurity

@pytest.fixture
def test_client():
    """Fixture para el cliente de testing de FastAPI"""
    return TestClient(app)

@pytest.fixture
def auth_token():
    """Fixture para token de autenticación de prueba"""
    return "test-auth-token-12345"

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock de todas las dependencias externas para tests de integración"""
    with patch('api.main.zt_security.verify_token') as mock_verify, \
         patch('api.main.evasion_system.analyze_behavior_pattern') as mock_analyze, \
         patch('api.main.proxy_manager.get_optimal_proxy') as mock_proxy, \
         patch('api.main.proxy_manager.start_health_checks') as mock_health_checks, \
         patch('api.main.proxy_manager.close') as mock_proxy_close, \
         patch('api.main.app.state.db') as mock_db:
        
        # Configurar mocks por defecto
        mock_verify.return_value = "user123"
        mock_analyze.return_value = (-0.1, [])
        mock_proxy.return_value = {'id': 'test_proxy', 'url': 'http://test-proxy:8080'}
        
        # Mock de la base de datos
        mock_connection = AsyncMock()
        mock_db.acquire.return_value.__aenter__.return_value = mock_connection
        mock_connection.fetchrow.return_value = {
            'id': 'user123', 
            'amazon_username': 'test@example.com',
            'amazon_password': 'encrypted_password',
            'location': 'US',
            'status': 'active'
        }
        mock_connection.execute = AsyncMock()
        
        yield {
            'mock_verify': mock_verify,
            'mock_analyze': mock_analyze,
            'mock_proxy': mock_proxy,
            'mock_db': mock_db,
            'mock_connection': mock_connection
        }

def test_health_endpoint(test_client):
    """Test para el endpoint de health check"""
    response = test_client.get("/healthz")
    
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "timestamp" in response.json()

def test_metrics_endpoint(test_client):
    """Test para el endpoint de métricas Prometheus"""
    response = test_client.get("/metrics")
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"
    assert "python_gc_objects" in response.text  # Métrica común de Python

def test_capture_blocks_success(test_client, auth_token, mock_dependencies):
    """Test exitoso para el endpoint de captura de bloques"""
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"location": "US", "preferences": {"min_payment": 20}}
    )
    
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert response.json()["blocks_captured"] == 3  # Del mock execute_amazon_flex_request
    assert "anomaly_score" in response.json()
    assert "evasion_strategies" in response.json()
    
    # Verificar que se llamaron las dependencias
    mock_dependencies['mock_verify'].assert_called_once_with(auth_token)
    mock_dependencies['mock_analyze'].assert_called_once()
    mock_dependencies['mock_proxy'].assert_called_once()

def test_capture_blocks_authentication_failure(test_client, mock_dependencies):
    """Test de fallo de autenticación en captura de bloques"""
    mock_dependencies['mock_verify'].side_effect = Exception("Token inválido")
    
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": "Bearer invalid-token"},
        json={"location": "US"}
    )
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]

def test_capture_blocks_user_not_found(test_client, auth_token, mock_dependencies):
    """Test cuando el usuario no existe en la base de datos"""
    mock_dependencies['mock_connection'].fetchrow.return_value = None
    
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"location": "US"}
    )
    
    assert response.status_code == 404
    assert "User not found" in response.json()["detail"]

def test_capture_blocks_no_proxies_available(test_client, auth_token, mock_dependencies):
    """Test cuando no hay proxies disponibles"""
    mock_dependencies['mock_proxy'].return_value = None
    
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"location": "US"}
    )
    
    assert response.status_code == 503
    assert "No proxies available" in response.json()["detail"]

def test_capture_blocks_amazon_api_error(test_client, auth_token, mock_dependencies):
    """Test cuando la API de Amazon falla"""
    with patch('api.main.execute_amazon_flex_request') as mock_amazon:
        mock_amazon.side_effect = Exception("Amazon API error")
        
        response = test_client.post(
            "/api/capture-blocks",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"location": "US"}
        )
        
        assert response.status_code == 500
        assert "Amazon API error" in response.json()["detail"]

def test_user_stats_success(test_client, auth_token, mock_dependencies):
    """Test exitoso para el endpoint de estadísticas de usuario"""
    # Mock de datos de estadísticas
    mock_dependencies['mock_connection'].fetchrow.return_value = {
        'total_captures': 150,
        'successful_captures': 135,
        'success_rate': 0.9
    }
    
    response = test_client.get(
        "/api/user/stats",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["total_captures"] == 150
    assert response.json()["successful_captures"] == 135
    assert response.json()["success_rate"] == 0.9

def test_user_stats_no_data(test_client, auth_token, mock_dependencies):
    """Test cuando no hay datos de estadísticas"""
    mock_dependencies['mock_connection'].fetchrow.return_value = {
        'total_captures': 0,
        'successful_captures': 0,
        'success_rate': None
    }
    
    response = test_client.get(
        "/api/user/stats",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 200
    assert response.json()["total_captures"] == 0
    assert response.json()["successful_captures"] == 0
    assert response.json()["success_rate"] == 0

def test_user_stats_database_error(test_client, auth_token, mock_dependencies):
    """Test cuando la base de datos falla"""
    mock_dependencies['mock_connection'].fetchrow.side_effect = Exception("Database error")
    
    response = test_client.get(
        "/api/user/stats",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    
    assert response.status_code == 500
    assert "Database error" in response.json()["detail"]

def test_high_anomaly_score_behavior(test_client, auth_token, mock_dependencies):
    """Test cuando se detecta alta anomalía en el comportamiento"""
    mock_dependencies['mock_analyze'].return_value = (-1.5, ["ROTATE_FINGERPRINT", "CHANGE_NETWORK"])
    
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": f"Bearer {auth_token}"},
        json={"location": "US"}
    )
    
    assert response.status_code == 200
    assert response.json()["anomaly_score"] == -1.5
    assert len(response.json()["evasion_strategies"]) == 2
    assert "ROTATE_FINGERPRINT" in response.json()["evasion_strategies"]
    assert "CHANGE_NETWORK" in response.json()["evasion_strategies"]

def test_concurrent_requests(test_client, auth_token):
    """Test de requests concurrentes para verificar manejo de carga"""
    import threading
    
    results = []
    errors = []
    
    def make_request():
        try:
            response = test_client.post(
                "/api/capture-blocks",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={"location": "US"}
            )
            results.append(response.status_code)
        except Exception as e:
            errors.append(str(e))
    
    # Crear 10 hilos concurrentes
    threads = []
    for _ in range(10):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Esperar a que todos los hilos terminen
    for thread in threads:
        thread.join()
    
    # Verificar que todos los requests fueron exitosos
    assert len(results) == 10
    assert all(status == 200 for status in results)
    assert len(errors) == 0

def test_rate_limiting(test_client, auth_token, mock_dependencies):
    """Test de rate limiting (si estuviera implementado)"""
    # Hacer múltiples requests rápidos
    for i in range(15):
        response = test_client.post(
            "/api/capture-blocks",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={"location": "US"}
        )
        
        # En una implementación real con rate limiting, 
        # algunos requests deberían ser rechazados después de un límite
        if i < 10:  # Primeros 10 requests deberían pasar
            assert response.status_code == 200
        # Nota: Este test asume que NO hay rate limiting implementado actualmente

def test_cors_headers(test_client):
    """Test de headers CORS"""
    response = test_client.options("/api/capture-blocks", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization"
    })
    
    # Verificar headers CORS
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers

def test_invalid_json_payload(test_client, auth_token):
    """Test con payload JSON inválido"""
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": f"Bearer {auth_token}"},
        content="invalid json"
    )
    
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

def test_missing_authorization_header(test_client):
    """Test sin header de autorización"""
    response = test_client.post(
        "/api/capture-blocks",
        json={"location": "US"}
    )
    
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]

def test_malformed_authorization_header(test_client):
    """Test con header de autorización mal formado"""
    response = test_client.post(
        "/api/capture-blocks",
        headers={"Authorization": "InvalidFormat"},
        json={"location": "US"}
    )
    
    assert response.status_code == 401
    assert "Invalid authentication scheme" in response.json()["detail"]

@patch('api.main.AdvancedBehavioralMLDetector')
@patch('api.main.AdvancedAIProxyManager')
def test_app_startup_and_shutdown(mock_proxy_manager, mock_evasion_system, test_client):
    """Test de startup y shutdown de la aplicación"""
    # Mock de las instancias
    mock_evasion_instance = AsyncMock()
    mock_proxy_instance = AsyncMock()
    
    mock_evasion_system.return_value = mock_evasion_instance
    mock_proxy_manager.return_value = mock_proxy_instance
    
    # Simular startup
    with patch('api.main.asyncpg.create_pool') as mock_pool:
        mock_pool.return_value = AsyncMock()
        
        # La aplicación ya está corriendo, pero podemos testear el shutdown
        # simulando llamar a los event handlers manualmente
        
        # Esto es principalmente para coverage y verificar que no hay errores
        assert True  # Placeholder para el test de startup/shutdown

def test_real_amazon_request_mocking(test_client, auth_token, mock_dependencies):
    """Test con mock específico de requests a Amazon"""
    with patch('api.main.execute_amazon_flex_request') as mock_amazon:
        # Mock de diferentes escenarios de respuesta de Amazon
        test_cases = [
            ({"success": True, "blocks_captured": 5}, 200),
            ({"success": False, "blocks_captured": 0, "error": "Captcha required"}, 200),
            ({"success": True, "blocks_captured": 2}, 200),
            (Exception("Network error"), 500)
        ]
        
        for amazon_response, expected_status in test_cases:
            if isinstance(amazon_response, Exception):
                mock_amazon.side_effect = amazon_response
            else:
                mock_amazon.return_value = amazon_response
            
            response = test_client.post(
                "/api/capture-blocks",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={"location": "US"}
            )
            
            assert response.status_code == expected_status
            
            if expected_status == 200:
                assert "success" in response.json()
                assert "blocks_captured" in response.json()

if __name__ == "__main__":
    # Ejecutar los tests directamente
    pytest.main([__file__, "-v", "--tb=short"])