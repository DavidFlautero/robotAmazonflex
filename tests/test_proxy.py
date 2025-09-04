import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from network.ai_proxy_manager import AdvancedAIProxyManager, ProxyHealth, ProxyType

@pytest.fixture
def proxy_manager():
    return AdvancedAIProxyManager()

@pytest.mark.asyncio
async def test_proxy_selection(proxy_manager):
    """Test intelligent proxy selection"""
    # Add some test proxies
    proxy_manager.proxy_pool = {
        'proxy1': {'id': 'proxy1', 'type': ProxyType.RESIDENTIAL, 'url': 'http://proxy1:8080'},
        'proxy2': {'id': 'proxy2', 'type': ProxyType.DATACENTER, 'url': 'http://proxy2:8080'},
        'proxy3': {'id': 'proxy3', 'type': ProxyType.MOBILE, 'url': 'http://proxy3:8080'}
    }
    
    proxy_manager.performance_metrics = {
        'proxy1': MagicMock(success_rate=0.95, response_time=800, health=ProxyHealth.HEALTHY, consecutive_failures=0),
        'proxy2': MagicMock(success_rate=0.75, response_time=1200, health=ProxyHealth.DEGRADED, consecutive_failures=1),
        'proxy3': MagicMock(success_rate=0.99, response_time=500, health=ProxyHealth.HEALTHY, consecutive_failures=0)
    }
    
    user_context = {
        'priority': 1,
        'required_stealth_level': 2,
        'location': 'US',
        'session_duration': 3600,
        'previous_attempts': 0
    }
    
    proxy = await proxy_manager.get_optimal_proxy("https://flex.amazon.com", user_context)
    
    assert proxy is not None
    assert proxy['id'] == 'proxy3'  # Should select the best performing proxy

@pytest.mark.asyncio
async def test_proxy_health_check(proxy_manager):
    """Test proxy health checking"""
    proxy_manager.proxy_pool = {
        'test_proxy': {'id': 'test_proxy', 'url': 'http://test-proxy:8080'}
    }
    
    # Mock the session to simulate successful health checks
    mock_response = MagicMock()
    mock_response.status = 200
    proxy_manager.session.get = AsyncMock(return_value=mock_response)
    
    await proxy_manager.check_proxy_health('test_proxy')
    
    assert 'test_proxy' in proxy_manager.performance_metrics
    assert proxy_manager.performance_metrics['test_proxy'].success_rate > 0

@pytest.mark.asyncio
async def test_proxy_performance_tracking(proxy_manager):
    """Test proxy performance tracking and updates"""
    proxy_id = 'test_proxy'
    
    # Initial request success
    proxy_manager.update_proxy_performance(proxy_id, True, 800)
    perf = proxy_manager.performance_metrics[proxy_id]
    assert perf.success_rate == 1.0
    assert perf.response_time == 800
    assert perf.health == ProxyHealth.HEALTHY
    
    # Subsequent failure
    proxy_manager.update_proxy_performance(proxy_id, False, 2000)
    perf = proxy_manager.performance_metrics[proxy_id]
    assert perf.success_rate < 1.0  # Should be decreased
    assert perf.consecutive_failures == 1

@pytest.mark.asyncio
async def test_proxy_availability(proxy_manager):
    """Test proxy availability checking"""
    proxy_manager.performance_metrics = {
        'healthy_proxy': MagicMock(health=ProxyHealth.HEALTHY, consecutive_failures=0),
        'unhealthy_proxy': MagicMock(health=ProxyHealth.UNHEALTHY, consecutive_failures=5),
        'degraded_proxy': MagicMock(health=ProxyHealth.DEGRADED, consecutive_failures=2)
    }
    
    assert proxy_manager.is_proxy_available('healthy_proxy') == True
    assert proxy_manager.is_proxy_available('unhealthy_proxy') == False
    assert proxy_manager.is_proxy_available('degraded_proxy') == False  # Should be unavailable due to recent degradation

def test_health_status_determination(proxy_manager):
    """Test health status determination logic"""
    config = proxy_manager.config
    
    # Test healthy proxy
    healthy_perf = MagicMock(success_rate=config.min_success_rate + 0.1, 
                           response_time=config.max_response_time - 100,
                           consecutive_failures=0)
    assert proxy_manager.determine_health_status(healthy_perf) == ProxyHealth.HEALTHY
    
    # Test degraded proxy
    degraded_perf = MagicMock(success_rate=config.min_success_rate - 0.05,
                            response_time=config.max_response_time * 1.5,
                            consecutive_failures=2)
    assert proxy_manager.determine_health_status(degraded_perf) == ProxyHealth.DEGRADED
    
    # Test unhealthy proxy
    unhealthy_perf = MagicMock(success_rate=config.min_success_rate * 0.4,
                              response_time=config.max_response_time * 3,
                              consecutive_failures=1)
    assert proxy_manager.determine_health_status(unhealthy_perf) == ProxyHealth.UNHEALTHY
    
    # Test dead proxy
    dead_perf = MagicMock(success_rate=config.min_success_rate * 0.2,
                         response_time=config.max_response_time * 4,
                         consecutive_failures=10)
    assert proxy_manager.determine_health_status(dead_perf) == ProxyHealth.DEAD

@pytest.mark.asyncio
async def test_rl_model_training(proxy_manager):
    """Test reinforcement learning model training"""
    # Mock the model and optimizer
    proxy_manager.rl_model = MagicMock()
    proxy_manager.optimizer = MagicMock()
    proxy_manager.loss_fn = MagicMock(return_value=MagicMock())
    
    # Test successful request
    proxy_manager.train_rl_model('test_proxy', True, 500)
    assert proxy_manager.optimizer.zero_grad.called
    assert proxy_manager.optimizer.step.called
    
    # Test failed request
    proxy_manager.train_rl_model('test_proxy', False, 3000)
    assert proxy_manager.optimizer.zero_grad.call_count == 2
    assert proxy_manager.optimizer.step.call_count == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])