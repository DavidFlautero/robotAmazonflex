import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from evasion.advanced_ml_evasion import AdvancedBehavioralMLDetector, EvasionStrategy

@pytest.fixture
def evasion_system():
    return AdvancedBehavioralMLDetector(model_dir='/tmp/test_models')

@pytest.mark.asyncio
async def test_behavior_analysis_normal(evasion_system):
    """Test behavior analysis with normal patterns"""
    session_data = {
        'click_intervals_std': 2.1,
        'mouse_velocity': 230,
        'scroll_variance': 450,
        'request_timing': 1.5,
        'action_entropy': 3.2,
        'location': 'US'
    }
    
    score, strategies = await evasion_system.analyze_behavior_pattern(session_data)
    
    assert isinstance(score, float)
    assert isinstance(strategies, list)
    assert all(isinstance(s, EvasionStrategy) for s in strategies)

@pytest.mark.asyncio
async def test_behavior_analysis_anomalous(evasion_system):
    """Test behavior analysis with anomalous patterns"""
    session_data = {
        'click_intervals_std': 15.8,  # Very high (anomalous)
        'mouse_velocity': 800,        # Very high
        'scroll_variance': 2000,      # Very high
        'request_timing': 0.1,        # Very low (too fast)
        'action_entropy': 0.5,        # Very low (too predictable)
        'location': 'US'
    }
    
    score, strategies = await evasion_system.analyze_behavior_pattern(session_data)
    
    assert score < -0.7  # Should be highly anomalous
    assert len(strategies) > 0  # Should suggest evasion strategies

@pytest.mark.asyncio
async def test_evasion_strategy_selection(evasion_system):
    """Test appropriate evasion strategy selection"""
    # Test critical anomaly
    critical_data = {'consecutive_failures': 5, 'suspicious_patterns': ['pattern1', 'pattern2']}
    strategies = evasion_system.determine_evasion_strategies(-1.2, critical_data)
    
    assert EvasionStrategy.FULL_SESSION_RESET in strategies
    assert EvasionStrategy.CHANGE_NETWORK in strategies
    assert EvasionStrategy.ROTATE_FINGERPRINT in strategies

    # Test moderate anomaly
    moderate_data = {'consecutive_failures': 2, 'suspicious_patterns': []}
    strategies = evasion_system.determine_evasion_strategies(-0.6, moderate_data)
    
    assert EvasionStrategy.ROTATE_FINGERPRINT in strategies
    assert EvasionStrategy.FULL_SESSION_RESET not in strategies

@pytest.mark.asyncio
async def test_model_training_and_update(evasion_system):
    """Test model training and update functionality"""
    # Add enough data to trigger model update
    for i in range(105):  # More than retrain interval
        session_data = {
            'click_intervals_std': 2.0 + (i * 0.1),
            'mouse_velocity': 200 + (i * 5),
            'scroll_variance': 400 + (i * 10),
            'request_timing': 1.2 + (i * 0.01),
            'action_entropy': 3.0 + (i * 0.05),
            'location': 'US'
        }
        
        features = evasion_system.extract_and_validate_features(session_data)
        normalized = evasion_system.normalize_features(features)
        evasion_system.online_learning_data.append(normalized)
    
    # This should trigger model update
    await evasion_system.update_models()
    
    assert len(evasion_system.online_learning_data) == 0  # Should be cleared after update

def test_feature_validation(evasion_system):
    """Test feature validation and clamping"""
    session_data = {
        'click_intervals_std': 50.0,  # Outside bounds (0-10)
        'mouse_velocity': -100,       # Outside bounds (0-500)
        'scroll_variance': 5000,      # Outside bounds (0-1000)
        'request_timing': 10.0,       # Outside bounds (0-5)
        'action_entropy': 20.0,       # Outside bounds (0-10)
    }
    
    features = evasion_system.extract_and_validate_features(session_data)
    
    # All features should be clamped to valid ranges
    assert 0 <= features[0][0] <= 10   # click_intervals_std
    assert 0 <= features[0][1] <= 500  # mouse_velocity
    assert 0 <= features[0][2] <= 1000 # scroll_variance
    assert 0 <= features[0][3] <= 5    # request_timing
    assert 0 <= features[0][4] <= 10   # action_entropy

@pytest.mark.asyncio
async def test_feature_drift_detection(evasion_system):
    """Test feature drift detection"""
    # Create features that should trigger drift detection
    drift_features = evasion_system.extract_and_validate_features({
        'click_intervals_std': 8.0,
        'mouse_velocity': 450,
        'scroll_variance': 900,
        'request_timing': 4.5,
        'action_entropy': 9.0
    })
    
    # Mock the drift detector to return drift detected
    evasion_system.drift_detector.predict = MagicMock(return_value={'data': {'is_drift': 1}})
    
    has_drift = evasion_system.detect_feature_drift(drift_features)
    assert has_drift == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])